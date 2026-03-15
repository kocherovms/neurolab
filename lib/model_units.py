import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from hp_utils import *

def create_unit(params, input_size=None):
    if isinstance(params, LinearModelUnitParams):
        assert input_size is not None
        assert input_size > 0
        return LinearModelUnit(input_size, params)
    elif isinstance(params, Conv2dModelUnitParams):
        return Conv2dModelUnit(params)
    elif isinstance(params, StateModelUnitParams):
        assert input_size is not None
        assert input_size > 0
        return StateModelUnit(input_size, params)
    else:
        assert False, f'Unsupported {type(params)=}'

class LinearModelUnit(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()
        
        if params.dropout is None:
            self.dropout = lambda i: i
        else:
            self.dropout = nn.Dropout(*LangUtils.coalesce(params.dropout.args, ()), **LangUtils.coalesce(params.dropout.kwargs, {}))
            
        self.linear = nn.Linear(
            in_features=input_size, 
            out_features=params.size, 
            bias=True
        )

        if params.nonlinearity is None:
            self.nonlinearity = lambda i: i
        else:
            self.nonlinearity = getattr(nn, params.nonlinearity.module)(*LangUtils.coalesce(params.nonlinearity.args, ()), **LangUtils.coalesce(params.nonlinearity.kwargs, {}))

        self.output_size = self.linear.out_features

    def forward(self, inp): 
        res = self.dropout(inp)
        res = self.linear(res)
        res = self.nonlinearity(res)
        return res

class Conv2dModelUnit(nn.Module):
    def __init__(self, params):
        super().__init__()
        convolution = params.convolution
        groups_count = convolution.in_channels_count // convolution.in_channels_count_per_kernel
        conv_kwargs = {}

        if convolution.padding is not None:
            conv_kwargs['padding'] = convolution.padding

        if convolution.stride is not None:
            conv_kwargs['stride'] = convolution.stride
        
        self.conv = getattr(nn, params.module)(
            in_channels=convolution.in_channels_count, 
            out_channels=convolution.out_channels_count, 
            kernel_size=convolution.kernel_size, 
            bias=convolution.with_bias,
            groups=groups_count,
            **conv_kwargs
        )

        if params.batch_norm2d is None:
            self.batch_norm2d = lambda i: i
        else:
            self.batch_norm2d = nn.BatchNorm2d(*LangUtils.coalesce(params.batch_norm2d.args, ()), **LangUtils.coalesce(params.batch_norm2d.kwargs, {}))

        if params.instance_norm2d is None:
            self.instance_norm2d = lambda i: i
        else:
            self.instance_norm2d = nn.InstanceNorm2d(*LangUtils.coalesce(params.instance_norm2d.args, ()), **LangUtils.coalesce(params.instance_norm2d.kwargs, {}))

        if params.nonlinearity is None:
            self.nonlinearity = lambda i: i
        else:
            self.nonlinearity = getattr(nn, params.nonlinearity.module)(*LangUtils.coalesce(params.nonlinearity.args, ()), **LangUtils.coalesce(params.nonlinearity.kwargs, {}))

        if not params.with_gain:
            self.gain = lambda i: i
        else:
            self.gain_params = nn.Parameter(torch.ones(1, 1, convolution.out_channels_count)) 
            self.gain = self.apply_gain

        if (rectification := params.rectification) is None:
            self.rectification = lambda i: i
        else:
            match rectification:
                case 'abs':
                    self.rectification = lambda i: i.abs()
                case 'relu':
                    self.rectification = nn.ReLU()
                case None:
                    self.rectification = lambda i: i
                case _:
                    assert False, f'Unsupported {rectification=}'

        if (normalization := params.normalization) is None:
            self.normalization = lambda i: i
        else:
            match normalization.norm_method:
                case 'LCN':
                    # There may be problems when moving model between GPU and CPU due to lambda closure side effects
                    self.normalization = lambda i: self.apply_lcn(i, normalization.lcn_window_width, normalization.lcn_window_sigma)
                case None:
                    self.normalization = lambda i: i
                case _:
                    assert False, f'Unsupported {normalization.norm_method=}'

        if (pooling := params.pooling) is None:
            self.boxcar = lambda i: i
            self.pool = lambda i: i
        else:
            if pooling.boxcar_width is None or pooling.boxcar_width == 0:
                self.boxcar = lambda i: i
            else:
                assert pooling.boxcar_width > 0
                padding = pooling.boxcar_width // 2 
                self.boxcar = nn.AvgPool2d(kernel_size=pooling.boxcar_width, stride=1, padding=padding, count_include_pad=False)
    
            match pooling.pool_method:
                case 'avg':
                    self.pool = nn.AvgPool2d(kernel_size=pooling.kernel_size)
                case 'max':
                    self.pool = nn.MaxPool2d(kernel_size=pooling.kernel_size)
                case _:
                    assert False, f'Unsupported {pooling.pool_method=}'

    def forward(self, inp): 
        # inp.shape: batch, channel, height, width
        res = self.conv(inp)
        res = self.batch_norm2d(res)
        res = self.instance_norm2d(res)
        res = self.nonlinearity(res)
        res = self.gain(res)
        res = self.rectification(res)
        res = self.normalization(res)
        res = self.boxcar(res)
        res = self.pool(res)
        return res

    def apply_gain(self, inp):
        # See https://gist.github.com/kocherovms/1568d303d584d5033af3c169b99eaa33 for breakdown of gain multiplication logic
        res = inp
        shape = res.shape
        res = res.reshape(res.shape[0], res.shape[1], -1)
        res = res.transpose(1, 2)
        res = self.gain_params * res
        res = res.transpose(1, 2)
        res = res.reshape(shape)
        return res

    def apply_lcn(self, feature_maps, window_width, window_sigma):
        # feature_maps.shape: batch, feature_map, height, width
        # See https://gist.github.com/kocherovms/a5a39939dafcee21e0754bad043adeca for breakdown of LCN logic
        if not hasattr(self, 'lcn_conv') or self.lcn_conv is None or self.lcn_conv.in_channels != feature_maps.shape[1] or self.lcn_conv.weight.device != feature_maps.device:
            window = signal.windows.gaussian(window_width, window_sigma)[:,np.newaxis] 
            window = window @ window.T # aka 2d kernel
            window_norm = window / window.sum() / feature_maps.shape[1]
            self.lcn_conv = nn.Conv2d(
                in_channels=feature_maps.shape[1], 
                out_channels=feature_maps.shape[1],
                kernel_size=window_width,
                padding='same',
                bias=False,
                groups=feature_maps.shape[1], # one filter per feature_map
                device=feature_maps.device,
            )
            self.lcn_conv.requires_grad_(False)
            self.lcn_conv.weight[:,:,...] = torch.tensor(window_norm).to(device=CONFIG.cuda_device) # upload window to conv kernels

        convolved = self.lcn_conv(feature_maps)
        means = convolved.sum(dim=-3)
        means = einops.rearrange(means, 'b h w -> b 1 h w')
        v = feature_maps - means
        squared_v = v ** 2
        convolved = self.lcn_conv(squared_v)
        sigmas = torch.sqrt(convolved.sum(dim=-3))
        c = einops.rearrange(sigmas, 'b h w -> b (h w)').mean(axis=-1)
        sigmas = einops.rearrange(sigmas, 'b h w -> b 1 h w')
        c = einops.rearrange(c, 'b -> b 1 1 1')
        return v / torch.where(sigmas > c, sigmas, c)

    def load_weights(self, weights_source, model_registry):
        for classifier, target, load_func in zip(
            ('kernels', 'biases', 'gains'), 
            (self.conv.weight, self.conv.bias, self.gain_params),
            (self.load_kernels, self.load_biases, self.load_gain_params),
        ):
            if target is None:
                continue

            asset = model_registry.get_asset_content(weights_source.asset_name, weights_source.asset_version, 'npy', classifier)

            with torch.no_grad():
                with io.BytesIO(asset) as b:
                    asset = np.load(b)
                    asset = torch.tensor(asset)
                    # LOG('Before', classifier, asset.shape, target.shape, asset.sum(), target.sum())
                    sum = load_func(asset)
                    # LOG('After', asset.sum(), target.sum())
                    assert asset.sum() == target.sum()
                    LOG(f'{classifier} loaded from {weights_source.asset_name}:{weights_source.asset_version}')

    def load_kernels(self, asset):
        shape = einops.parse_shape(self.conv.weight, 'out_ch in_ch h w')
        self.conv.weight[:,:,...] = einops.rearrange(asset, 'k (h w) -> k 1 h w', h=shape['h'], w=shape['w'])

    def load_biases(self, asset):
        self.conv.bias[:] = asset

    def load_gain_params(self, asset):
        self.gain_params.data[:,:,...] = einops.rearrange(asset, 'g -> 1 1 g')

class StateModelUnit(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()

        match params.module:
            case 'LSTM':
                self.module = nn.LSTM(input_size, params.hidden_size, num_layers=params.num_layers, batch_first=True)
            case _:
                assert False, f'Unsupported {params.module=}'

        self.output_size = self.module.hidden_size

    def forward(self, inp): 
        # inp.shape: batch, chunk, input (of embedding_size or of ouput_size of preceeding unit)
        res, (hidden, cell) = self.module(inp)
        return res