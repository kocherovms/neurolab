from collections import namedtuple
from dataclasses import dataclass
import itertools
import lark
from utils import *

@dataclass
class LinearModelUnitParams: 
    size: int = None
    with_bias: bool = None
    nonlinearity: object = None 
    dropout: object = None

@dataclass
class Conv2dModelUnitParams: 
    Convolution = namedtuple('Convolution', 'in_channels_count out_channels_count in_channels_count_per_kernel kernel_size with_bias padding stride')
    Normalization = namedtuple('Normalization', 'norm_method lcn_window_width lcn_window_sigma')
    Pooling = namedtuple('Pooling', 'boxcar_width pool_method kernel_size')
    WeightsSource = namedtuple('WeightsSource', 'asset_name asset_version')

    module: str = None
    convolution: object = None
    batch_norm2d: object = None
    instance_norm2d: object = None
    nonlinearity: object = None 
    with_gain: bool = None
    rectification: str = None
    normalization: object = None
    pooling: object = None
    weights_source: str = None

@dataclass
class StateModelUnitParams: 
    module: str = None
    hidden_size: int = None
    num_layers: int = None

@dataclass
class NonlinearityParams:
    module: str = None
    args: list = None
    kwargs: dict = None

@dataclass
class OrdinaryParams:
    args: list = None
    kwargs: dict = None

@dataclass
class LearnRateParams: 
    Plateau = namedtuple('Plateau', 'factor patience')
    
    learn_rate: float = None
    plateau: object = None

def get_lark_tree_value(tree, var_name, default_value=None):
    try:
        return next(tree.scan_values(lambda i: i is not None and i.type == var_name)).value
    except StopIteration:
        return default_value

def get_lark_tree_values(tree, var_name):
    return list(map(lambda x: x.value, tree.scan_values(lambda i: i is not None and i.type == var_name)))

def to_basic_type(v):
    assert isinstance(v, str)

    if v.startswith('"'):
        return v.strip('"')
    elif v == "None":
        return None

    return LangUtils.to_number(v)

def parse_arg_list(t):
    args = list(map(to_basic_type, get_lark_tree_values(t, 'ARG_VALUE')))
    kwarg_names = get_lark_tree_values(t, 'KWARG_NAME')
    kwarg_values = list(map(to_basic_type, get_lark_tree_values(t, 'KWARG_VALUE')))
    kwargs = dict(zip(kwarg_names, kwarg_values))
    return args, kwargs

class ModelUnitParser:
    def __init__(self):
        grammar = '''
            spec: linear_unit_spec | lstm_unit_spec | conv2d_unit_spec | conv_transpose2d_unit_spec
    
            # LINEAR UNIT
            linear_unit_spec: "Linear" ":" _linear_unit_spec | "Linear" "(" _linear_unit_spec ")"
            _linear_unit_spec: (dropout_spec "->")? linear_size_spec BIAS? ("->" nonlinearity_spec)?
            linear_size_spec: expand_var_spec | SIZE
    
            # LSTM UNIT
            lstm_unit_spec: "LSTM" "(" SIZE ")" ("x" COUNT)? | "LSTM" ":" SIZE ("x" COUNT)?
    
            # CONV_2D UNIT
            conv2d_unit_spec: "Conv2d" "(" _conv_unit_spec ")" | "Conv2d" ":" _conv_unit_spec
            _conv_unit_spec: conv_spec ("->" norm2d_spec)? ("->" nonlinearity_spec)? ("->" GAIN)? ("->" RECTIFICATION)? ("->" normalization_spec)? ("->" filtered_pool_spec)? (";" weights_source_spec)?
            conv_spec: "conv" "(" IN_CHANNELS_COUNT "->" OUT_CHANNELS_COUNT "(" IN_CHANNELS_COUNT_PER_KERNEL ")" "x" KERNEL_SIZE BIAS? [("," padding_spec) | ("," stride_spec)]* ")"
            padding_spec: "padding" "=" PADDING
            stride_spec: "stride" "=" STRIDE
            IN_CHANNELS_COUNT: INT
            OUT_CHANNELS_COUNT: INT
            IN_CHANNELS_COUNT_PER_KERNEL: INT
            PADDING: "valid" | INT
            STRIDE: INT
        
            normalization_spec: lcn_spec
            lcn_spec: "LCN" "(" WINDOW_WIDTH "," WINDOW_SIGMA ")"
            WINDOW_WIDTH: INT
            WINDOW_SIGMA: INT
        
            filtered_pool_spec: (boxcar_spec "+")? pool_spec
        
            boxcar_spec: "boxcar(" WIDTH ")"
            
            pool_spec: POOL_METHOD "(" KERNEL_SIZE ")"
            POOL_METHOD: "avg" | "max"
    
            weights_source_spec: "from:" ASSET_NAME "," ASSET_VERSION
            ASSET_NAME: EXT_IDENTIFIER
            ASSET_VERSION: EXT_IDENTIFIER
    
            # CONV TRANSPOSE_2D UNIT
            conv_transpose2d_unit_spec: "ConvTranspose2d" ":" _conv_unit_spec
    
            ## DROPOUT
            dropout_spec: "dropout" ( "(" arg_list_spec ")" )?
    
            ## NONLINEARITY
            nonlinearity_spec: NONLINEARITY ( "(" arg_list_spec ")" )?
            NONLINEARITY: WORD

            ## NORM_2D
            norm2d_spec: batch_norm2d_spec | instance_norm2d_spec
    
            ## BATCH_NORM_2D
            batch_norm2d_spec: "BatchNorm2d" ( "(" arg_list_spec ")" )?

            ## INSTANCE_NORM_2D
            instance_norm2d_spec: "InstanceNorm2d" ( "(" arg_list_spec ")" )?
    
            ### Shared specs
            expand_var_spec: "$" IDENTIFIER
            IDENTIFIER: LETTER (LETTER|DIGIT|"_")*
            EXT_IDENTIFIER: (LETTER|DIGIT|"_"|"-")+
    
            arg_list_spec: (ARG_VALUE ("," ARG_VALUE)* ("," KWARG_NAME "=" KWARG_VALUE)*)? (KWARG_NAME "=" KWARG_VALUE ("," KWARG_NAME "=" KWARG_VALUE)*)?
            ARG_VALUE: NUMBER | ESCAPED_STRING
            KWARG_NAME: IDENTIFIER
            KWARG_VALUE: NUMBER | ESCAPED_STRING
    
            ### Shared types
            SIZE: INT
            KERNEL_SIZE: INT
            COUNT: INT
            WIDTH: INT
            
            BIAS: "+bias" | "-bias"
            GAIN: "gain"
            RECTIFICATION: WORD

            %import common.ESCAPED_STRING
            %import common.LETTER
            %import common.WORD
            %import common.DIGIT
            %import common.NUMBER
            %import common.INT
            %import common.WS
            %ignore WS
        '''
        self.lark_parser = lark.Lark(grammar, start='spec')

    def parse(self, unit_spec, expand_vars={}):
        tree = self.lark_parser.parse(unit_spec)

        if spec_subtree := list(tree.find_data('linear_unit_spec')):
            spec_subtree = spec_subtree[0]
            gtv = lambda var_name, default_value='': get_lark_tree_value(spec_subtree, var_name, default_value)
            params = LinearModelUnitParams()

            if t := list(spec_subtree.find_data('expand_var_spec')):
                expand_var_name = get_lark_tree_value(t[0], 'IDENTIFIER')
                params.size = int(expand_vars[expand_var_name])
            else:
                params.size = int(gtv('SIZE'))

            params.with_bias = gtv('BIAS', '') in ['+bias', '']
            params.nonlinearity = self.create_nonlinearity_params(spec_subtree)
            params.dropout = self.create_dropout_params(spec_subtree)
        elif spec_subtree := list(itertools.chain(tree.find_data('conv2d_unit_spec'), tree.find_data('conv_transpose2d_unit_spec'))):
            spec_subtree = spec_subtree[0]
            gtv = lambda var_name, default_value='': get_lark_tree_value(spec_subtree, var_name, default_value)
            params = Conv2dModelUnitParams()
            params.module = 'Conv2d' if any(spec_subtree.find_data('conv2d_unit_spec')) else 'ConvTranspose2d'
            coerce_padding = lambda p: p if p in ["valid", "same"] else int(p)
            params.convolution = Conv2dModelUnitParams.Convolution(
                in_channels_count=int(gtv('IN_CHANNELS_COUNT')),
                out_channels_count=int(gtv('OUT_CHANNELS_COUNT')),
                in_channels_count_per_kernel=int(gtv('IN_CHANNELS_COUNT_PER_KERNEL')),
                kernel_size=int(gtv('KERNEL_SIZE')),
                with_bias=gtv('BIAS', '') == '+bias',
                padding=LangUtils.coalesce_fn(gtv('PADDING', None), coerce_padding, None),
                stride=LangUtils.coalesce_fn(gtv('STRIDE', None), int, None),
            )

            params.batch_norm2d = self.create_batch_norm2d_params(spec_subtree)
            params.instance_norm2d = self.create_instance_norm2d_params(spec_subtree)
            params.nonlinearity = self.create_nonlinearity_params(spec_subtree)
            params.with_gain = bool(gtv('GAIN', ''))
            params.rectification = gtv('RECTIFICATION', None)
            
            if t := list(spec_subtree.find_data('lcn_spec')):
                params.normalization = Conv2dModelUnitParams.Normalization(
                    norm_method='LCN', 
                    lcn_window_width=int(get_lark_tree_value(t[0], 'WINDOW_WIDTH')),
                    lcn_window_sigma=int(get_lark_tree_value(t[0], 'WINDOW_SIGMA')),
            )

            if t := list(spec_subtree.find_data('filtered_pool_spec')):
                params.pooling = Conv2dModelUnitParams.Pooling(
                    boxcar_width=int(get_lark_tree_value(t[0], 'WIDTH', 0)),
                    pool_method=get_lark_tree_value(t[0], 'POOL_METHOD'),
                    kernel_size=int(get_lark_tree_value(t[0], 'KERNEL_SIZE')),
                )
            
            if list(spec_subtree.find_data('weights_source_spec')):
                params.weights_source = Conv2dModelUnitParams.WeightsSource(
                    asset_name=gtv('ASSET_NAME'), 
                    asset_version=gtv('ASSET_VERSION'), 
            )
        elif spec_subtree := list(tree.find_data('lstm_unit_spec')):
            spec_subtree = spec_subtree[0]
            gtv = lambda var_name, default_value='': get_lark_tree_value(spec_subtree, var_name, default_value)
            params = StateModelUnitParams()
            params.module = 'LSTM'
            params.hidden_size = int(gtv('SIZE'))
            params.num_layers = int(gtv('COUNT', 1))
        else:
            assert False, f'Unsupported unit spec="{unit}"'

        return params

    def create_dropout_params(self, tree):
        if t := list(tree.find_data('dropout_spec')):
            args, kwargs = parse_arg_list(t[0])
            return OrdinaryParams(args=args, kwargs=kwargs)

        return None
    
    def create_nonlinearity_params(self, tree):
        if t := list(tree.find_data('nonlinearity_spec')):
            args, kwargs = parse_arg_list(t[0])
            return NonlinearityParams(module=get_lark_tree_value(t[0], 'NONLINEARITY'), args=args, kwargs=kwargs)

        return None

    def create_batch_norm2d_params(self, tree):
        if t := list(tree.find_data('batch_norm2d_spec')):
            args, kwargs = parse_arg_list(t[0])
            return OrdinaryParams(args=args, kwargs=kwargs)

        return None

    def create_instance_norm2d_params(self, tree):
        if t := list(tree.find_data('instance_norm2d_spec')):
            args, kwargs = parse_arg_list(t[0])
            return OrdinaryParams(args=args, kwargs=kwargs)

        return None
        
def hp_parse_model_units(units, expand_vars={}):
    parser = ModelUnitParser()
    params_list = []

    for unit in units:
        params = parser.parse(unit, expand_vars)
        params_list.append(params)
    
    return params_list

def hp_parse_learn_rate(learn_rate):
    params = LearnRateParams()

    if isinstance(learn_rate, float):
        params.learn_rate = learn_rate
        return params
        
    grammar = '''
        spec: initial_lr_spec(","plateau_spec)?
    
        initial_lr_spec: INITIAL_LR
        INITIAL_LR: NUMBER
    
        plateau_spec: "plateau" "(" (|plateau_params_spec ("," plateau_params_spec)*) ")"
        plateau_params_spec: plateau_factor_spec | plateau_patience_spec
        plateau_factor_spec: "factor" "=" plateau_factor_value_spec
        plateau_factor_value_spec: PLATEAU_FACTOR_VALUE
        plateau_patience_spec: "patience" "=" plateau_patience_value_spec
        plateau_patience_value_spec: PLATEAU_PATIENCE_VALUE
        PLATEAU_FACTOR_VALUE: NUMBER
        PLATEAU_PATIENCE_VALUE: NUMBER
        
        %import common.NUMBER
        %import common.WS
        %ignore WS
    '''
    parser = lark.Lark(grammar, start='spec')
    tree = parser.parse(learn_rate)
    gtv = lambda var_name, default_value='': get_lark_tree_value(tree, var_name, default_value)
    params.learn_rate = float(gtv('INITIAL_LR'))
    
    if list(tree.find_data('plateau_spec')):
        params.plateau = LearnRateParams.Plateau(
            factor=float(gtv('PLATEAU_FACTOR_VALUE', 0.1)),
            patience=int(gtv('PLATEAU_PATIENCE_VALUE', 10)),
        )

    return params
