from collections import namedtuple
from dataclasses import dataclass
import einops
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    @dataclass 
    class Params:
        image_size: int = None
        d_model: int = None # aka embedding_size
        heads_count: int = None
        layers_count: int = None
        cls_targets_count: int = 10
        
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.features_extractor = nn.Sequential(
            # Layer 1: Detect basic edges/curves (128 -> 64)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Layer 2: Intermediate textures (64 -> 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Layer 3: "Token" formation (32 -> 8)
            # This results in an 8x8 grid (64 tokens)
            nn.Conv2d(64, params.d_model, kernel_size=4, stride=4) 
        )

        self.reconstructor = nn.Sequential(
            # First upsample: 8x8 -> 16x16
            nn.ConvTranspose2d(params.d_model, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Second upsample: 16x16 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Final reconstruction: 64x64 -> 128x128
            # The overlap here (kernel 4, stride 2) helps remove boundaries
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Output pixels in [0, 1]
        )

        with torch.no_grad():
            probe_tensor = torch.zeros((1, 1, params.image_size, params.image_size))
            output = self.features_extractor(probe_tensor)
            output = einops.rearrange(output, '1 d h w -> 1 h w d')
            assert output.shape[1] == output.shape[2]
            self.feature_map_size = output.shape[1]
            self.seq_len = self.feature_map_size ** 2

        with torch.no_grad():
            probe_tensor = torch.zeros((1, self.seq_len, params.d_model))
            probe_tensor = einops.rearrange(probe_tensor, '1 (h w) d -> 1 d h w', h=self.feature_map_size)
            output = self.reconstructor(probe_tensor)
            shape = einops.parse_shape(output, '1 1 h w')
            assert shape['h'] == shape['w']
            assert shape['h'] == params.image_size
        
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, params.d_model)) # combined token+pos
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.d_model, 
            nhead=params.heads_count, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=params.layers_count)
        self.classifier = nn.Linear(params.d_model, params.cls_targets_count)

    ForwardResult = namedtuple('ForwardResult', 'recon cls')

    def forward(self, masked_images):
        shape = einops.parse_shape(masked_images, 'b 1 h w')
        batch_size = shape['b']
        tvs = self.features_extractor(masked_images) # tv = thought vector
        tvs = einops.rearrange(tvs, 'b d h w -> b (h w) d')
        assert tvs.shape[1] == self.seq_len
        cls_tokens = self.cls_embedding.expand(batch_size, -1, -1) # cls_tokens shape becomes [batch, 1, d_model]
        tvs = torch.cat((cls_tokens, tvs), dim=1) # [batch, 1+seq_len, d_model]
        output = self.encoder(tvs, is_causal=False)
        cls_output = output[:,0,:] # [batch, d_model]
        main_output = output[:,1:,:] # [batch, seq_len, d_model]
        main_output = einops.rearrange(main_output, 'b (h w) d -> b d h w', h=self.feature_map_size)
        return HybridModel.ForwardResult(
            recon=self.reconstructor(main_output),
            cls=self.classifier(cls_output),
        )