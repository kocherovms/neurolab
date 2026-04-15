from dataclasses import dataclass

import torch
import torch.nn as nn

class NoncausalTransformer(nn.Module):
    PAD_TOKEN_IND = 0
    MASK_TOKEN_IND = 1
    
    @dataclass 
    class Params:
        vocab_size: int = None
        pos_tokens_count: int = None
        embedding_size: int = None # aka d_model
        heads_count: int = None
        layers_count: int = None
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_embedding = nn.Embedding(params.vocab_size, params.embedding_size)
        self.pos_embedding = nn.Embedding(params.pos_tokens_count, params.embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params.embedding_size, 
            nhead=params.heads_count, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=params.layers_count)

    def forward(self, vocab_token_inds, pos_token_inds, attach_task_specific_info_fn=None):
        x = self.vocab_embedding(vocab_token_inds)
        x = x + self.pos_embedding(pos_token_inds)

        if attach_task_specific_info_fn is not None:
            x = attach_task_specific_info_fn(x)
            
        return self.encoder(x, is_causal=False)
