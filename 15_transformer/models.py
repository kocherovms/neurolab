from dataclasses import dataclass
from collections import namedtuple

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


class ComboModel(nn.Module):
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
        self.cls_vocab_embedding = nn.Parameter(torch.zeros(1, 1, params.embedding_size))
        self.cls_pos_embedding = nn.Parameter(torch.zeros(1, 1, params.embedding_size))
        ntp = NoncausalTransformer.Params(**dataclasses.asdict(params))
        self.transformer = NoncausalTransformer(ntp) 
        self.predictor = nn.Linear(params.embedding_size, params.vocab_size)
        self.classifier = nn.Linear(params.embedding_size, 10)

    ForwardResult = namedtuple('ForwardResult', 'pred, cls')

    def forward(self, vocab_token_inds, pos_token_inds):
        def attach_cls(x):
            batch_size = len(x)
            cls_tokens = self.cls_vocab_embedding + self.cls_pos_embedding
            cls_tokens = cls_tokens.expand(batch_size, -1, -1) # cls_tokens shape becomes [batch, 1, emb_size]
            x = torch.cat((cls_tokens, x), dim=1) # [batch, 1+seq_len, emb_size]
            return x
            
        output = self.transformer(vocab_token_inds, pos_token_inds, attach_cls)
        cls_output = output[:,0,:] # [batch, emb_size]
        main_output = output[:,1:,:] # [batch, seq_len, emb_size]
        return ComboModel.ForwardResult(
            pred=self.predictor(main_output), 
            cls=self.classifier(cls_output)
        )
