import torch.nn as nn
import torch.nn.functional as F

class LinearModelUnit(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()
        if params.dropout is None:
            self.dropout = lambda i: i
        else:
            assert params.dropout >= 0
            self.dropout = nn.Dropout(params.dropout)
            
        self.linear = nn.Linear(
            in_features=input_size, 
            out_features=params.size, 
            bias=True
        )

        if params.nonlinearity is None:
            self.nonlinearity = lambda i: i
        else:
            self.nonlinearity = getattr(nn, params.nonlinearity)()

        self.output_size = self.linear.out_features

    def forward(self, inp): 
        res = self.dropout(inp)
        res = self.linear(res)
        res = self.nonlinearity(res)
        return res

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