import os
from dataclasses import dataclass

@dataclass
class Bundle:
    df_sources: object = None
    df_vocab: object = None
    df_chunks: object = None
