import sys
import os
import datetime

import numpy as np

import cgf

sys.path.append('..')
from lib.utils import *
from lib.logging import *

def fit_bases(bases, patches, noise_var, β, σ, tol):
    λ = 1 / noise_var
    max_iter = 100
    assert len(bases) == len(patches)
    L = len(bases)
    M = bases.shape[1]
    N = patches.shape[1]
        
    bases_norm2 = np.sum(bases * bases, axis=0)
    # assert np.all(bases_norm2 > 0)
    coeffs_init = bases.T @ patches
    coeffs_init = (coeffs_init.T / bases_norm2).T
    # coeffs = np.zeros((M, N), dtype=np.double)
    coeffs = np.zeros((N, M), dtype=np.double)
    
    req = lambda a: np.require(a, dtype=np.double, requirements=['C_CONTIGUOUS'])
    
    cgf.cgf(L, M, N, req(bases.T), req(patches.T), req(coeffs_init.T), req(coeffs), λ, β, σ, tol, max_iter)

    return coeffs.T

bases = np.array([[0.17896048, 0.69793457],
        [0.68849418, 0.40182392],
        [0.80887865, 0.58090589],
        [0.3946178 , 0.42024509],
        [0.38471327, 0.70067435]])

patches = np.array([[0.42034354, 0.59886737, 0.17633227],
        [0.33168091, 0.29535977, 0.70301482],
        [0.19661624, 0.19843791, 0.05798285],
        [0.30898698, 0.74006256, 0.66647841],
        [0.50870405, 0.29780236, 0.13198214]])

# [0.42034354, 0.59886737, 0.17633227; 0.33168091, 0.29535977, 0.70301482; 0.19661624, 0.19843791, 0.05798285; 0.30898698, 0.74006256, 0.66647841; 0.50870405, 0.29780236, 0.13198214]

# patches = np.array([[0.42034354],
#         [0.33168091],
#         [0.19661624],
#         [0.30898698],
#         [0.50870405]])

print(fit_bases(bases, patches, 0.01, 2.2, 0.316, 0.01))

