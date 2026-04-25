import numpy as np
import cupy as cp

import lang_utils as lu

xp = None
from_gpu = None 
to_gpu = None
to_gpu_copy = None

def init(is_cuda=None):
    if lu.coalesce(is_cuda, cp.cuda.is_available()):
        xp = cp
        from_gpu = lambda a: a.get() if isinstance(a, cp.ndarray) else a
        to_gpu = lambda a: cp.asarray(a) if isinstance(a, np.ndarray) else a
        to_gpu_copy = lambda a: cp.asarray(a) if isinstance(a, np.ndarray) else a
    else:
        xp = np
        from_gpu = lambda a: a
        to_gpu = lambda a: a
        to_gpu_copy = lambda a: a.copy()

def indices(list_like):
    return np.arange(len(list_like))

def v2sm(v, pad_value=None):
    assert v.ndim == 1
    root = np.sqrt(v.shape[0])
    sz = int(root)

    if sz * sz != v.shape[0]:
        assert pad_value is not None, f'Failed to make square matrix without padding from vector of size {v.shape}'
        sz = int(np.ceil(root))
        pad_elems_count = (sz * sz) - len(v)
        assert pad_elems_count > 0
        return np.r_[v, np.full(pad_elems_count, pad_value)].reshape(sz, sz)
    else:
        return v.reshape(sz, sz)

def ensure_dtype(a, dt):
    assert a.dtype == dt, (a.dtype, dt)
    return a

def ensure_shape(a, shape):
    assert a.shape == shape, (a.shape, shape)
    return a
    
def ensure_len(a, l):
    assert len(a) == l, (len(a), l)
    return a