import os
import numpy as np
import cupy as cp
import math

###
class ArrayUtils:
    xp = None
    from_gpu = None 
    to_gpu = None
    to_gpu_copy = None

    @staticmethod
    def init(is_cuda=None):
        if LangUtils.coalesce(is_cuda, cp.cuda.is_available()):
            ArrayUtils.xp = cp
            ArrayUtils.from_gpu = lambda a: a.get() if isinstance(a, cp.ndarray) else a
            ArrayUtils.to_gpu = lambda a: cp.asarray(a) if isinstance(a, np.ndarray) else a
            ArrayUtils.to_gpu_copy = lambda a: cp.asarray(a) if isinstance(a, np.ndarray) else a
        else:
            ArrayUtils.xp = np
            ArrayUtils.from_gpu = lambda a: a
            ArrayUtils.to_gpu = lambda a: a
            ArrayUtils.to_gpu_copy = lambda a: a.copy()
    
    @staticmethod
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

    @staticmethod
    def ensure_dtype(a, dt):
        assert a.dtype == dt, (a.dtype, dt)
        return a

    @staticmethod
    def ensure_shape(a, shape):
        assert a.shape == shape, (a.shape, shape)
        return a
        
    @staticmethod
    def ensure_len(a, l):
        assert len(a) == l, (len(a), l)
        return a

###
class LangUtils:
    @staticmethod
    def from_str(cast_func, s, default_value):
        try:
            return cast_func(s)
        except ValueError:
            return default_value

    @staticmethod
    def coalesce_fn(v, fn, default_value):
        if v is None:
            return default_value

        return fn(v)

    @staticmethod
    def coalesce(v, default_value):
        if v is None:
            return default_value

        return v

    @staticmethod
    def when(v, if_true, if_false):
        if v:
            return if_true
        else:
            return if_false

    @staticmethod
    def to_number(v):
        assert isinstance(v, str)
        
        if '.' in v:
            return float(v)
        else:
            return int(v)

###
class CudaUtils:
    @staticmethod
    def exec_cuda_kernel(kernel, items_count, params):
        cuda_block_size = 256
        cuda_blocks_count = math.ceil(items_count / cuda_block_size)
        kernel((cuda_blocks_count, ), (cuda_block_size,), params)