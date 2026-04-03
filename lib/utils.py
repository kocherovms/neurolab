import os
import numpy as np
import cupy as cp
import math

### 
class DBUtils:
    @staticmethod
    def is_table_exists(db_con, table_name):
        cur = db_con.cursor() 
        return len(cur.execute('SELECT name FROM sqlite_master WHERE type=:type AND name=:table_name', {'type': 'table', 'table_name': table_name}).fetchall()) > 0

    @staticmethod
    def is_table_empty(db_con, table_name):
        cur = db_con.cursor() 
        return len(cur.execute(f'SELECT * FROM {table_name} LIMIT 1').fetchall()) < 1

    @staticmethod
    def drop_table_safe(db_con, tn):
        if DBUtils.is_table_exists(db_con, tn):
            db_con.cursor().execute(f'DROP TABLE {tn}')
            db_con.commit()

    @staticmethod
    def get_full_db_file_name(config, db_file_name, with_prefix=True):
        base_path = os.path.dirname(os.path.abspath(config.config_fname))
        return os.path.join(os.path.join(base_path, config.dataset_path), ('', config.db_file_name_prefix)[with_prefix] + db_file_name)

    @staticmethod
    def get_column_names(db_con, table_name):
        cur = db_con.cursor() 
        return list(map(lambda row: row[1], cur.execute(f'PRAGMA table_info({table_name})').fetchall()))

    @staticmethod
    def ensure_table_columns(db_con, table_name, column_names):
        cur = db_con.cursor() 
        existing_column_names = set(map(lambda row: row[1], cur.execute(f'PRAGMA table_info({table_name})').fetchall()))
        missing_column_names = set(column_names) - existing_column_names

        if not missing_column_names:
            return

        for column_name in missing_column_names:
            cur.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name}')

        db_con.commit()

###
class MathUtils:
    @staticmethod
    def softmax(x):
        max_x = np.max(x)
        exp_x = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x

    @staticmethod
    def conflate(pdfs):    
        n = np.prod(pdfs, axis=0)
        d = n.sum()
    
        if np.isclose(d, 0):
            return np.zeros(len(pdfs))
            
        return n / d

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n-1] = np.array(a[:n-1]) * n
        return ret / n

    @staticmethod
    def get_angle_diff(a_from, a_to):
        andiff = a_to - a_from
        return (andiff + 180) % 360 - 180

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