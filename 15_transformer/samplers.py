from collections import namedtuple
import numpy as np

class BaseSampler:
    CommonParams = namedtuple('CommonParams', 'spec image_size patch_size max_seq_length rng')
    
    def __init__(self, common_params):
        self.spec = common_params.spec
        self.image_size = common_params.image_size
        self.patch_size = common_params.patch_size
        self.max_seq_length = common_params.max_seq_length
        self.rng = common_params.rng

class BaseRandomSampler(BaseSampler):
    def __init__(self, common_params, max_overlap):
        super().__init__(common_params)
        self.image_shape = (self.image_size, self.image_size)
        self.patch_area = self.patch_size ** 2
        assert 0 < max_overlap <= 1, max_overlap
        self.max_overlap = max_overlap

    def __call__(self, df_encoding):
        seq = []
        busy_map = np.zeros(self.image_shape, dtype='b')
        visited_row_inds = set()
        cycles_count = 0

        while len(seq) < self.max_seq_length:
            cycles_count += 1
            row_ind = self.rng.choice(len(df_encoding))

            if row_ind in visited_row_inds:
                continue

            enc_row = next(df_encoding.iloc[[row_ind]].itertuples())
            overlap = busy_map[enc_row.i:enc_row.i2,enc_row.j:enc_row.j2].sum() / self.patch_area

            if overlap > self.max_overlap:
                continue

            if not self.is_bfr_ok(enc_row.bfr):
                continue
                
            seq.append((enc_row.pos_token_ind, enc_row.vocab_token_ind))
            busy_map[enc_row.i:enc_row.i2,enc_row.j:enc_row.j2] = 1
            visited_row_inds.add(row_ind)
        
        return np.array(seq, dtype=int), cycles_count

class LinearRandomSampler(BaseRandomSampler):
    def __init__(self, common_params, max_overlap, b):
        super().__init__(common_params, max_overlap)
        # build y=kx+b for line which passes through point (1,1)
        self.k = 1 - b
        self.b = b

    def is_bfr_ok(self, bfr):
        threshold = self.k * bfr + self.b
        return self.rng.uniform() < threshold

class SigmoidRandomSampler(BaseRandomSampler):
    def __init__(self, common_params, max_overlap, k, b):
        super().__init__(common_params, max_overlap)
        self.k = k
        self.b = b

    def is_bfr_ok(self, bfr):
        threshold = 1 / (1 + np.exp(self.k * (-bfr + self.b)))
        return self.rng.uniform() < threshold

class SpiralSampler(BaseSampler):
    def __init__(self, common_params, direction, stride=None):
        super().__init__(common_params)
        stride = self.patch_size if stride is None else stride
        
        match direction:
            case 'cw': self.steps = (-stride, +stride, +stride, -stride)
            case 'ccw': self.steps = (-stride, -stride, +stride, +stride)
            case _: assert False, f'Unsupported {direction=}'
        
    def __call__(self, df_encoding):
        df_non_empty_encoding = df_encoding[df_encoding.bfr > 0]
        start_i, start_j = int(df_non_empty_encoding.center_i.mean()), int(df_non_empty_encoding.center_j.mean())
        d = dict(map(lambda row: ((row.center_i, row.center_j), row),  df_encoding.itertuples()))
        seq = []
        cycles_count = 0
        
        for i, j, step_size in self.spiral_generator(start_i, start_j):
            # print(i, j, d[(i, j)].i, d[(i, j)].i2, d[(i, j)].j, d[(i, j)].j2)
            cycles_count += 1
            
            if r := d.get((i, j), False):
                seq.append((r.pos_token_ind, r.vocab_token_ind))
    
                if len(seq) >= self.max_seq_length:
                    break
                
            if step_size > self.image_size * 2:
                break

        return np.array(seq, dtype=int), cycles_count
        
    def spiral_generator(self, i, j):
        step_size = 1
        yield (i, j, step_size)
        
        while True:
            # 1. Move Up (i decreases)
            for _ in range(step_size):
                # i -= self.patch_size
                i += self.steps[0]
                yield (i, j, step_size)
            
            # 2. CW: Move Right (j increases) 
            # 2. CCW: Move Left (j decreases) 
            for _ in range(step_size):
                # j += self.patch_size
                j += self.steps[1]
                yield (i, j, step_size)
            
            step_size += 1 # Increase step length
            
            # 3. Move Down (i increases)
            for _ in range(step_size):
                # i += self.patch_size
                i += self.steps[2]
                yield (i, j, step_size)
                
            # 4. CW: Move Left (j decreases)
            # 4. CCW: Move Right (j increases)
            for _ in range(step_size):
                # j -= self.patch_size
                j += self.steps[3]
                yield (i, j, step_size)
                
            step_size += 1 

class ZigzagSampler(BaseSampler):
    def __init__(self, common_params, stride=None):
        super().__init__(common_params)
        self.stride = self.patch_size if stride is None else stride

    def __call__(self, df_encoding):
        df_non_empty_encoding = df_encoding[df_encoding.bfr > 0]
        min_i, min_j = int(df_non_empty_encoding.i.min()), int(df_non_empty_encoding.j.min())
        max_i, max_j = int(df_non_empty_encoding.i.max()), int(df_non_empty_encoding.j.max())
        d = dict(map(lambda row: ((row.i, row.j), row),  df_encoding.itertuples()))
        seq = []
        cycles_count = 0
        ij_list = []

        for i in range(min_i, max_i + 1, self.stride):
            for j in range(min_j, max_j + 1, self.stride):
                ij_list.append((i,j))

        for ij in ij_list:
            cycles_count += 1
            r = d[ij]
            seq.append((r.pos_token_ind, r.vocab_token_ind))

            if len(seq) >= self.max_seq_length:
                break

        return np.array(seq, dtype=int), cycles_count