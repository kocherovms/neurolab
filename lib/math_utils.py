import numpy as np

def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def conflate(pdfs):    
    n = np.prod(pdfs, axis=0)
    d = n.sum()

    if np.isclose(d, 0):
        return np.zeros(len(pdfs))
        
    return n / d

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n-1] = np.array(a[:n-1]) * n
    return ret / n

def get_angle_diff(a_from, a_to):
    andiff = a_to - a_from
    return (andiff + 180) % 360 - 180

# Dr. Shane Ross explained this beautifuly in https://www.youtube.com/watch?v=HCd-leV8OkU
# Also improvement is made to support batched input
class RecursiveAverageFilter:
    def __init__(self):
        self.n = 0
        self.v = 0

    def __call__(self, x, batch_size=1):
        n_new = self.n + batch_size
        self.v = (self.n * self.v + batch_size * x) / n_new
        self.n = n_new
        return self.v

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return f'RecursiveAverageFilter(v={self.v}, n={self.n})'
    