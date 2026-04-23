import math

def exec_cuda_kernel(kernel, items_count, params):
    cuda_block_size = 256
    cuda_blocks_count = math.ceil(items_count / cuda_block_size)
    kernel((cuda_blocks_count, ), (cuda_block_size,), params)

