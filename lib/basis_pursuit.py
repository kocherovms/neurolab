import numpy as np
import cupy as cp
import einops
import torch
import scipy.linalg
from utils import *

# Хотим, найти только один, но самый лучший базис. Т.е. что-то типа победитель забирает все (winner takes it all)
# X - patches as rows
# W - bases as columns, we don't expect anything on W (i.e. no orthogonality, normality and so on)
def bp_batch_solo(X, W):
    # Use broadcasting to get differences
    diff = einops.rearrange(X, 'n d -> n 1 d') - einops.rearrange(W, 'd m -> 1 m d')
    
    if isinstance(diff, torch.Tensor):
        loss_matrix = torch.mean(diff**2, dim=2)
        best_inds = torch.argmin(loss_matrix, dim=1, keepdim=True)
        Z = torch.zeros_like(loss_matrix).scatter_(1, best_inds, 1.0)
    elif isinstance(diff, np.ndarray):
        loss_matrix = (diff ** 2).mean(axis=2)
        best_inds = np.argmin(loss_matrix, axis=1)
        Z = np.zeros_like(loss_matrix)
        Z[np.arange(loss_matrix.shape[0]), best_inds] = 1
    elif isinstance(diff, cp.ndarray):
        loss_matrix = (diff ** 2).mean(axis=2)
        best_inds = cp.argmin(loss_matrix, axis=1)
        Z = cp.zeros_like(loss_matrix)
        Z[cp.arange(loss_matrix.shape[0]), best_inds] = 1
    else:
        assert False, f'Unsupported {type(diff)=}'

    return Z


# W = matrix, wich each column is a basis
def bp_ista(x, W, init_z, pred_z, rho=0.5, gamma=0.1, L=1, iters_count=300):
    z = init_z
    alpha = 1 / L # L - Lipschitz constant
    lambd = rho * alpha
    
    for _ in range(iters_count):
        recon_grad = W.T @ (W @ z - x) # turn z to pred_x, compute diff and turn diff back to z space
        pred_grad = gamma * (z - pred_z)
        total_grad = recon_grad + pred_grad
        
        z_next = z - alpha * total_grad
        # soft shrink
        z = np.where(
            z_next > lambd, 
            z_next - lambd, 
            np.where(
                z_next < -lambd, 
                z_next + lambd, 
                0)
            )
        
    return z

cuda_ista_kernel = None

def bp_init_cuda_ista_kernel():
    global cuda_ista_kernel
    code = r'''
        #include <assert.h>
        #include <math.h>
        
        static __device__
        void softshrink(float * const theVector, const int theVectorSize, const float theLambda) {
            assert(theLambda > 0);
            
            for(int i = 0; i < theVectorSize; i++) {
                if(abs(theVector[i]) <= theLambda) {
                    theVector[i] = 0;
                }
                else {
                    theVector[i] += theVector[i] < 0 ? theLambda : -theLambda;
                }
            }
        }
        
        static __device__
        float dotProduct(const float * const theVector1, const float * const theVector2, const int theVectorSize) {
            float r = 0;
        
            for(int i = 0; i < theVectorSize; i++) {
                r += theVector1[i] * theVector2[i];
            }
        
            return r;
        }
        
        static __device__
        void loadVector(float * theVector, const float * theMatrix, const int theN, const int theM, const int theRow, const int theColumn) {
            assert((theRow >= 0) ^ (theColumn >= 0));
        
            if(theRow >= 0) {
                assert(theRow < theN);
                const float * const row = theMatrix + theRow * theM;
                
                for(int j = 0; j < theM; j++) {
                    theVector[j] = row[j];
                }
            }
            else if(theColumn >= 0) {
                assert(theColumn < theM);
                const float * const column = theMatrix + theColumn;
        
                for(int i = 0; i < theN; i++) {
                    theVector[i] = column[i * theM];
                }
            }
        }
        
        static const int MAX_INPUT_SIZE = 256;
        static const int MAX_BASES_COUNT = 256;
        
        extern "C" __global__
        void ista(
            // INPUT
            const int theInputsCount,
            const int theInputSize,
            const float * const theInputsMatrix, // matrix[theInputsCount][theInputSize]
            const int theBasesCount,
            const float * const theBasesMatrix, // matrix[theInputSize][theBasesCount], COLUMN BASED LAYOUT!
            const float * const theInitCoeffsMatrix, // // matrix[theInputsCount][theBasesCount]
            const float * const thePredCoeffsMatrix, // // matrix[theInputsCount][theBasesCount]
            const float theRho, 
            const float theGamma,
            const float theL,
            const int theItersCount,
            // OUTPUT
            float * const theCoeffsMatrix  // matrix[theInputsCount][theBasesCount], 
        ) {
            assert(MAX_INPUT_SIZE >= theInputSize);
            assert(MAX_BASES_COUNT >= theBasesCount);
            assert(theL != 0);
            const int inputInd = blockDim.x * blockIdx.x + threadIdx.x;
        
            if(inputInd >= theInputsCount)
                return;
        
            const auto inputsPtrOffset = inputInd * theInputSize;
            const float * const input = theInputsMatrix + inputsPtrOffset;
            
            const auto coeffsPtrOffset = inputInd * theBasesCount;
            float * const coeffs = theCoeffsMatrix + coeffsPtrOffset;
            const float * const initCoeffs = theInitCoeffsMatrix + coeffsPtrOffset;
            memcpy(coeffs, initCoeffs, sizeof(*coeffs) * theBasesCount);
        
            const float * const predCoeffs = thePredCoeffsMatrix + coeffsPtrOffset;
        
            const float alpha = 1 / theL;
            const float softshrinkLambda = theRho * alpha;
        
            float workVectorInput[MAX_INPUT_SIZE] = {0};
            float workVectorBase[MAX_INPUT_SIZE] = {0};
            float workVectorCoeffs[MAX_BASES_COUNT] = {0};
        
            for(int iter = 0; iter < theItersCount; iter++) {
                // recon_grad = W.T @ (W @ z - x) # turn z to pred_x, compute diff and turn diff back to z space
                // pred_grad = gamma * (z - pred_z)
                // total_grad = recon_grad + pred_grad
                
                // z_next = z - alpha * total_grad
                
                // (W @ z - x) -> workVectorInput
                for(int i = 0; i < theInputSize; i++) {
                    loadVector(workVectorCoeffs, theBasesMatrix, theInputSize, theBasesCount, i, -1); // load row
                    const float dp = dotProduct(workVectorCoeffs, coeffs, theBasesCount);
                    workVectorInput[i] = dp - input[i];
                }
        
                float * const totalGrad = workVectorCoeffs;
                
                // compute recon_grad and store in totalGrad
                for(int j = 0; j < theBasesCount; j++) {
                    loadVector(workVectorBase, theBasesMatrix, theInputSize, theBasesCount, -1, j); // load column
                    const float dp = dotProduct(workVectorBase, workVectorInput, theInputSize);
                    totalGrad[j] = dp;
                }
        
                // compute pred_grad and update totalGrad in place
                for(int j = 0; j < theBasesCount; j++) {
                    totalGrad[j] += theGamma * (coeffs[j] - predCoeffs[j]); 
                }
        
                for(int j = 0; j < theBasesCount; j++) {
                    coeffs[j] -= alpha * totalGrad[j];
                }
        
                softshrink(coeffs, theBasesCount, softshrinkLambda);
            }
        }'''
    cuda_ista_kernel = cp.RawKernel(code, 'ista', backend='nvcc')
    cuda_ista_kernel.compile()

def bp_batch_ista(patches, W, init_Z, pred_Z, is_cuda, vocab_size, rho, gamma, iters_count):
    assert init_Z.shape[0] == len(patches)
    assert init_Z.shape[1] == vocab_size
    assert pred_Z.shape == init_Z.shape
    
    max_index = W.shape[1] - 1
    L = scipy.linalg.eigvalsh(W.T @ W, subset_by_index=(max_index, max_index)).item()
    
    if is_cuda:
        if cuda_ista_kernel is None:
            bp_init_cuda_ista_kernel()

        assert cuda_ista_kernel is not None
        
        cuda_init_Z = ArrayUtils.to_gpu(init_Z)
        cuda_pred_Z = ArrayUtils.to_gpu(pred_Z)
        cuda_Z = cp.zeros((len(patches), vocab_size), dtype='f')
        cuda_kernel_params = (
            # INPUT
            cp.int32(patches.shape[0]),
            cp.int32(patches.shape[1]),
            ArrayUtils.ensure_dtype(ArrayUtils.to_gpu(patches), cp.float32),
            cp.int32(W.shape[1]),
            ArrayUtils.ensure_dtype(ArrayUtils.to_gpu(W), cp.float32),
            ArrayUtils.ensure_dtype(cuda_init_Z, cp.float32),
            ArrayUtils.ensure_dtype(cuda_pred_Z, cp.float32),
            cp.float32(rho),
            cp.float32(gamma),
            cp.float32(L),
            cp.int32(iters_count),
            # OUTPUT
            ArrayUtils.ensure_dtype(cuda_Z, cp.float32),
        )
        CudaUtils.exec_cuda_kernel(cuda_ista_kernel, len(patches), cuda_kernel_params)
        return ArrayUtils.from_gpu(cuda_Z)
    else:
        Z = np.zeros((len(patches), vocab_size)).astype('f') 
        
        for ind, patch in enumerate(patches):
            Z[ind] = bp_ista(
                patch, 
                W, 
                init_z=init_Z[ind], 
                pred_z=pred_Z[ind], 
                rho=rho, 
                gamma=gamma, 
                L=L, 
                iters_count=iters_count)

        return Z