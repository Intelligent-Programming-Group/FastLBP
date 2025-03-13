/// @file bp.cu
/// @brief Defines kernel functions for calculating messages parallelly.
///
/// @date 2024-07-25

#include <kernel/bp.h>
#include <utils/cuda_utils.h>

namespace lbp {
namespace kernel {

__global__ void initArrayKernel(Real *arr, Real val, size_t size) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        arr[idx] = val;
    }
}

__global__ void calcProductKernel(
    Real *output,
    const Real *oldMessage,
    size_t **nbs,
    const size_t *numNbs,
    const size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i =  idx; i < size; i += stride) {
        size_t numNbs_i = numNbs[i];
        size_t *nbs_i = nbs[i];
        Real ans = output[i];
        for (size_t nb_i = 0; nb_i < numNbs_i; nb_i++) {
            size_t nb = nbs_i[nb_i];
            ans *= oldMessage[nb];
        }
        output[i] = ans;
    }
}

__global__ void marginalizeKernel(
    Real *output, 
    const Real *joint, 
    size_t **index, 
    const size_t *numIndex, 
    const size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i =  idx; i < size; i += stride) {
        Real ans = 0.0;
        for (size_t nb_i = 0; nb_i < numIndex[i]; nb_i++) {
            size_t nb = index[i][nb_i];
            ans += joint[nb];
        }
        output[i] = ans;
    }
}

__global__ void normalizeKernel(
    Real *vecs, size_t *offsets, size_t *lengths, size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride) {
        size_t offset = offsets[i];
        size_t length = lengths[i];
        Real sum = 0.0;
        for (size_t j = 0; j < length; j++) {
            sum += vecs[offset + j];
        }
        for (size_t j = 0; j < length; j++) {
            vecs[offset + j] /= sum;
        }
    }
}

__global__ void distKernel(Real *p, const Real *q, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        Real diff = p[tid] - q[tid];
        p[tid] = fabs(diff);
    }
}

__global__ void calcMaxKernel(Real *d_out, const Real *d_in, int N) {
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // allocated in the kernel call with <<<b, t, shmem>>>
    extern __shared__ Real shared[];
    shared[tid] = idx < N ? d_in[idx] : 0;
    __syncthreads(); // make sure the entire block is loaded!
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        __syncthreads(); // make sure all adds at one stage are done!
    }
    // only thread 0 writes result for this block to global memory
    if (tid == 0) {
        d_out[blockIdx.x] = shared[0];
    }
}

void initArray(Real *arr, Real val, size_t size) {
    const size_t blockSize = 256;
    const size_t numBlocks = (size + blockSize - 1) / blockSize;
    initArrayKernel<<<numBlocks, blockSize>>>(arr, val, size);
}

void calcProduct(
    Real *output,
    const Real *oldMessage,
    size_t **nbs,
    const size_t *numNbs,
    const size_t size
) {
    const size_t blockSize = 256;
    const size_t numBlocks = (size + blockSize - 1) / blockSize;
    calcProductKernel<<<numBlocks, blockSize>>>(output, oldMessage, nbs, numNbs, size);
}

void marginalize(
    Real *output, 
    const Real *joint, 
    size_t **index, 
    const size_t *numIndex, 
    const size_t size
) {
    const size_t blockSize = 256;
    const size_t numBlocks = (size + blockSize - 1) / blockSize;
    marginalizeKernel<<<numBlocks, blockSize>>>(output, joint, index, numIndex, size);
}

void normalize(
    Real *vecs, size_t *offsets, size_t *lengths, size_t size
) {
    const size_t blockSize = 256;
    const size_t numBlocks = (size + blockSize - 1) / blockSize;
    normalizeKernel<<<numBlocks, blockSize>>>(vecs, offsets, lengths, size);
}

void dist(Real *p, const Real *q, size_t size) {
    const size_t blockSize = 256;
    const size_t numBlocks = (size + blockSize - 1) / blockSize;
    distKernel<<<numBlocks, blockSize>>>(p, q, size);
}

Real calcMax(Real *p, size_t size) {
    const size_t blockSize = 256;
    const size_t numBlocks = (size + blockSize - 1) / blockSize;
    Real maxDiff;

    Real *d_tmp;
    cudaMalloc((void **)&d_tmp, numBlocks * sizeof(Real));
    int num = size;
    Real *ptr_in = p; 
    Real *ptr_out = d_tmp;
    int kShared = blockSize * sizeof(Real);
    while (num > 1) {
        size_t blocks = (num + blockSize - 1) / blockSize;
        // Dynamically allocate shared memory <<<b, t, shmem>>>
        calcMaxKernel<<<blocks, blockSize, kShared>>>(ptr_out, ptr_in, num);
        num = blocks;
        std::swap(ptr_in, ptr_out);
    }
    cudaMemcpy(&maxDiff, ptr_in, sizeof(Real), cudaMemcpyDeviceToHost);
    cudaFree(d_tmp);
    return maxDiff;
}

} // namespace kernel
} // namespace lbp
