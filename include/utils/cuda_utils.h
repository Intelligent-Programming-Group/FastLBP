/// @file utils.h
///
/// @brief Defines utilities for error checking.
/// @date 2024-09-10

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <iostream>
#include <utils/log.h>

namespace lbp {

#define CUDA_CHECK_ERROR(err) do {                                 \
    cudaError_t err_ = err;                                        \
    if (err_ != cudaSuccess) {                                     \
        LOG_FATAL("CUDA Error: %s", cudaGetErrorString(err_));     \
    }                                                              \
} while(0)

// Use 512 or 256 threads per block
const int kCudaThreadsNum = 128;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;     \
    i < (n);                                                   \
    i += blockDim.x * gridDim.x) 


/// @brief 
/// @tparam T 
/// @param os 
/// @param x 
/// @return 
template<class T>
std::ostream& operator << (std::ostream& os, const thrust::host_vector<T> & x) {
    os << "(";
    for ( typename thrust::host_vector<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << ")";
    return os;
}

/// @brief 
/// @tparam T 
/// @param os 
/// @param x 
/// @return 
template<class T>
std::ostream& operator << (std::ostream& os, const thrust::device_vector<T> & x) {
    os << "(";
    for ( typename thrust::device_vector<T>::const_iterator it = x.begin(); it != x.end(); it++ )
        os << (it != x.begin() ? ", " : "") << *it;
    os << ")";
    return os;
}
    
} // namespace lbp 
