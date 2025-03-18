/// @file causal_bp_parall.cu
/// @brief
/// 
/// @date 2025-03-07

#include <thrust/device_vector.h>
#include <kernel/causal_bp_parall.h>
#include <kernel/causal_bp_seq.h>
#include <utils/cuda_utils.h>

namespace lbp {
namespace kernel {

extern StreamHelper stream_helper;

inline static __device__ void scale(Real &x, Real &y) {
    Real m = fmax(x, y);
    x /= m;
    y /= m;
}

__global__ void calcProdAllKernel(
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Real *message_fv_0,
    const Real *message_fv_1,
    const size_t *row_ptr_fv,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        size_t start_ind = row_ptr_fv[i], end_ind = row_ptr_fv[i + 1];
        Real prod_0 = 1.0, prod_1 = 1.0;
        size_t zero_0 = 0, zero_1 = 0;
        for (size_t j = start_ind; j < end_ind; j++) {
            Real m0 = message_fv_0[j], m1 = message_fv_1[j];
            if (m0 == 0) {
                zero_0++;
            } else {
                prod_0 *= m0;
            }
            if (m1 == 0) {
                zero_1++;
            } else {
                prod_1 *= m1;
            }
            scale(prod_0, prod_1);
        }
        prod_fv_0[i] = prod_0;
        prod_fv_1[i] = prod_1;
        num_zeros_0[i] = zero_0;
        num_zeros_1[i] = zero_1;
    }
}

__global__ void calcMessageVFAllKernel(
    Real *message_vf_0, 
    Real *message_vf_1, 
    const Real *message_fv_0, 
    const Real *message_fv_1, 
    const Real *prod_fv_0,
    const Real *prod_fv_1,
    const size_t *num_zeros_0, 
    const size_t *num_zeros_1, 
    const Index3 *ind_vf, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index3 index = ind_vf[i];
        size_t ind = index.ind, v = index.v, src_ind = index.src_ind;
        size_t zero_cnt_0 = num_zeros_0[v], zero_cnt_1 = num_zeros_1[v];
        Real m_src_0 = message_fv_0[src_ind], m_src_1 = message_fv_1[src_ind];
        Real prod_0 = prod_fv_0[v], prod_1 = prod_fv_1[v];

        if ((zero_cnt_0 >= 2) || (zero_cnt_0 == 1 && m_src_0 != 0)) {
            prod_0 = 0.0;
        } else if (m_src_0 != 0) {
            prod_0 = prod_0 / m_src_0;
        }
        if ((zero_cnt_1 >= 2) || (zero_cnt_1 == 1 && m_src_1 != 0)) {
            prod_1 = 0.0;
        } else if (m_src_1 != 0) {
            prod_1 = prod_1 / m_src_1;
        }

        Real prod = prod_0 + prod_1;
        message_vf_0[ind] = prod_0 / prod;
        message_vf_1[ind] = prod_1 / prod;
    }
}

__global__ void calcMessageFVIAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Index2Real *ind_fv_i,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index2Real index = ind_fv_i[i];
        size_t ind = index.ind;
        Real p1 = index.prob;
        
        message_fv_0[ind] = 1 - p1;
        message_fv_1[ind] = p1;
    }
}

__global__ void calcMessageFVAndTDAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real2 *ind_fv_and_td, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real2 index = ind_fv_and_td[i];
        size_t ind = index.ind;
        size_t src_ind = index.src_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real prod_1 = 1.0, prod_10 = 1.0;
        Real marg0, marg1;
        if (end_ind - start_ind == 1) {
            marg0 = 1 - p1;
            marg1 = p1;
        } else if (end_ind - start_ind == 2) {
            size_t j = start_ind;
            if (start_ind == src_ind) {
                j++;
            }
            Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
            marg0 = (1 - p1) * m_1 + (1 - p0) * m_0;
            marg1 = p1 * m_1 + p0 * m_0;
        } else {
            Real e1 = 0;
            for (size_t j = start_ind; j < end_ind; j++) {
                if (j != src_ind) {
                    Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
                    Real a0 = m_0 + m_1, a1 = m_1;
                    prod_1 *= a1;
                    prod_10 *= a0;
                    scale(prod_1, prod_10);
                    Real delta = m_0;
                    if (a1 != 0 && a0 == a1 && delta != 0) {
                        e1 += delta / a1;
                    }
                }
            }
            marg0 = (1 - p1) * prod_1 + (1 - p0) * (e1 * prod_10 + (prod_10 - prod_1));
            marg1 = p1 * prod_1 + p0 * (e1 * prod_10 + (prod_10 - prod_1));
        }
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            // printf("p0=%lf p1=%lf prod1=%lf prod10=%lf\n", p0, p1, prod_1, prod_10);
            marg0 = 0.5;
            marg1 = 0.5;
        }

        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVOrTDAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real2 *ind_fv_or_td, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real2 index = ind_fv_or_td[i];
        size_t ind = index.ind;
        size_t src_ind = index.src_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        Real prod_0 = 1.0, prod_10 = 1.0;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real marg0, marg1;
        if (end_ind - start_ind == 1) {
            marg0 = p1;
            marg1 = 1 - p1;
        } else if (end_ind - start_ind == 2) {
            size_t j = start_ind;
            if (start_ind == src_ind) {
                j++;
            }
            Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
            marg0 = p1 * m_0 + p0 * m_1;
            marg1 = (1 - p1) * m_0 + (1 - p0) * m_1;
        } else {    
            Real e1 = 0;
            for (size_t j = start_ind; j < end_ind; j++) {
                if (j != src_ind) {
                    Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
                    Real a0 = m_0 + m_1, a1 = m_0;
                    prod_0 *= a1;
                    prod_10 *= a0;
                    scale(prod_0, prod_10);
                    Real delta = m_1;
                    if (a1 != 0 && a0 == a1 && delta != 0) {
                        e1 += delta / a1;
                    }
                }
            }
            marg0 = p1 * prod_0 + p0 * (e1 * prod_10 + (prod_10 - prod_0));
            marg1 = (1 - p1) * prod_0 + (1 - p0) * (e1 * prod_10 + (prod_10 - prod_0));
        }
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {  
            // printf("503 p0=%lf p1=%lf prod0=%lf prod10=%lf\n", p0, p1, prod_0, prod_10); 
            marg0 = 0.5;
            marg1 = 0.5;
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVAndTDClampedAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real4 *ind_fv_and_td,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real4 index = ind_fv_and_td[i];
        size_t ind = index.ind, src_ind = index.src_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real mask0 = index.mask0, mask1 = index.mask1;
        Real prod_1 = 1.0, prod_10 = 1.0;
        Real marg0, marg1;
        if (end_ind - start_ind == 1) {
            marg0 = 1 - p1;
            marg1 = p1;
        } else if (end_ind - start_ind == 2) {
            size_t j = start_ind;
            if (start_ind == src_ind) {
                j++;
            }
            Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
            marg0 = (1 - p1) * m_1 + (1 - p0) * m_0;
            marg1 = p1 * m_1 + p0 * m_0;
        } else {
            Real e1 = 0;
            for (size_t j = start_ind; j < end_ind; j++) {
                if (j != src_ind) {
                    Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
                    Real a0 = m_0 + m_1, a1 = m_1;
                    prod_1 *= a1;
                    prod_10 *= a0;
                    scale(prod_1, prod_10);
                    Real delta = m_0;
                    if (a1 != 0 && a0 == a1 && delta != 0) {
                        e1 += delta / a1;
                    }
                }
            }
            marg0 = (1 - p1) * prod_1 + (1 - p0) * (e1 * prod_10 + (prod_10 - prod_1));
            marg1 = p1 * prod_1 + p0 * (e1 * prod_10 + (prod_10 - prod_1));
        }
        marg0 *= mask0;
        marg1 *= mask1;
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            // printf("p0=%lf p1=%lf prod1=%lf prod10=%lf\n", p0, p1, prod_1, prod_10);
            if (mask0 == 0) {
                marg0 = 0.0;
                marg1 = 1.0;
            } else if (mask1 == 0) {
                marg0 = 1.0;
                marg1 = 0.0;
            } else {
                marg0 = 0.5;
                marg1 = 0.5;
            }
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVOrTDClampedAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real4 *ind_fv_or_td,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real4 index = ind_fv_or_td[i];
        size_t ind = index.ind, src_ind = index.src_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real mask0 = index.mask0, mask1 = index.mask1;
        Real prod_0 = 1.0, prod_10 = 1.0;
        Real marg0, marg1;
        if (end_ind - start_ind == 1) {
            marg0 = p1;
            marg1 = 1 - p1;
        } else if (end_ind - start_ind == 2) {
            size_t j = start_ind;
            if (start_ind == src_ind) {
                j++;
            }
            Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
            marg0 = p1 * m_0 + p0 * m_1;
            marg1 = (1 - p1) * m_0 + (1 - p0) * m_1;
        } else {    
            Real e1 = 0;
            for (size_t j = start_ind; j < end_ind; j++) {
                if (j != src_ind) {
                    Real m_0 = message_vf_0[j], m_1 = message_vf_1[j];
                    Real a0 = m_0 + m_1, a1 = m_0;
                    prod_0 *= a1;
                    prod_10 *= a0;
                    scale(prod_0, prod_10);
                    Real delta = m_1;
                    if (a1 != 0 && a0 == a1 && delta != 0) {
                        e1 += delta / a1;
                    }
                }
            }
            marg0 = p1 * prod_0 + p0 * (e1 * prod_10 + (prod_10 - prod_0));
            marg1 = (1 - p1) * prod_0 + (1 - p0) * (e1 * prod_10 + (prod_10 - prod_0));
        }
        marg0 *= mask0;
        marg1 *= mask1;
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            if (mask0 == 0) {
                marg0 = 0.0;
                marg1 = 1.0;
            } else if (mask1 == 0) {
                marg0 = 1.0;
                marg1 = 0.0;
            } else {
                marg0 = 0.5;
                marg1 = 0.5;
            }
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVAndBUAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real2 *ind_fv_and_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real2 index = ind_fv_and_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind;
        size_t h = index.head;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        Real m0h = message_vf_0[h], m1h = message_vf_1[h];
        Real prod_1 = (p1 - p0) * (m1h - m0h);
        Real prod_10 = p0 * m1h + (1 - p0) * m0h;
        scale(prod_1, prod_10);
        Real marg0, marg1;
        if (prod_10 != 0) {
            for (size_t j = start_ind; j < end_ind; j++) {
                if ((j != src_ind) && (j != h)) {
                    Real m0 = message_vf_0[j], m1 = message_vf_1[j], m = m0 + m1;
                    prod_10 *= m;
                    prod_1 *= m1;
                    scale(prod_1, prod_10);
                }
            }
        }

        marg0 = prod_10;
        marg1 = prod_10 + prod_1;
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            // printf("%d prod1=%lf prod10=%lf\n",(int)__LINE__, prod_1, prod_10);
            marg0 = 0.5;
            marg1 = 0.5;
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVOrBUAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real2 *ind_fv_or_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real2 index = ind_fv_or_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind;
        size_t h = index.head;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        Real m0h = message_vf_0[h], m1h = message_vf_1[h];
        Real prod_0 = (p1 - p0) * (m0h - m1h);
        Real prod_10 = p0 * m0h + (1 - p0) * m1h;
        scale(prod_0, prod_10);
        Real marg0, marg1;
        if (prod_10 != 0) {
            for (size_t j = start_ind; j < end_ind; j++) {
                if ((j != src_ind) && (j != h)) {
                    Real m0 = message_vf_0[j], m1 = message_vf_1[j], m = m0 + m1; 
                    prod_10 *= m;
                    prod_0 *= m0;
                    scale(prod_0, prod_10);
                }
            }
        }

        marg0 = prod_0 + prod_10;
        marg1 = prod_10;
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            // printf("%d prod0=%lf prod10=%lf\n", (int)__LINE__, prod_0, prod_10);
            marg0 = 0.5;
            marg1 = 0.5;
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVAndBUClampedAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv_and_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real4 index = ind_fv_and_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind;
        size_t h = index.head;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        Real mask0 = index.mask0, mask1 = index.mask1;
        Real m0h = message_vf_0[h], m1h = message_vf_1[h];
        Real prod_1 = (p1 - p0) * (m1h * mask1 - m0h * mask0);
        Real prod_10 = p0 * m1h * mask1 + (1 - p0) * m0h * mask0;
        scale(prod_1, prod_10);
        Real marg0, marg1;
        if (prod_10 != 0) {
            for (size_t j = start_ind; j < end_ind; j++) {
                if ((j != src_ind) && (j != h)) {
                    Real m0 = message_vf_0[j], m1 = message_vf_1[j], m = m0 + m1;
                    prod_10 *= m;
                    prod_1 *= m1;
                    scale(prod_1, prod_10);
                }
            }
        }

        marg0 = prod_10;
        marg1 = prod_10 + prod_1;
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            // printf("%d prod1=%lf prod10=%lf\n",(int)__LINE__, prod_1, prod_10);
            marg0 = 0.5;
            marg1 = 0.5;
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

__global__ void calcMessageFVOrBUClampedAllKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv_or_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real4 index = ind_fv_or_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind;
        size_t h = index.head;
        size_t start_ind = index.start_ind, end_ind = index.end_ind;
        Real p0 = index.prob_default, p1 = index.prob;
        Real mask0 = index.mask0, mask1 = index.mask1;
        Real m0h = message_vf_0[h], m1h = message_vf_1[h];
        Real prod_0 = (p1 - p0) * (m0h * mask0 - m1h * mask1);
        Real prod_10 = p0 * m0h * mask0 + (1 - p0) * m1h * mask1;
        scale(prod_0, prod_10);
        Real marg0, marg1;
        if (prod_10 != 0) {
            for (size_t j = start_ind; j < end_ind; j++) {
                if ((j != src_ind) && (j != h)) {
                    Real m0 = message_vf_0[j], m1 = message_vf_1[j], m = m0 + m1; 
                    prod_10 *= m;
                    prod_0 *= m0;
                    scale(prod_0, prod_10);
                }
            }
        }

        marg0 = prod_0 + prod_10;
        marg1 = prod_10;
        Real marg = marg0 + marg1;
        if (marg != 0) {
            marg0 = marg0 / marg;
            marg1 = marg1 / marg;
        } else {   
            // printf("%d prod0=%lf prod10=%lf\n", (int)__LINE__, prod_0, prod_10);
            marg0 = 0.5;
            marg1 = 0.5;
        }
        message_fv_0[ind] = marg0;
        message_fv_1[ind] = marg1;
    }
}

void calcProdAll(
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Real *message_fv_0,
    const Real *message_fv_1,
    const size_t *row_ptr_fv,
    size_t n
) {
    calcProdAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
        prod_fv_0,
        prod_fv_1,
        num_zeros_0, 
        num_zeros_1, 
        message_fv_0, 
        message_fv_1, 
        row_ptr_fv, 
        n
    );
}

void calcMessageVFAll(
    Real *message_vf_0, 
    Real *message_vf_1, 
    const Real *message_fv_0, 
    const Real *message_fv_1, 
    const Real *prod_fv_0,
    const Real *prod_fv_1,
    const size_t *num_zeros_0, 
    const size_t *num_zeros_1, 
    const Index3 *ind_vf, 
    size_t n
) {
    calcMessageVFAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
        message_vf_0, 
        message_vf_1, 
        message_fv_0, 
        message_fv_1, 
        prod_fv_0,
        prod_fv_1,
        num_zeros_0, 
        num_zeros_1, 
        ind_vf, 
        n
    );
}

void calcMessageFVIAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Index2Real *ind_fv_i, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVIAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            ind_fv_i, 
            n
        );
    } else {
        calcMessageFVIAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            ind_fv_i, 
            n
        );
    }
}

void calcMessageFVAndTDAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real2 *ind_fv_and_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndTDAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_td, 
            n
        );
    } else {
        calcMessageFVAndTDAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_td, 
            n
        );
    }
}

void calcMessageFVOrTDAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real2 *ind_fv_or_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrTDAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_td, 
            n
        );
    } else {
        calcMessageFVOrTDAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_td, 
            n
        );
    }
}

void calcMessageFVAndTDClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real4 *ind_fv_and_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndTDClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_td, 
            n
        );
    } else {
        calcMessageFVAndTDClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_td, 
            n
        );
    }
}

void calcMessageFVOrTDClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real4 *ind_fv_or_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrTDClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_td,
            n
        );
    } else {
        calcMessageFVOrTDClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_td,
            n
        );
    }
}

void calcMessageFVAndBUAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real2 *ind_fv_and_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndBUAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_bu, 
            n
        );
    } else {
        calcMessageFVAndBUAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_bu, 
            n
        );
    }
}

void calcMessageFVOrBUAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real2 *ind_fv_or_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrBUAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_bu, 
            n
        );
    } else {
        calcMessageFVOrBUAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_bu, 
            n
        );
    }
}

void calcMessageFVAndBUClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv_and_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndBUClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_bu, 
            n
        );
    } else {
        calcMessageFVAndBUClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_and_bu, 
            n
        );
    }
}

void calcMessageFVOrBUClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv_or_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrBUClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_bu, 
            n
        );
    } else {
        calcMessageFVOrBUClampedAllKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            ind_fv_or_bu, 
            n
        );
    }
}

} // namespace kernel 
} // namespace lbp 
