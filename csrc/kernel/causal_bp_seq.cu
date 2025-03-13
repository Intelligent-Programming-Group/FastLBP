/// @file causal_bp_seq.cu
/// @brief
/// 
/// @date 2025-01-14

#include <cusparse.h>
#include <thrust/device_vector.h>
#include <kernel/causal_bp_seq.h>
#include <utils/cuda_utils.h>

namespace lbp {
namespace kernel {

StreamHelper stream_helper;

void streamSynchronize() {
    for (int i = 0; i < NSTREAM; i++) {
        cudaStreamSynchronize(stream_helper.stream[i]);
    }
}

inline static __device__ void scale(Real &x, Real &y) {
    Real m = fmax(x, y);
    // if (m == 0.0) {
    //     printf("scale error\n");
    // }
    x /= m;
    y /= m;
}

// static __device__ void scale_debug(Real &x, Real &y, size_t lineno) {
//     Real m = x <= y ? y : x;
//     if (m == 0.0) {
//         printf("%d scale error\n", (int)lineno);
//     }
//     x /= m;
//     y /= m;
// }

inline __device__ void update_message(
    Real *message_fv_0, Real *message_fv_1, 
    Real *prod_fv_0, Real *prod_fv_1, 
    size_t *num_zeros_0, size_t *num_zeros_1, 
    size_t ind, size_t v, Real marg0, Real marg1
) {
    Real old_msg_0 = message_fv_0[ind], old_msg_1 = message_fv_1[ind];
    Real prod_0 = prod_fv_0[v], prod_1 = prod_fv_1[v];
    size_t zero_cnt_0 = num_zeros_0[v], zero_cnt_1 = num_zeros_1[v];

    // The code below is the optimized version of the code in the annotation.
    // if (old_msg_0 == 0) {
    //     zero_cnt_0--;
    // } else {
    //     prod_0 /= old_msg_0;
    // }
    // if (marg0 == 0) {
    //     zero_cnt_0++;
    // } else {
    //     prod_0 *= marg0;
    // }
    // if (old_msg_1 == 0) {
    //     zero_cnt_1--;
    // } else {
    //     prod_1 /= old_msg_1;
    // }
    // if (marg1 == 0) {
    //     zero_cnt_1++;
    // } else {
    //     prod_1 *= marg1;
    // }
    // Optimize using masks.
    Real mask0 = old_msg_0 == 0;
    Real mask1 = marg0 == 0;
    zero_cnt_0 += mask1 - mask0;
    prod_0 *= ((1.0 - mask1) * marg0 + mask1) / ((1.0 - mask0) * old_msg_0 + mask0);

    Real mask2 = old_msg_1 == 0;
    Real mask3 = marg1 == 0;
    zero_cnt_1 += mask3 - mask2;
    prod_1 *= ((1.0 - mask3) * marg1 + mask3) / ((1.0 - mask2) * old_msg_1 + mask2);
    scale(prod_0, prod_1);

    prod_fv_0[v] = prod_0;
    prod_fv_1[v] = prod_1;
    num_zeros_0[v] = zero_cnt_0;
    num_zeros_1[v] = zero_cnt_1;
    message_fv_0[ind] = marg0;
    message_fv_1[ind] = marg1;
}

__global__ void calcBeliefsVKernel(
    Real *output, 
    const Real *message, 
    const size_t *row_ptr, 
    const size_t *col_ind, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        size_t start_ind = row_ptr[i], end_ind = row_ptr[i + 1];
        Real prod = 1.0;
        for (size_t j = start_ind; j < end_ind; j++) {
            prod *= message[j];
        }
        output[i] = prod;
    }
}

__global__ void getBeliefsVKernel(
    Real *output, 
    const Real *prod, 
    const size_t *num_zeros, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        output[i] = prod[i] * (!num_zeros[i]);
    }
}

__global__ void calcMessageVFKernel(
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

__global__ void calcMessageFVIKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index2Real *ind_fv_i,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index2Real index = ind_fv_i[i];
        size_t ind = index.ind, v = index.v;
        Real p1 = index.prob;

        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, 1 - p1, p1
        );
    }
}

__global__ void calcMessageFVAndTDKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real2 *ind_fv_and_td, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real2 index = ind_fv_and_td[i];
        size_t ind = index.ind;
        size_t src_ind = index.src_ind, v = index.v;
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

        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVOrTDKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real2 *ind_fv_or_td, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real2 index = ind_fv_or_td[i];
        size_t ind = index.ind;
        size_t src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVAndTDClampedKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real4 *ind_fv_and_td,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real4 index = ind_fv_and_td[i];
        size_t ind = index.ind, src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVOrTDClampedKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real4 *ind_fv_or_td,
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index5Real4 index = ind_fv_or_td[i];
        size_t ind = index.ind, src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVAndBUKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real2 *ind_fv_and_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real2 index = ind_fv_and_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVOrBUKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real2 *ind_fv_or_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real2 index = ind_fv_or_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVAndBUClampedKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real4 *ind_fv_and_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real4 index = ind_fv_and_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

__global__ void calcMessageFVOrBUClampedKernel(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real4 *ind_fv_or_bu, 
    size_t n
) {
    CUDA_KERNEL_LOOP(i, n) {
        Index6Real4 index = ind_fv_or_bu[i];
        size_t ind = index.ind, src_ind = index.src_ind, v = index.v;
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
        update_message(
            message_fv_0, message_fv_1, prod_fv_0, prod_fv_1, 
            num_zeros_0, num_zeros_1, ind, v, marg0, marg1
        );
    }
}

void calcBeliefsV(
    Real *output, 
    const Real *message, 
    const size_t *row_ptr, 
    const size_t *col_ind, 
    size_t n
) {
    calcBeliefsVKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
        output, message, row_ptr, col_ind, n
    );
}

void getBeliefsV(
    Real *output, 
    const Real *prod, 
    const size_t *num_zeros, 
    size_t n
) {
    getBeliefsVKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
        output, prod, num_zeros, n
    );
}

void calcMessageVF(
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
    calcMessageVFKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
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

void calcMessageFVI(
    Real *message_fv_0, 
    Real *message_fv_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index2Real *ind_fv_i, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVIKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_i, 
            n
        );
    } else {
        calcMessageFVIKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_i, 
            n
        );
    }
}

void calcMessageFVAndTD(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real2 *ind_fv_and_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndTDKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_td, 
            n
        );
    } else {
        calcMessageFVAndTDKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_td, 
            n
        );
    }
}

void calcMessageFVOrTD(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real2 *ind_fv_or_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrTDKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_td, 
            n
        );
    } else {
        calcMessageFVOrTDKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_td, 
            n
        );
    }
}

void calcMessageFVAndTDClamped(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real4 *ind_fv_and_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndTDClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_td, 
            n
        );
    } else {
        calcMessageFVAndTDClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_td, 
            n
        );
    }
}

void calcMessageFVOrTDClamped(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index5Real4 *ind_fv_or_td, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrTDClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_td,
            n
        );
    } else {
        calcMessageFVOrTDClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_td,
            n
        );
    }
}

void calcMessageFVAndBU(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real2 *ind_fv_and_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndBUKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_bu, 
            n
        );
    } else {
        calcMessageFVAndBUKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_bu, 
            n
        );
    }
}

void calcMessageFVOrBU(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real2 *ind_fv_or_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrBUKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_bu, 
            n
        );
    } else {
        calcMessageFVOrBUKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_bu, 
            n
        );
    }
}

void calcMessageFVAndBUClamped(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real4 *ind_fv_and_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVAndBUClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_bu, 
            n
        );
    } else {
        calcMessageFVAndBUClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_and_bu, 
            n
        );
    }
}

void calcMessageFVOrBUClamped(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index6Real4 *ind_fv_or_bu, 
    size_t n, 
    int stream_no
) {
    if (n == 0) {
        return;
    }
    if (stream_no >= NSTREAM || stream_no < 0) {
        calcMessageFVOrBUClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_bu, 
            n
        );
    } else {
        calcMessageFVOrBUClampedKernel<<<CudaGetBlocks(n), kCudaThreadsNum, 0, stream_helper.stream[stream_no]>>>(
            message_fv_0, 
            message_fv_1, 
            message_vf_0, 
            message_vf_1, 
            prod_fv_0,
            prod_fv_1,
            num_zeros_0, 
            num_zeros_1, 
            ind_fv_or_bu, 
            n
        );
    }
}

} // namespace kernel 
} // namespace lbp 
