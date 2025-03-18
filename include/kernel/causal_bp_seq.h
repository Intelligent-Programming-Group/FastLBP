/// @file causal_bp.h
/// @brief
/// 
/// @date 2025-01-14

#pragma once

#include <utils/utils.h>

namespace lbp {
namespace kernel {

#define NSTREAM 9

class StreamHelper {
public:
    cudaStream_t stream[NSTREAM];
    StreamHelper() {}
    ~StreamHelper() {
        for (int i = 0; i < NSTREAM; i++) {
            cudaStreamDestroy(stream[i]);
        }
    }
};

extern StreamHelper stream_helper;

void streamSynchronize();

void streamCreate();

void calcBeliefsV(
    Real *output, 
    const Real *message, 
    const size_t *row_ptr, 
    const size_t *col_ind, 
    size_t n
);


void getBeliefsV(
    Real *output, 
    const Real *prod, 
    const size_t *num_zeros, 
    size_t n
);

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
);

void calcMessageFVI(
    Real *message_fv_0, 
    Real *message_fv_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Index2Real *ind_fv_i, 
    size_t n, 
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);

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
    int stream_no=-1
);
    
} // namespace kernel 
} // namespace lbp 
