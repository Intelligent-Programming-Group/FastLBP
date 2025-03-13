/// @file causal_bp_parall.h
/// @brief
/// 
/// @date 2025-03-07

#pragma once

#include <utils/utils.h>

namespace lbp {
namespace kernel {

void calcProdAll(
    Real *prod_fv_0,
    Real *prod_fv_1,
    size_t *num_zeros_0, 
    size_t *num_zeros_1, 
    const Real *message_fv_0,
    const Real *message_fv_1,
    const size_t *row_ptr_fv,
    size_t n
);

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
);

void calcMessageFVIAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Index2Real *ind_fv_i, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVAndTDAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real2 *ind_fv_and_td, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVOrTDAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real2 *ind_fv_or_td, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVAndTDClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real4 *ind_fv_and_td, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVOrTDClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index5Real4 *ind_fv_or_td, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVAndBUAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real2 *ind_fv_and_bu, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVOrBUAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real2 *ind_fv_or_bu, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVAndBUClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv_and_bu, 
    size_t n, 
    int stream_no=-1
);

void calcMessageFVOrBUClampedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv_or_bu, 
    size_t n, 
    int stream_no=-1
);
    
} // namespace kernel 
} // namespace lbp 
