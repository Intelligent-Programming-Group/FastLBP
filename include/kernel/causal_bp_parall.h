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
    Size *num_zeros_0, 
    Size *num_zeros_1, 
    const Real *message_fv_0,
    const Real *message_fv_1,
    const Size *row_ptr_fv,
    size_t n
);

void calcMessageVFAll(
    Real *message_vf_0, 
    Real *message_vf_1, 
    const Real *message_fv_0, 
    const Real *message_fv_1, 
    const Real *prod_fv_0,
    const Real *prod_fv_1,
    const Size *num_zeros_0, 
    const Size *num_zeros_1, 
    const Index3 *ind_vf, 
    size_t n
);

void calcMessageFVFusedAll(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    const Index6Real4 *ind_fv, 
    size_t n
);

} // namespace kernel 
} // namespace lbp 
