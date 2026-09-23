/// @file causal_bp.h
/// @brief
/// 
/// @date 2025-01-14

#pragma once

#include <utils/utils.h>

namespace lbp {
namespace kernel {

void calcBeliefsV(
    Real *output, 
    const Real *message, 
    const Size *row_ptr, 
    const Size *col_ind, 
    size_t n
);

void getBeliefsV(
    Real *output, 
    const Real *prod, 
    const Size *num_zeros, 
    size_t n
);

void calcMessageVF(
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

void calcMessageFVFused(
    Real *message_fv_0, 
    Real *message_fv_1, 
    const Real *message_vf_0, 
    const Real *message_vf_1, 
    Real *prod_fv_0,
    Real *prod_fv_1,
    Size *num_zeros_0, 
    Size *num_zeros_1, 
    const Index7Real4 *ind_fv, 
    size_t n
);
    
} // namespace kernel 
} // namespace lbp 
