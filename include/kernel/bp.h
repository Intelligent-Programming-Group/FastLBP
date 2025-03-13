/// @file bp.h
/// @brief
///
/// @date 2024-07-25

#pragma once

#include <stddef.h>
#include <utils/utils.h>

namespace lbp {
namespace kernel {

void initArray(Real *arr, Real val, size_t size);

void calcProduct(
    Real *output,
    const Real *oldMessage,
    size_t **nbs,
    const size_t *numNbs,
    const size_t size
);

void marginalize(
    Real *output, 
    const Real *joint, 
    size_t **index, 
    const size_t *numIndex, 
    const size_t size
);

void normalize(Real *vecs, size_t *offsets, size_t *lengths, size_t size);

void dist(Real *p, const Real *q, size_t size);

Real calcMax(Real *p, size_t size);

} // namespace kernel
} // namespace lbp
