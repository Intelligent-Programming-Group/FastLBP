/// @file graph.cpp
///
/// @date 2024-07-24

#include <lbp/graph.h>

namespace lbp {

Neighbor::Neighbor(size_t iter, size_t node, size_t dual): iter(iter), node(node), dual(dual) {}

Neighbor::operator size_t() const {
    return node;
}

} // namespace lbp 
