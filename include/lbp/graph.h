/// @file graph.h
/// @brief
///
/// @date 2024-07-24

#pragma once

#include <stddef.h>
#include <vector>

namespace lbp {

/// @brief Describes the neighbor relationship of two nodes in a graph.
struct Neighbor {
    /// @brief Corresponds to the index of this Neighbor entry in the vector 
    /// of neighbors
    size_t iter;
    /// @brief Contains the absolute index of the neighboring node
    size_t node;
    /// @brief Contains the "dual" index (i.e., the index of this node in the 
    /// Neighbors vector of the neighboring node)
    size_t dual;

    /// @brief Constructor that allows setting the values of the member 
    /// variables
    /// @param iter 
    /// @param node 
    /// @param dual 
    Neighbor(size_t iter, size_t node, size_t dual);

    operator size_t() const;
};

/// @brief Describes the set of neighbors of some node in a graph.
typedef std::vector<Neighbor> Neighbors;

/// @brief Represents an edge in a graph.
typedef std::pair<size_t, size_t> Edge;
    
} // namespace lbp
