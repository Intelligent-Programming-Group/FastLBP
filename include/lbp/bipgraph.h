/// @file bipgraph.h
/// @brief Defines the BipartiteGraph class, which represents a bipartite graph.
///
/// @date 2024-07-14

#pragma once

#include <lbp/graph.h>

namespace lbp {

/// @brief Represents the neighborhood structure of nodes in an undirected, 
/// bipartite graph.
class BipartiteGraph {
private:
    /// @brief Contains for each node of type 1 a vector of its neighbors
    std::vector<Neighbors> _nb1;
    /// @brief Contains for each node of type 2 a vector of its neighbors
    std::vector<Neighbors> _nb2;
public:
    BipartiteGraph(/* args */) = default;
    ~BipartiteGraph() = default;

    /// @brief 
    /// @param i1 
    /// @return constant reference to all neighbors of node \a i1 of type 1
    const Neighbors &nb1(size_t i1) const;
    /// @brief 
    /// @param i2
    /// @return constant reference to all neighbors of node \a i1 of type 2
    const Neighbors &nb2(size_t i2) const;
    
    template<typename EdgeInputIterator>
    void construct(size_t nrNodes1, size_t nrNodes2, EdgeInputIterator begin, EdgeInputIterator end, bool check=true);

    /// @brief Adds an edge between `n1` of type 1 and `n2`.
    /// @param n1 a node of type 1
    /// @param n2 a node of type 2
    /// @param check If `check` == `true`, only adds the edge if it does not 
    /// exist already.
    /// @return 
    BipartiteGraph& addEdge(size_t n1, size_t n2, bool check = true);

    /// @brief 
    /// @return number of nodes of type 1
    size_t nrNodes1() const;

    /// @brief Calculates the number of edges, time complexity: O(nrNodes1())
    /// @return 
    size_t nrEdges() const;
};

/// @brief (Re)constructs BipartiteGraph from a range of edges.
/// @tparam EdgeInputIterator Iterator that iterates over instances of Edge.
/// @param nrNodes1 The number of nodes of type 1, i.e., variables.
/// @param nrNodes2 The number of nodes of type 2, i.e., factors.
/// @param begin Points to the first edge.
/// @param end Points just beyond the last edge.
/// @param check Whether to only add an edge if it does not exist already.
template<typename EdgeInputIterator>
void BipartiteGraph::construct(size_t nrNodes1, size_t nrNodes2, EdgeInputIterator begin, EdgeInputIterator end, bool check) {
    _nb1.clear();
    _nb1.resize(nrNodes1);
    _nb2.clear();
    _nb2.resize(nrNodes2);
    for (EdgeInputIterator e = begin; e != end; e++) {
        addEdge(e->first, e->second, check);
    }
}

}
