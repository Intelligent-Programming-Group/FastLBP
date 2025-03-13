/// @file bipgraph.cpp
///
/// @date 2024-07-24

#include <lbp/bipgraph.h>
#include <iostream>

namespace lbp {

const Neighbors &BipartiteGraph::nb1(size_t i1) const {
    if (i1 > _nb1.size()) {
        
    }
    return _nb1[i1];
}

const Neighbors &BipartiteGraph::nb2(size_t i2) const {
    if (i2 > _nb2.size()) {
        
    }
    return _nb2[i2];
}

BipartiteGraph& BipartiteGraph::addEdge(size_t n1, size_t n2, bool check) {
    bool exists = false;
    if (check) {
        // Check whether the edge already exists
        for (auto &nb2: _nb1[n1]) {
            if (nb2 == n2) {
                exists = true;
                break;
            }
        }
    }
    if (!exists) { // Add edge
        Neighbor nb_1(_nb1[n1].size(), n2, _nb2[n2].size());
        Neighbor nb_2(nb_1.dual, n1, nb_1.iter);
        _nb1[n1].push_back(nb_1);
        _nb2[n2].push_back(nb_2);
    }
    return *this;
}

size_t BipartiteGraph::nrEdges() const {
    size_t sum = 0;
    for( size_t i1 = 0; i1 < nrNodes1(); i1++ )
        sum += nb1(i1).size();
    return sum;
}

/// Returns number of nodes of type 1
size_t BipartiteGraph::nrNodes1() const {
    return _nb1.size(); 
}
    
} // namespace lbp 
