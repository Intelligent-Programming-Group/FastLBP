/// @file factorgraph.cpp
///
/// @date 2024-07-14

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <lbp/factor.h>
#include <lbp/factorgraph.h>
#include <lbp/graph.h>
#include <lbp/index.h>
#include <lbp/var.h>
#include <lbp/varset.h>
#include <utils/utils.h>

namespace lbp {

void FactorGraph::constructGraph(size_t nrEdges) {
    // create a mapping for indices
    std::unordered_map<size_t, size_t> hashmap;
    
    for (size_t i = 0; i < _vars.size(); i++) {
        hashmap[_vars[i].label()] = i;
    }

    // create edge list
    std::vector<Edge> edges;
    edges.reserve(nrEdges);
    for (size_t i2 = 0; i2 < _factors.size(); i2++) {
        const VarSet& ns = _factors[i2].vars();
        for (auto q: ns) {
            edges.push_back(Edge(hashmap[q.label()], i2));
        }
    }

    // create bipartite graph
    _G.construct(_vars.size(), _factors.size(), edges.begin(), edges.end());
}

FactorGraph::FactorGraph(const std::vector<Factor>& P): _G() {
    // add factors, obtain variables
    std::set<Var> varset;
    _factors.reserve(P.size());
    size_t nrEdges = 0;
    for (auto p2: P) {
        _factors.push_back(p2);
        std::copy(p2.vars().begin(), p2.vars().end(), std::inserter(varset, varset.begin()));
        nrEdges += p2.vars().size();
    }

    // add vars
    _vars.reserve(varset.size());
    for (auto p1: varset) {
        _vars.push_back(p1);
    }

    // create graph structure
    constructGraph(nrEdges);
}

const Var &FactorGraph::var(size_t i) const {
    return _vars[i];
}

const Factor &FactorGraph::factor(size_t I) const {
    return _factors[I];
}

const Neighbors &FactorGraph::nbV(size_t i) const {
    return _G.nb1(i);
}

const Neighbors &FactorGraph::nbF(size_t I) const {
    return _G.nb2(I);;
}

size_t FactorGraph::nrVars() const {
    return _vars.size();
}

size_t FactorGraph::nrFactors() const {
    return _factors.size();
}

size_t FactorGraph::findVar(const Var &n) const {
    size_t i = find( _vars.begin(), _vars.end(), n ) - _vars.begin();
    if( i == nrVars() ) {
        std::cerr << "var not found \n";
    }
    return i;
}

void FactorGraph::ReadFromFile(const char *filename) {
    std::ifstream infile;
    infile.open(filename);
    if (infile.is_open()) {
        infile >> *this;
        infile.close();
    } else {
        exit(1);
    }
}

std::istream& operator>>(std::istream& is, FactorGraph& fg) {
    long verbose = 0;

    std::vector<Factor> facs;
    size_t nr_Factors;
    std::string line;

    read_annotation(is, line);

    is >> nr_Factors;
    if (is.fail()) {
        exit(1);
    }
    if (verbose >= 1) {
        std::cerr << "Reading " << nr_Factors << " factors..." << std::endl;
    }

    getline(is, line);
    if (is.fail() || line.size() > 0) {
        exit(1);
    }

    std::map<long, size_t> vardims;
    for (size_t I = 0; I < nr_Factors; I++) {
        if (verbose >= 2) {
            std::cerr << "Reading factor " << I << "..." << std::endl;
        }
        size_t nr_members;
        read_annotation(is, line);
        is >> nr_members;
        if (verbose >= 2) {
            std::cerr << "  nr_members: " << nr_members << std::endl;
        }

        std::vector<long> labels;
        for (size_t mi = 0; mi < nr_members; mi++) {
            long mi_label;
            read_annotation(is, line);
            is >> mi_label;
            labels.push_back(mi_label);
        }
        if (verbose >= 2) {
            std::cerr << "  labels: " << labels << std::endl;
        }

        std::vector<size_t> dims;
        for (size_t mi = 0; mi < nr_members; mi++) {
            size_t mi_dim;
            read_annotation(is, line);
            is >> mi_dim;
            dims.push_back(mi_dim);
        }
        if (verbose >= 2) {
            std::cerr << "  dimensions: " << dims << std::endl;
        }

        // add the Factor
        std::vector<Var> Ivars;
        Ivars.reserve(nr_members);
        for (size_t mi = 0; mi < nr_members; mi++) {
            auto vdi = vardims.find(labels[mi]);
            if (vdi != vardims.end()) {
                // check whether dimensions are consistent
                if (vdi->second != dims[mi]) {
                    exit(1);
                }
            } else {
                vardims[labels[mi]] = dims[mi];
            }
            Ivars.push_back(Var(labels[mi], dims[mi]));
        }
        facs.push_back(Factor(VarSet(Ivars.begin(), Ivars.end(), Ivars.size()), (Real)0.0));
        if (verbose >= 2) {
            std::cerr << "  vardims: " << vardims << std::endl;
        }

        // calculate permutation object
        Permute permindex(Ivars);

        // read values
        size_t nr_nonzeros;
        read_annotation(is, line);
        is >> nr_nonzeros;
        if (verbose >= 2) {
            std::cerr << "  nonzeroes: " << nr_nonzeros << std::endl;
        }
        for (size_t k = 0; k < nr_nonzeros; k++) {
            size_t li;
            Real val;
            read_annotation(is, line);
            is >> li;
            read_annotation(is, line);
            is >> val;

            // store value, but permute indices first according to internal representation
            facs.back().set(permindex.convertLinearIndex(li), val);
        }
    }

    if (verbose >= 3) {
        std::cout << "factors:" << facs << std::endl;
    }

    fg = FactorGraph(facs);

    return is;
}

}
