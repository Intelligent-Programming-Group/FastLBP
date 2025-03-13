/// @file causal_fg.h
/// @brief Defines `CausalFactorGraph`.
/// 
/// @date 2024-10-19

#pragma once

#include <map>
#include <ext/causal_factor.h>
#include <lbp/bipgraph.h>
#include <utils/log.h>

namespace lbp {

class CausalFactorGraph {
private:
    BipartiteGraph _G;
    std::vector<Var> _vars;
    std::vector<CausalFactor> _factors;

    void setFactor( size_t I, const CausalFactor& newFactor) {
        _factors[I] = newFactor;
    }

    void setFactors(const std::map<size_t, CausalFactor>& facs) {
        for(auto fac = facs.begin(); fac != facs.end(); fac++ ) {
            setFactor( fac->first, fac->second );
        }
    }

    /// @brief Part of constructors (creates edges, neighbors and adjacency matrix)
    /// @param nrEdges 
    void constructGraph( size_t nrEdges );
public:
    CausalFactorGraph(/* args */) = default;
    /// @brief Construct graph with a vector of factors
    /// @param P 
    CausalFactorGraph(const std::vector<CausalFactor> &P);
    ~CausalFactorGraph() = default;

    const Var &var( size_t i ) const;
    const CausalFactor &factor(size_t I) const;
    const std::vector<Var>& vars() const;
    const Neighbors& nbV( size_t i ) const;
    const Neighbor& nbV( size_t i, size_t _I ) const;
    const Neighbors& nbF( size_t I ) const;
    /// @brief 
    /// @return number of variables
    size_t nrVars() const;
    /// @brief 
    /// @return number of factors
    size_t nrFactors() const;
    /// @brief 
    /// @return 
    size_t nrEdges() const;
    /// @brief 
    /// @param n 
    /// @return 
    size_t findVar(const Var &n) const;

    void ReadFromFile(const char *filename);

    /// @brief 
    /// @param i 
    /// @param x 
    void clamp(size_t i, size_t x);

    /// @brief Reads a FactorGraph from an input stream
    /// @param is 
    /// @param fg 
    /// @return 
    friend std::istream &operator>>(std::istream& is, CausalFactorGraph& fg);
};
    
} // namespace lbp 
