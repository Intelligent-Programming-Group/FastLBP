/// @file factorgraph.h
/// @brief Defines the FactorGraph class, which represents factor graphs.
///
/// @date 2024-7-14

#pragma once

#include <iostream>
#include <vector>
#include <lbp/bipgraph.h>
#include <lbp/factor.h>

namespace lbp {

/// @brief Represents a factor graph.
class FactorGraph {
private:
    /// @brief Stores the neighborhood structure
    BipartiteGraph _G;
    /// @brief Stores the variables
    std::vector<Var> _vars;
    /// @brief Stores the factors
    std::vector<Factor> _factors;

    /// @brief Part of constructors (creates edges, neighbors and adjacency matrix)
    /// @param nrEdges 
    void constructGraph(size_t nrEdges);
public:
    /// @brief Default constructor
    FactorGraph() = default;

    /// @brief Destructor
    ~FactorGraph() = default;

    /// @brief Constructs a factor graph from a vector of factors
    /// @param P 
    FactorGraph(const std::vector<Factor>& P);

    /// @brief 
    /// @param i 
    /// @return constant reference the \a i 'th variable
    const Var &var(size_t i) const;
    /// @brief 
    /// @param I 
    /// @return constant reference to \a I 'th factor
    const Factor &factor(size_t I) const;
    
    /// @brief 
    /// @param i 
    /// @return reference to neighbors of the `i`'th variable
    const Neighbors &nbV(size_t i) const;
    /// @brief 
    /// @param I 
    /// @return constant reference to neighbors of the `I`'th factor
    const Neighbors &nbF(size_t I) const;

    /// @brief 
    /// @return number of variables
    size_t nrVars() const;
    /// @brief 
    /// @return number of factors
    size_t nrFactors() const;

    /// @brief 
    /// @param n 
    /// @return the index of a particular variable
    size_t findVar(const Var &n) const;

    /// @brief Reads a factor graph from a file
    /// @param filename 
    void ReadFromFile(const char *filename);

    /// @brief Reads a factor graph from an input stream
    /// @param is 
    /// @param fg 
    /// @return `is`
    friend std::istream& operator>>(std::istream& is, FactorGraph& fg);
};

}
