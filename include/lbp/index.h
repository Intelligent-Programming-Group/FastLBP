/// @file index.h
/// @brief 
///
/// @date 2024-07-23

#pragma once

#include <vector>
#include <lbp/varset.h>

namespace lbp {

/// @brief Tool for looping over the states of several variables.
class IndexFor {
private:
    /// @brief The current linear index corresponding to the state of indexVars
    long _index;
    /// @brief For each variable in forVars, the amount of change in _index
    std::vector<long> _sum;
    /// @brief For each variable in forVars, the current state
    std::vector<size_t> _state;
    /// @brief For each variable in forVars, its number of possible values
    std::vector<size_t> _ranges;
public:
    /// @brief Default constructor
    IndexFor() = default;
    /// @brief Construct IndexFor object from \a indexVars and \a forVars
    /// @param indexVars 
    /// @param forVars 
    IndexFor(const VarSet &indexVars, const VarSet &forVars);

    /// @brief Conversion to `size_t`: returns linear index of the current 
    /// state of indexVars
    operator size_t() const;
    /// @brief Increments the current state of \a forVars (prefix)
    /// @return 
    IndexFor &operator++();
    /// @brief Increments the current state of \a forVars (postfix)
    /// @param  
    void operator++(int);

    /// @brief
    /// @return true if the current state is valid
    bool valid() const;
};

/// @brief Tool for calculating permutations of linear indices of 
/// multi-dimensional arrays.
class Permute {
private:
    /// @brief Stores the number of possible values of all indices
    std::vector<size_t>  _ranges;
    /// @brief Stores the permutation
    std::vector<size_t>  _sigma;
public:
    Permute(/* args */) = default;
    ~Permute() = default;

    /// @brief Construct from vector of variables.
    /// @param vars 
    /// @param reverse 
    Permute(const std::vector<Var>& vars, bool reverse = false);

    /// @brief Calculates a permuted linear index.
    /// @param li 
    /// @return 
    size_t convertLinearIndex(size_t li) const;
};
    
} // namespace lbp

