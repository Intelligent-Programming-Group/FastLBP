/// @file var.h
/// @brief Defines class Var, which represents a discrete random variable.
///
/// @date 2024-07-21

#pragma once

#include <stddef.h>
#include <iostream>

namespace lbp {

/// @brief Represents a discrete random variable.
class Var {
private:
    /// @brief Label of the variable (its unique ID)
    size_t _label;
    /// @brief Number of possible values
    size_t _nstates;
public:
    /// @brief Constructs a variable with a given label and number of states
    /// @param label 
    /// @param nstates 
    Var(size_t label = 0, size_t nstates = 0);
    ~Var() = default;

    /// @brief 
    /// @return the label
    size_t label() const;

    /// @brief 
    /// @return the number of states
    size_t states() const;

    /// @brief Smaller-than operator (only compares labels)
    /// @param n 
    /// @return 
    bool operator<(const Var &n) const;
    /// @brief Smaller-than-or-equal-to operator (only compares labels)
    /// @param n 
    /// @return 
    bool operator<=(const Var &n) const;
    /// @brief Not-equal-to operator (only compares labels)
    /// @param n 
    /// @return 
    bool operator!=(const Var &n) const;
    /// @brief Equal-to operator (only compares labels)
    /// @param n 
    /// @return 
    bool operator==(const Var &n) const;

    friend std::ostream& operator<<(std::ostream &os, const Var &n);
};

}
