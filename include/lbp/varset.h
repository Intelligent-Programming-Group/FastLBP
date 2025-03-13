/// @file varset.h
/// @brief Defines the VarSet class, which represents a set of random variables.
///
/// @date 2024-07-21

#pragma once

#include <iostream>
#include <lbp/smallset.h>
#include <lbp/var.h>

namespace lbp {

/// @brief Represents a set of variables.
class VarSet: public SmallSet<Var> {
private:
    /* data */
public:
    /// @brief Default constructor (constructs an empty set)
    VarSet(/* args */) = default;
    /// Construct from \link SmallSet \endlink<\link Var \endlink> \a x
        VarSet( const SmallSet<Var> &x ) : SmallSet<Var>(x) {}
    /// @brief Construct a VarSet with one element
    /// @param v 
    VarSet(const Var &v);
    ~VarSet() = default;

    /// @brief Construct a `VarSet` from the range between `begin` and `end`.
    /// @tparam VarIterator Iterates over instances of type Var.
    /// @param begin Points to first Var to be added.
    /// @param end Points just beyond last Var to be added.
    /// @param sizeHint For efficiency, the number of elements can be 
    /// speficied by `sizeHint`.
    template<typename VarIterator>
    VarSet(VarIterator begin, VarIterator end, size_t sizeHint = 0): SmallSet<Var>(begin, end, sizeHint) {}

    /// @brief Calculates the number of states of this VarSet, which is simply 
    /// the number of possible joint states of the variables in `*this`.
    /// @return 
    size_t nrStates() const;

    friend std::ostream &operator<<(std::ostream &os, const VarSet &vs);
};
    
}
