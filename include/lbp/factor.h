/// @file factor.h
/// @brief Defines TFactor<> and Factor classes which represent factors in 
/// probability distributions.
///
/// @date 2024-07-14

#pragma once

#include <lbp/prob.h>
#include <lbp/varset.h>

namespace lbp {

/// @brief Represents a (probability) factor.
/// @tparam T 
template<typename T>
class TFactor {
private:
    /// @brief Stores the variables on which the factor depends
    VarSet _vs;
    /// @brief Stores the factor values
    TProb<T> _p;
public:
    TFactor(/* args */) = default;
    ~TFactor() = default;

    /// @brief Constructs factor depending on the variable \a v with uniform 
    /// distribution
    /// @param v 
    TFactor(const Var &v);

    /// @brief Constructs factor depending on variables in \a vars with uniform 
    /// distribution
    /// @param vars 
    TFactor(const VarSet &vars);

    /// @brief Constructs factor depending on variables in `vars` with uniform 
    /// distribution
    /// @param vars 
    /// @param p 
    TFactor(const VarSet& vars, T p);

    /// @brief Constructs factor depending on variables in \a vars, copying 
    /// the values from \a p
    /// @param vars 
    /// @param p 
    TFactor(const VarSet& vars, const TProb<T> &p);

    /// @brief Sets `i`th entry to `val`
    /// @param i 
    /// @param val 
    void set(size_t i, T val);

    /// @brief 
    /// @return Pointer that points to the probabilities of the factor.
    const std::vector<T> &toVec() const;

    /// Returns reference to value vector
    TProb<T>& p() { return _p; }

    /// Returns a copy of the \a i 'th entry of the value vector
    T operator[] (size_t i) const { return _p[i]; }

    /// @brief 
    /// @return constant reference to variable set (i.e., the variables on 
    /// which the factor depends)
    const VarSet& vars() const;

    /// @brief 
    /// @return the number of possible joint states of the variables on which 
    /// the factor depends, \f$\prod_{l\in L} S_l\f$
    size_t nrStates() const;

    T normalize() { return _p.normalize(); }
    /// Returns marginal on \a vars, obtained by summing out all variables except those in \a vars, and normalizing the result if \a normed == \c true
    TFactor<T> marginal(const VarSet &vars, bool normed=true) const;
};

template<typename T>
std::ostream& operator<< (std::ostream& os, const TFactor<T>& f) {
    os << "(" << f.vars() << ", (";
    for ( size_t i = 0; i < f.nrStates(); i++ )
        os << (i == 0 ? "" : ", ") << f[i];
    os << "))";
    return os;
}

template class TFactor<Real>;

typedef TFactor<Real> Factor;

}
