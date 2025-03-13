/// @file causal_factor.h
/// @brief Defines `CausalFactor`
///
/// @date 2024-10-19

#pragma once

#include <lbp/prob.h>
#include <lbp/varset.h>

namespace lbp {

/// @brief 
class CausalFactor {
public:
    /// @brief 
    enum CausalType {
        Singleton = 'I',
        DefiniteAnd = '*',
        DefiniteOr = '+',
    };

    bool head_clamped;
    Prob head_mask;

private:
    /// @brief 
    CausalType _type;
    /// @brief 
    Var _head;
    /// @brief 
    VarSet _body;
    /// @brief 
    Real _p;
    /// @brief 
    Real _q = 0;
    /// @brief 
    VarSet _vs;

public:
    /// @brief Constructor of a `Singleton` factor.
    /// @param v 
    /// @param p 
    CausalFactor(const Var &v, Real p);
    /// @brief Constructor of a `DefiniteAnd` or `DefiniteOr` factor.
    /// @param head 
    /// @param body 
    /// @param isAnd 
    /// @param p 
    /// @param q 
    CausalFactor(const Var &head, const VarSet &body, bool isAnd, Real p, Real q);

    /// @brief 
    /// @return 
    Real prob() const;
    /// @brief 
    /// @return 
    Real prob_default() const;
    /// @brief 
    /// @return `0` if `Singleton`, `1` if `DefiniteAnd`, `2` if `DefiniteOr`
    int type() const;
    const Var& head() const;
    const VarSet& body() const;
    /// @brief 
    /// @return constant reference to variable set (i.e., the variables on 
    /// which the factor depends)
    const VarSet &vars() const;

    /// @brief 
    /// @param i 
    /// @param x 
    /// @return 
    CausalFactor gen_clamped(Var i, size_t x) const;

    friend std::ostream& operator<< (std::ostream&, const CausalFactor&);
};
    
} // namespace lbp
