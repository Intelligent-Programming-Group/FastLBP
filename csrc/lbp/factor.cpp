/// @file factor.cpp
///
/// @date 2024-07-23

#include <iostream>
#include <lbp/index.h>
#include <lbp/factor.h>

namespace lbp {

template<typename T>
TFactor<T>::TFactor(const Var &v): _vs(v), _p(v.states()) {}

template<typename T>
TFactor<T>::TFactor(const VarSet &vars): _vs(vars), _p(vars.nrStates()) {}

template<typename T>
TFactor<T>::TFactor(const VarSet& vars, T p): _vs(vars), _p() {
    _p = TProb<T>(_vs.nrStates(), p);
}

template<typename T>
TFactor<T>::TFactor(const VarSet& vars, const TProb<T> &p): _vs(vars), _p(p) {
    if (_vs.nrStates() != _p.size()) {
        std::cerr << "states dismatch" << std::endl;
    }
}

template<typename T>
void TFactor<T>::set(size_t i, T val) {
    _p.set(i, val);
}

template<typename T>
const std::vector<T> &TFactor<T>::toVec() const {
    return _p.toVec();
}

template<typename T>
const VarSet& TFactor<T>::vars() const {
    return _vs;
}

template<typename T>
size_t TFactor<T>::nrStates() const {
    return _p.size();
}

template<typename T>
TFactor<T> TFactor<T>::marginal(const VarSet &vars, bool normed) const {
    VarSet res_vars = vars & _vs;

    TFactor<T> res( res_vars, 0.0 );

    IndexFor i_res( res_vars, _vs );
    for( size_t i = 0; i < _p.size(); i++, ++i_res )
        res.set( i_res, res[i_res] + _p[i] );

    if( normed )
        res.normalize();

    return res;
}

} // namespace lbp
