/// @file prob.cpp
///
/// @date 2024-07-23

#include <lbp/prob.h>

namespace lbp {

template<typename T>
TProb<T>::TProb(size_t n): _p(n, T(1) / n) {}

template<typename T>
TProb<T>::TProb(size_t n, T p): _p(n, p) {}
    
template<typename T>
void TProb<T>::set(size_t i, T val) {
    _p.at(i) = val;
}

template<typename T>
const std::vector<T> &TProb<T>::toVec() const {
    return _p;
}

template<typename T>
size_t TProb<T>::size() const {
    return _p.size();
}

} // namespace lbp

