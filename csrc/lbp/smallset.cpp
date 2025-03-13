/// @file smallset.cpp
///
/// @date 2024-07-23

#include <lbp/smallset.h>

namespace lbp {

template<typename T>
SmallSet<T>::SmallSet(const T &t) {
    _elements.push_back(t);
}

template<typename T>
SmallSet<T> &SmallSet<T>::insert(const T &t) {
    SmallSet<T>::iterator it = std::lower_bound(_elements.begin(), _elements.end(), t);
    if ((it == _elements.end()) || (*it != t)) {
        _elements.insert(it, t);
    }
    return *this;
}

template<typename T>
bool SmallSet<T>::operator>>(const SmallSet &x) const {
    return std::includes( _elements.begin(), _elements.end(), x._elements.begin(), x._elements.end() );
}

template<typename T>
typename SmallSet<T>::const_iterator SmallSet<T>::begin() const {
    return _elements.begin();
}

template<typename T>
typename SmallSet<T>::const_iterator SmallSet<T>::end() const {
    return _elements.end();
}

template<typename T>
typename std::vector<T>::size_type SmallSet<T>::size() const {
    return _elements.size();
}
    
} // namespace lbp

