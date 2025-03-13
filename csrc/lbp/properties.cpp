/// @file properties.cpp
///
/// @date 2024-09-09

#include <iostream>
#include <lbp/properties.h>

namespace lbp {

PropertySet &PropertySet::set(const PropertyKey &key, const PropertyValue &val) {
    this->operator[](key) = val;
    return *this;
}

const PropertyValue &PropertySet::get(const PropertyKey &key) const {
    auto x = find(key);
    if (x == end()) {
        std::cerr << "Did not find key: " << key << std::endl;
    }
    return x->second;
}

bool PropertySet::hasKey(const PropertyKey &key) const {
    return find(key) != end();
}
    
} // namespace lbp 