/// @file varset.cpp
/// 
/// @date 2024-07-23

#include <lbp/varset.h>

namespace lbp {

VarSet::VarSet(const Var &v): SmallSet<Var>(v) {}

size_t VarSet::nrStates() const {
    size_t states = 1;
    for (VarSet::const_iterator n = begin(); n != end(); n++) {
        states *= n->states();
    }
    return states;
}

std::ostream &operator<<(std::ostream &os, const VarSet &vs) {
    os << "{";
    for (VarSet::const_iterator v = vs.begin(); v != vs.end(); v++) {
        os << (v != vs.begin() ? ", " : "") << *v;
    }
    os << "}";
    return os;
}

}
