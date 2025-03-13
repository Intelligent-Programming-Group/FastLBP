/// @file var.cpp
/// 
/// @date 2024-07-21

#include <lbp/var.h>

namespace lbp {

Var::Var(size_t label, size_t nstates): _label(label), _nstates(nstates) {}

size_t Var::label() const {
    return _label;
}

size_t Var::states() const {
    return _nstates;
}

bool Var::operator<(const Var& n) const {
    return _label < n._label;
}

bool Var::operator<=(const Var& n) const {
    return _label <= n._label;
}

bool Var::operator!=(const Var &n) const {
    return _label != n._label;
}

bool Var::operator==(const Var& n) const {
    return _label == n._label;
}

std::ostream& operator<<(std::ostream &os, const Var &n) {
    return os << "x" << n.label();
}

}