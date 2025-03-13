/// @file causal_factor.cpp
/// @brief
///
/// @date 2024-10-19

#include <iomanip>
#include <ext/causal_factor.h>
#include <utils/log.h>

namespace lbp {

CausalFactor::CausalFactor(const Var &v, Real p = 1): _head(v), _body(), _p(p), _type(CausalType::Singleton) {
    _vs = VarSet();
    _vs.insert(_head);
    head_clamped = false;
    head_mask = Prob(2, 1);
}

CausalFactor::CausalFactor(const Var &head, const VarSet &body, bool isAnd, Real p, Real q): _head(head), _body(body), _p(p), _q(q) {
    if (isAnd) {
        _type = CausalType::DefiniteAnd;
    } else {
        _type = CausalType::DefiniteOr;
    }
    _vs = VarSet(_body);
    _vs.insert(_head);
    head_clamped = false;
    head_mask = Prob(2, 1);
}

Real CausalFactor::prob() const {
    return _p;
}

Real CausalFactor::prob_default() const {
    return _q;
}

int CausalFactor::type() const {
    switch (_type) {
    case CausalType::Singleton:
        return 0;
    case CausalType::DefiniteAnd:
        return 1;
    case CausalType::DefiniteOr:
        return 2;
    default:
        LOG_FATAL("unreachable");
    }
}

const Var &CausalFactor::head() const {
    return _head;
}
const VarSet &CausalFactor::body() const {
    return _body;
}

const VarSet &CausalFactor::vars() const {
    return _vs;
}

CausalFactor CausalFactor::gen_clamped(Var i, size_t x) const {
    CausalFactor newFac(*this);
    switch (_type) {
        case CausalType::Singleton:
            if (i == _head){
                newFac._p = x;
            } else {
                std::cerr << "Clamp unrelated var " << i << " in factor " << *this << std::endl;
            }
            break;
        case CausalType::DefiniteAnd:
        case CausalType::DefiniteOr:
            if (i == _head) {
                newFac.head_clamped = true;
                newFac.head_mask = Prob(2, 0);
                newFac.head_mask.set(x, 1); // x=true =>(0, 1), x=false =>(1, 0)
                if (head_clamped) {
                    std::cerr << "Duplicate clamp head of factor " << *this << "with val " << x << std::endl;
                    newFac.head_mask *= head_mask;
                }
            } else {
                if (_body.contains(i)) {
                    newFac._body.erase(i);
                }
                // Note: if the body is negated, the clamp is also erased
                if ((x == 0 && _type == CausalType::DefiniteAnd) || (x > 0 && _type == CausalType::DefiniteOr)) {
                    newFac._p = _q;
                }
            }
            break;
    }
    return newFac;
}


std::ostream& operator<< (std::ostream& os, const CausalFactor& f) {
    os << f.head() << std::endl << static_cast<char>(f._type);
    if (f._type == CausalFactor::Singleton) {
        os << std::endl << std::setw(os.precision()+4) << f.prob() << std::endl;
    } else {
        os << std::setw(os.precision()+4) << f.prob() << ";"
            << std::setw(os.precision()+4) << f.prob_default() << std::endl;
        os << f.body().size() << std::endl;
        for (const auto & v : f.body()) {
            os << v.label() << " ";
        }
        os << std::endl;
    }
    os << std::endl;
    return os;
}
    
} // namespace lbp 
