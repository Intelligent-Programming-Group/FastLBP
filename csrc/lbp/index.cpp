/// @file index.cpp
///
/// @date 2024-07-23

#include <algorithm>
#include <lbp/index.h>
#include <lbp/varset.h>

namespace lbp {

IndexFor::IndexFor(const VarSet &indexVars, const VarSet &forVars): _state(forVars.size(), 0) {
    long sum = 1;
    _ranges.reserve(forVars.size());
    _sum.reserve(forVars.size());

    VarSet::const_iterator j = forVars.begin();
    for (VarSet::const_iterator i = indexVars.begin(); i != indexVars.end(); i++) {
        for (; j != forVars.end() && *j <= *i; j++) {
            _ranges.push_back(j->states());
            _sum.push_back(*i == *j ? sum : 0);
        }
        sum *= i->states();
    }
    for (; j != forVars.end(); j++) {
        _ranges.push_back(j->states());
        _sum.push_back(0);
    }
    _index = 0;
}

IndexFor::operator size_t() const {
    return _index;
}

IndexFor &IndexFor::operator++() {
    if (_index >= 0) {
        size_t i = 0;
        while(i < _state.size()) {
            _index += _sum[i];
            if (++_state[i] < _ranges[i]) {
                break;
            }
            _index -= _sum[i] * _ranges[i];
            _state[i] = 0;
            i++;
        }
        if (i == _state.size()) {
            _index = -1;
        }
    }
    return *this;
}

void IndexFor::operator++(int) {
    operator++();
}

bool IndexFor::valid() const {
    return _index >= 0;
}

Permute::Permute(const std::vector<Var>& vars, bool reverse): _ranges(), _sigma() {
    size_t N = vars.size();

    // construct ranges
    _ranges.reserve(N);
    for (size_t i = 0; i < N; i++) {
        if (reverse) {
            _ranges.push_back(vars[N - 1 - i].states());
        } else {
            _ranges.push_back(vars[i].states());
        }
    }

    // construct `VarSet` out of `vars`
    VarSet vs(vars.begin(), vars.end(), N /* vars.size() */);

    // construct `_sigma`
    _sigma.reserve(N);
    for (VarSet::const_iterator vs_i = vs.begin(); vs_i != vs.end(); vs_i++) {
        size_t ind = std::find(vars.begin(), vars.end(), *vs_i) - vars.begin();
        if (reverse) {
            _sigma.push_back(N - 1 - ind);
        } else {
            _sigma.push_back(ind);
        }
    }
}

size_t Permute::convertLinearIndex(size_t li) const {
    size_t N = _ranges.size();

    // calculate vector index corresponding to linear index
    std::vector<size_t> vi;
    vi.reserve(N);
    for (size_t k = 0; k < N; k++) {
        vi.push_back(li % _ranges[k]);
        li /= _ranges[k];
    }

    // convert permuted vector index to corresponding linear index
    size_t prod = 1;
    size_t sigma_li = 0;
    for (size_t k = 0; k < N; k++) {
        sigma_li += vi[_sigma[k]] * prod;
        prod *= _ranges[_sigma[k]];
    }

    return sigma_li;
}
    
} // namespace lbp 
