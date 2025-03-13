/// @file causal_bp.cu
/// @brief We use .cu suffix to compile this file with nvcc so as to use thrust.
///
/// @date 2025-01-14

#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <queue>
#include <ext/causal_bp.h>
#include <kernel/causal_bp_parall.h>
#include <kernel/causal_bp_seq.h>
#include <utils/cuda_utils.h>
#include <utils/utils.h>

namespace lbp {

struct dist_functor {
    __device__ Real operator()(const Real x, const Real y) const {
        return std::fabs(x - y);
    }
};

struct norm_functor {
    __device__ Real operator()(const Real x, const Real y) const {
        return y / (x + y);
    }
};

struct count_functor {
    Real tolerence;

    count_functor(Real tol): tolerence(tol) {}

    __device__ size_t operator()(const Real x) const {
        return (size_t)(x > tolerence);
    }
};

const Prob &CausalBP::newMessage(size_t i, size_t _I) const {
    return _edges[i][_I].message;
}

void CausalBP::setProperties(const PropertySet &opts) {
    if (opts.hasKey("tol")) {
        props.tol = opts.getStringAs<Real>("tol");
    } else {
        props.tol = 1e-9;
    }
    if (opts.hasKey("logdomain")) {
        props.logdomain = opts.getStringAs<bool>("logdomain");
    } else {
        props.logdomain = true;
    }
    if (opts.hasKey("maxiter")) {
        props.maxiter = opts.getStringAs<size_t>("maxiter");
    } else {
        props.maxiter = 10000;
    }
    if (opts.hasKey("maxtime")) {
        props.maxtime = opts.getStringAs<Real>("maxtime");
    } else {
        props.maxtime = 10;
    }
    if (opts.hasKey("updates")) {
        std::string updates = opts.getStringAs<std::string>("updates");
        if (updates == "PARALL") {
            props.updates = Properties::UpdateType::PARALL;
        } else if (updates == "SEQFIX") {
            props.updates = Properties::UpdateType::SEQFIX;
        } else if (updates == "SEQRND") {
            props.updates = Properties::UpdateType::SEQRND;
        } else {
            props.updates = Properties::UpdateType::SEQFIX;
        }
    } else {
        props.updates = Properties::UpdateType::SEQFIX;
    }
    if (opts.hasKey("verbose")) {
        props.verbose = opts.getStringAs<size_t>("verbose");
    } else {
        props.verbose = 0;
    }
}

void CausalBP::construct() {
    // initialize device vectors
    edgePropKernel.row_ptr_fv.resize(nrVars());
    edgePropKernel.prod_fv_0.resize(nrVars());
    edgePropKernel.prod_fv_1.resize(nrVars());
    edgePropKernel.num_zeros_fv_0.resize(nrVars());
    edgePropKernel.num_zeros_fv_1.resize(nrVars());
    edgePropKernel.beliefs_0.resize(nrVars());
    edgePropKernel.beliefs_1.resize(nrVars());

    size_t message_len = 0; // messages v->f and f->v have the same length
    for (size_t I = 0; I < nrFactors(); I++) {
        message_len += factor(I).vars().size();
    }
    edgePropKernel.message_fv_0.resize(message_len);
    edgePropKernel.message_fv_1.resize(message_len);
    edgePropKernel.message_vf_0.resize(message_len);
    edgePropKernel.message_vf_1.resize(message_len);
    edgePropKernel.col_ind_fv.resize(message_len);

    // prob/prob_default
    thrust::host_vector<Real> h_prob(nrFactors());
    thrust::host_vector<Real> h_prob_default(nrFactors());
    for (size_t I = 0; I < nrFactors(); I++) {
        h_prob[I] = factor(I).prob();
        h_prob_default[I] = factor(I).prob_default();
    }

    // indices
    thrust::host_vector<size_t> h_row(message_len);
    thrust::host_vector<size_t> h_col(message_len);
    thrust::host_vector<size_t> h_row_ptr_vf(nrFactors() + 1);
    thrust::host_vector<size_t> h_col_ind_vf(message_len);
    thrust::host_vector<size_t> h_head(nrFactors());
    size_t idx = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        size_t head_label = factor(I).head().label();
        for (auto &i: nbF(I)) {
            h_row[idx] = I;
            h_col[idx] = i;
            if (i == head_label) {
                h_head[I] = idx;
            }
            idx++;
        }
    }
    coo2csr(h_row, h_col, h_row_ptr_vf, h_col_ind_vf);

    idx = 0;
    thrust::host_vector<size_t> h_ind_fv2vf(message_len);
    thrust::host_vector<size_t> h_ind_vf2fv(message_len);
    for (size_t i = 0; i < nrVars(); i++) {
        for (auto &I: nbV(i)) {
            size_t I_ = I.node;
            for (size_t j = h_row_ptr_vf[I_]; j < h_row_ptr_vf[I_ + 1]; j++) {
                if (h_col_ind_vf[j] == i) {
                    h_ind_fv2vf[idx] = j;
                    h_ind_vf2fv[j] = idx;
                    break;
                }
            }
            idx++;
        }
    }

    idx = 0;
    thrust::host_vector<size_t> h_row_ptr_fv(nrVars() + 1);
    thrust::host_vector<size_t> h_col_ind_fv(message_len);
    for (size_t i = 0; i < nrVars(); i++) {
        for (auto &I: nbV(i)) {
            size_t I_ = I.node;
            h_row[idx] = i;
            h_col[idx] = I_;
            idx++;
        }
    }
    coo2csr(h_row, h_col, h_row_ptr_fv, h_col_ind_fv);
    edgePropKernel.row_ptr_fv = h_row_ptr_fv;
    edgePropKernel.col_ind_fv = h_col_ind_fv;

    _oldBeliefsV.resize(nrVars());
    _oldBeliefsF.resize(nrFactors());
    _beliefsV.resize(nrVars());

    thrust::host_vector<Real> h_mask0(nrFactors());
    thrust::host_vector<Real> h_mask1(nrFactors());
    for (size_t I = 0; I < nrFactors(); I++) {
        h_mask0[I] = factor(I).head_mask[0];
        h_mask1[I] = factor(I).head_mask[1];
    }

    if (props.updates != Properties::UpdateType::PARALL) {
        _updateSeq.clear();
        _updateSeq.reserve(nrEdges());
        for (size_t I = 0; I < nrFactors(); I++) {
            for (auto &i: nbF(I)) {
                _updateSeq.push_back(Edge(i, i.dual));
            }
        }
        
        if (props.updates == Properties::UpdateType::SEQFIX) {
            putUpdateSeqToKernel(
                h_row_ptr_fv, h_row_ptr_vf, h_head, 
                h_prob_default, h_prob, h_mask0, h_mask1
            );
        } else if (props.updates == Properties::UpdateType::SEQRND) {
            hostProp.h_row_ptr_fv = h_row_ptr_fv;
            hostProp.h_row_ptr_vf = h_row_ptr_vf;
            hostProp.h_head = h_head;
            hostProp.h_prob_default = h_prob_default;
            hostProp.h_prob = h_prob;
            hostProp.h_mask0 = h_mask0;
            hostProp.h_mask1 = h_mask1;
        }
    } else {
        putParallUpdateSeqToKernel(
            h_row_ptr_fv, h_row_ptr_vf, h_head, 
            h_prob_default, h_prob, h_mask0, h_mask1
        );
    }
}

void CausalBP::calcNewMessageAll() {
    // std::cerr << "Iteration " << i << std::endl;
    // v -> f
    kernel::calcProdAll(
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.row_ptr_fv.data()),
        nrVars()
    );
    kernel::calcMessageVFAll(
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqVF[0].data()),
        _updateSeqVF[0].size()
    );

    // f -> v
    kernel::calcMessageFVIAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqI[0].data()),
        _updateSeqI[0].size(), 0
    );
    kernel::calcMessageFVAndTDAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndTD[0].data()),
        _updateSeqAndTD[0].size(), 1
    );
    kernel::calcMessageFVOrTDAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrTD[0].data()),
        _updateSeqOrTD[0].size(), 2
    );
    kernel::calcMessageFVAndTDClampedAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndClampedTD[0].data()),
        _updateSeqAndClampedTD[0].size(), 3
    );
    kernel::calcMessageFVOrTDClampedAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrClampedTD[0].data()),
        _updateSeqOrClampedTD[0].size(), 4
    );
    kernel::calcMessageFVAndBUAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndBU[0].data()),
        _updateSeqAndBU[0].size(), 5
    );
    kernel::calcMessageFVOrBUAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrBU[0].data()),
        _updateSeqOrBU[0].size(), 6
    );
    kernel::calcMessageFVAndBUClampedAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndClampedBU[0].data()),
        _updateSeqAndClampedBU[0].size(), 7
    );
    kernel::calcMessageFVOrBUClampedAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrClampedBU[0].data()),
        _updateSeqOrClampedBU[0].size(), 8
    );
    kernel::streamSynchronize();
}

void CausalBP::calcNewMessage(size_t i) {
    // std::cerr << "Iteration " << i << std::endl;
    // v -> f
    kernel::calcMessageVF(
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqVF[i].data()),
        _updateSeqVF[i].size()
    );

    // f -> v
    kernel::calcMessageFVI(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqI[i].data()),
        _updateSeqI[i].size(), 0
    );
    kernel::calcMessageFVAndTD(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndTD[i].data()),
        _updateSeqAndTD[i].size(), 1
    );
    kernel::calcMessageFVOrTD(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrTD[i].data()),
        _updateSeqOrTD[i].size(), 2
    );
    kernel::calcMessageFVAndTDClamped(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndClampedTD[i].data()),
        _updateSeqAndClampedTD[i].size(), 3
    );
    kernel::calcMessageFVOrTDClamped(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrClampedTD[i].data()),
        _updateSeqOrClampedTD[i].size(), 4
    );
    kernel::calcMessageFVAndBU(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndBU[i].data()),
        _updateSeqAndBU[i].size(), 5
    );
    kernel::calcMessageFVOrBU(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrBU[i].data()),
        _updateSeqOrBU[i].size(), 6
    );
    kernel::calcMessageFVAndBUClamped(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndClampedBU[i].data()),
        _updateSeqAndClampedBU[i].size(), 7
    );
    kernel::calcMessageFVOrBUClamped(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqOrClampedBU[i].data()),
        _updateSeqOrClampedBU[i].size(), 8
    );
    kernel::streamSynchronize();
}

void CausalBP::updateMessage(size_t i) {

}

void CausalBP::calcBeliefsV(thrust::device_vector<Real> &newBeliefsV) {
    kernel::calcBeliefsV(
        thrust::raw_pointer_cast(edgePropKernel.beliefs_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.row_ptr_fv.data()),
        thrust::raw_pointer_cast(edgePropKernel.col_ind_fv.data()),
        edgePropKernel.beliefs_0.size()
    );
    kernel::calcBeliefsV(
        thrust::raw_pointer_cast(edgePropKernel.beliefs_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.row_ptr_fv.data()),
        thrust::raw_pointer_cast(edgePropKernel.col_ind_fv.data()),
        edgePropKernel.beliefs_1.size()
    );
    // kernel::getBeliefsV(
    //     thrust::raw_pointer_cast(edgePropKernel.beliefs_0.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
    //     nrVars()
    // );
    // kernel::getBeliefsV(
    //     thrust::raw_pointer_cast(edgePropKernel.beliefs_1.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
    //     nrVars()
    // );
    thrust::transform(
        edgePropKernel.beliefs_0.begin(), edgePropKernel.beliefs_0.end(), 
        edgePropKernel.beliefs_1.begin(), newBeliefsV.begin(), 
        norm_functor()
    );
}

void CausalBP::calcBeliefsF(thrust::device_vector<Real> &newBeliefsF) {

}

void CausalBP::calcBeliefV(size_t i, Prob &p) const {
    std::vector<Real> v = {1 - _beliefsV[i], _beliefsV[i]};
    p = Prob(v);
}

Factor CausalBP::belief(const VarSet &vs) const {
    if (vs.size() == 0) {
        return Factor();
    } else if (vs.size() == 1) {
        return beliefV(findVar(*(vs.begin())));
    } else {
        // size_t I;
        // for (I = 0; I < nrFactors(); I++) {
        //     if (factor(I).vars() >> vs) {
        //         break;
        //     }
        // }
        // if (I == nrFactors()) {
        //     std::cerr << "Belief not available!" << std::endl;
        // }
        // return beliefF(I).marginal(vs);
        LOG_FATAL("unreachable");
    }
}

Factor CausalBP::beliefV(size_t i) const {
    Prob p;
    calcBeliefV(i, p);
    return Factor(var(i), p);
}

void CausalBP::transferMessageToHost() {
    _beliefsV = _oldBeliefsV;
}

void CausalBP::coo2csr(
    thrust::host_vector<size_t> &rows, thrust::host_vector<size_t> &cols, 
    thrust::host_vector<size_t> &h_row_ptr, 
    thrust::host_vector<size_t> &h_col_ind
) {
    size_t num_rows = h_row_ptr.size() - 1;

    size_t nnz = rows.size();
    std::vector<size_t> row_cnt(num_rows, 0);
    for (size_t i = 0; i < nnz; i++) {
        row_cnt[rows[i]]++;
    }

    h_row_ptr[0] = 0;
    for (size_t i = 0; i < num_rows; i++) {
        h_row_ptr[i + 1] = h_row_ptr[i] + row_cnt[i];
    }

    std::vector<size_t> cur_index(num_rows, 0);
    for (size_t i = 0; i < nnz; i++) {
        size_t r = rows[i];
        size_t dest_index = h_row_ptr[r] + cur_index[r];
        cur_index[r]++;
        h_col_ind[dest_index] = cols[i];
    }
}

void CausalBP::putUpdateSeqToKernel(
    thrust::host_vector<size_t> &h_row_ptr_fv, 
    thrust::host_vector<size_t> &h_row_ptr_vf, 
    thrust::host_vector<size_t> &h_head, 
    thrust::host_vector<Real> &h_prob_default, 
    thrust::host_vector<Real> &h_prob, 
    thrust::host_vector<Real> &h_mask0, 
    thrust::host_vector<Real> &h_mask1
) {
    _updateSeqVF.clear();
    _updateSeqI.clear();
    _updateSeqAndTD.clear();
    _updateSeqOrTD.clear();
    _updateSeqAndClampedTD.clear();
    _updateSeqOrClampedTD.clear();
    _updateSeqAndBU.clear();
    _updateSeqOrBU.clear();
    _updateSeqAndClampedBU.clear();
    _updateSeqOrClampedBU.clear();

    thrust::host_vector<Index3> update_seq_vf;
    thrust::host_vector<Index2Real> update_seq_i;
    thrust::host_vector<Index5Real2> update_seq_and_td;
    thrust::host_vector<Index5Real2> update_seq_or_td;
    thrust::host_vector<Index5Real4> update_seq_and_clamped_td;
    thrust::host_vector<Index5Real4> update_seq_or_clamped_td;
    thrust::host_vector<Index6Real2> update_seq_and_bu;
    thrust::host_vector<Index6Real2> update_seq_or_bu;
    thrust::host_vector<Index6Real4> update_seq_and_clamped_bu;
    thrust::host_vector<Index6Real4> update_seq_or_clamped_bu;
    std::set<Edge> fv_set;
    std::set<Edge> vf_set;
    std::set<size_t> v_set;
    for (const auto &e: _updateSeq) {
        size_t i = e.first;
        auto I = nbV(e.first, e.second);
        // check confliction
        bool conflict = false;
        for (auto &j: nbF(I)) {
            if (j != i) {
                Edge vf_e(I, j.iter);
                if (vf_set.find(vf_e) == vf_set.end()) {
                    for (auto &J: nbV(j)) {
                        if (J != I) {
                            Edge fv_e(j, J.iter);
                            if (fv_set.find(fv_e) != fv_set.end()) {
                                conflict = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (conflict) {
                break;
            }
        }
        if (!conflict) {
            if (v_set.find(i) != v_set.end()) {
                conflict = true;
            }
        }

        if (conflict) {
            _updateSeqVF.push_back(update_seq_vf);
            _updateSeqI.push_back(update_seq_i);
            _updateSeqAndTD.push_back(update_seq_and_td);
            _updateSeqOrTD.push_back(update_seq_or_td);
            _updateSeqAndClampedTD.push_back(update_seq_and_clamped_td);
            _updateSeqOrClampedTD.push_back(update_seq_or_clamped_td);
            _updateSeqAndBU.push_back(update_seq_and_bu);
            _updateSeqOrBU.push_back(update_seq_or_bu);
            _updateSeqAndClampedBU.push_back(update_seq_and_clamped_bu);
            _updateSeqOrClampedBU.push_back(update_seq_or_clamped_bu);

            // std::cerr << update_seq_vf.size() << " " << update_seq_i.size() << " " << update_seq_and_td.size()
            //  << " " << update_seq_or_td.size() << " " << update_seq_and_clamped_td.size()  << " " << update_seq_or_clamped_td.size()
            //  << " " << update_seq_and_bu.size() << " " << update_seq_or_bu.size() << " " << update_seq_and_clamped_bu.size()
            //  << " " << update_seq_or_clamped_bu.size() << std::endl;

            v_set.clear();
            fv_set.clear();
            vf_set.clear();
            update_seq_vf.clear();
            update_seq_i.clear();
            update_seq_and_td.clear();
            update_seq_or_td.clear();
            update_seq_and_clamped_td.clear();
            update_seq_or_clamped_td.clear();
            update_seq_and_bu.clear();
            update_seq_or_bu.clear();
            update_seq_and_clamped_bu.clear();
            update_seq_or_clamped_bu.clear();
        }

        v_set.insert(i);
        fv_set.insert(e);
        size_t ind = h_row_ptr_fv[i] + e.second;
        size_t src_ind = h_row_ptr_vf[I] + I.dual;
        size_t start_ind = h_row_ptr_vf[I], end_ind = h_row_ptr_vf[I + 1];
        size_t head = h_head[I];
        Real prob_default = h_prob_default[I], prob = h_prob[I];
        if (factor(I).type() == 0) {
            update_seq_i.push_back({ind, i, h_prob[I]});
        } else if (factor(I).type() == 1) {
            if (factor(I).head_clamped) {
                if (factor(I).head().label() == i) {
                    update_seq_and_clamped_td.push_back({
                        ind, i, src_ind, start_ind, end_ind, 
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                } else {
                    update_seq_and_clamped_bu.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                }
            } else {
                if (factor(I).head().label() == i) {
                    update_seq_and_td.push_back({
                        ind, i, src_ind, start_ind, end_ind, prob_default, prob
                    });
                } else {
                    update_seq_and_bu.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 
                        prob_default, prob
                    });
                }
            }
        } else {
            if (factor(I).head_clamped) {
                if (factor(I).head().label() == i) {
                    update_seq_or_clamped_td.push_back({
                        ind, i, src_ind, start_ind, end_ind, 
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                } else {
                    update_seq_or_clamped_bu.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                }
            } else {
                if (factor(I).head().label() == i) {
                    update_seq_or_td.push_back({
                        ind, i, src_ind, start_ind, end_ind, prob_default, prob
                    });
                } else {
                    update_seq_or_bu.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 
                        prob_default, prob
                    });
                }
            }
        }
        for (auto &j: nbF(I)) {
            if (j != i) {
                Edge vf_e(I, j.iter);
                if (vf_set.find(vf_e) == vf_set.end()) {
                    vf_set.insert(vf_e);
                    update_seq_vf.push_back({
                        h_row_ptr_vf[I] + j.iter, j, h_row_ptr_fv[j] + j.dual
                    });
                }
            }
        }
    }
    _updateSeqVF.push_back(update_seq_vf);
    _updateSeqI.push_back(update_seq_i);
    _updateSeqAndTD.push_back(update_seq_and_td);
    _updateSeqOrTD.push_back(update_seq_or_td);
    _updateSeqAndClampedTD.push_back(update_seq_and_clamped_td);
    _updateSeqOrClampedTD.push_back(update_seq_or_clamped_td);
    _updateSeqAndBU.push_back(update_seq_and_bu);
    _updateSeqOrBU.push_back(update_seq_or_bu);
    _updateSeqAndClampedBU.push_back(update_seq_and_clamped_bu);
    _updateSeqOrClampedBU.push_back(update_seq_or_clamped_bu);
    // std::cerr << "len: " <<  _updateSeqVF.size() << std::endl;
}

void CausalBP::putParallUpdateSeqToKernel(
    thrust::host_vector<size_t> &h_row_ptr_fv, 
    thrust::host_vector<size_t> &h_row_ptr_vf, 
    thrust::host_vector<size_t> &h_head, 
    thrust::host_vector<Real> &h_prob_default, 
    thrust::host_vector<Real> &h_prob, 
    thrust::host_vector<Real> &h_mask0, 
    thrust::host_vector<Real> &h_mask1
) {
    thrust::host_vector<Index3> update_seq_vf;
    thrust::host_vector<Index2Real> update_seq_i;
    thrust::host_vector<Index5Real2> update_seq_and_td;
    thrust::host_vector<Index5Real2> update_seq_or_td;
    thrust::host_vector<Index5Real4> update_seq_and_clamped_td;
    thrust::host_vector<Index5Real4> update_seq_or_clamped_td;
    thrust::host_vector<Index6Real2> update_seq_and_bu;
    thrust::host_vector<Index6Real2> update_seq_or_bu;
    thrust::host_vector<Index6Real4> update_seq_and_clamped_bu;
    thrust::host_vector<Index6Real4> update_seq_or_clamped_bu;

    for (size_t i = 0; i < nrVars(); i++) {
        for (auto &I: nbV(i)) {
            size_t ind = h_row_ptr_fv[i] + I.iter;
            size_t src_ind = h_row_ptr_vf[I] + I.dual;
            size_t start_ind = h_row_ptr_vf[I], end_ind = h_row_ptr_vf[I + 1];
            size_t head = h_head[I];
            Real prob_default = h_prob_default[I], prob = h_prob[I];
            if (factor(I).type() == 0) {
                update_seq_i.push_back({ind, i, h_prob[I]});
            } else if (factor(I).type() == 1) {
                if (factor(I).head_clamped) {
                    if (factor(I).head().label() == i) {
                        update_seq_and_clamped_td.push_back({
                            ind, i, src_ind, start_ind, end_ind, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    } else {
                        update_seq_and_clamped_bu.push_back({
                            ind, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    }
                } else {
                    if (factor(I).head().label() == i) {
                        update_seq_and_td.push_back({
                            ind, i, src_ind, start_ind, end_ind, prob_default, prob
                        });
                    } else {
                        update_seq_and_bu.push_back({
                            ind, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob
                        });
                    }
                }
            } else {
                if (factor(I).head_clamped) {
                    if (factor(I).head().label() == i) {
                        update_seq_or_clamped_td.push_back({
                            ind, i, src_ind, start_ind, end_ind, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    } else {
                        update_seq_or_clamped_bu.push_back({
                            ind, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    }
                } else {
                    if (factor(I).head().label() == i) {
                        update_seq_or_td.push_back({
                            ind, i, src_ind, start_ind, end_ind, prob_default, prob
                        });
                    } else {
                        update_seq_or_bu.push_back({
                            ind, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob
                        });
                    }
                }
            }
        }
    }
    for (size_t I = 0; I < nrFactors(); I++) {
        for (auto &i: nbF(I)) {
            update_seq_vf.push_back({
                h_row_ptr_vf[I] + i.iter, i, h_row_ptr_fv[i] + i.dual
            });
        }
    }
    _updateSeqVF.push_back(update_seq_vf);
    _updateSeqI.push_back(update_seq_i);
    _updateSeqAndTD.push_back(update_seq_and_td);
    _updateSeqOrTD.push_back(update_seq_or_td);
    _updateSeqAndClampedTD.push_back(update_seq_and_clamped_td);
    _updateSeqOrClampedTD.push_back(update_seq_or_clamped_td);
    _updateSeqAndBU.push_back(update_seq_and_bu);
    _updateSeqOrBU.push_back(update_seq_or_bu);
    _updateSeqAndClampedBU.push_back(update_seq_and_clamped_bu);
    _updateSeqOrClampedBU.push_back(update_seq_or_clamped_bu);
}

CausalBP::CausalBP(const CausalFactorGraph &fg, const PropertySet &opts): DAIAlg<CausalFactorGraph>(fg), _iters(0), edgePropKernel() {
    setProperties(opts);
    construct();
}

void CausalBP::init() {
    Real c = props.logdomain ? 0.0 : 1.0;
    thrust::fill(
        edgePropKernel.message_fv_0.begin(), 
        edgePropKernel.message_fv_0.end(), c
    );
    thrust::fill(
        edgePropKernel.message_fv_1.begin(), 
        edgePropKernel.message_fv_1.end(), c
    );
    thrust::fill(
        edgePropKernel.message_vf_0.begin(), 
        edgePropKernel.message_vf_0.end(), c
    );
    thrust::fill(
        edgePropKernel.message_vf_1.begin(), 
        edgePropKernel.message_vf_1.end(), c
    );
    thrust::fill(
        edgePropKernel.prod_fv_0.begin(), edgePropKernel.prod_fv_0.end(), c
    );
    thrust::fill(
        edgePropKernel.prod_fv_1.begin(), edgePropKernel.prod_fv_1.end(), c
    );
    thrust::fill(
        edgePropKernel.num_zeros_fv_0.begin(), 
        edgePropKernel.num_zeros_fv_0.end(), 0
    );
    thrust::fill(
        edgePropKernel.num_zeros_fv_1.begin(), 
        edgePropKernel.num_zeros_fv_1.end(), 0
    );
}

void CausalBP::run() {
    if (props.verbose >= 1) {
        std::cerr << "Starting ..." << std::endl;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);

    Real tic = toc();
    Real maxDiff = INFINITY;

    for (; _iters < props.maxiter && maxDiff > props.tol && (toc() - tic) < props.maxtime; _iters++) {
        if (props.updates == Properties::UpdateType::PARALL) {
            calcNewMessageAll();
        } else {
            if (props.updates == Properties::UpdateType::SEQRND) {
                std::shuffle(_updateSeq.begin(), _updateSeq.end(), engine);
                putUpdateSeqToKernel(
                    hostProp.h_row_ptr_fv, hostProp.h_row_ptr_vf, hostProp.h_head, 
                    hostProp.h_prob_default, hostProp.h_prob, hostProp.h_mask0, 
                    hostProp.h_mask1
                );
            }
            for (size_t i = 0; i < _updateSeqVF.size(); i++) {  
                calcNewMessage(i);
                updateMessage(i);
            }
        }
        maxDiff = -INFINITY;
        thrust::device_vector<Real> newBeliefsV(nrVars());
        thrust::device_vector<Real> beliefsVDiff(nrVars());
        calcBeliefsV(newBeliefsV);
        thrust::transform(
            _oldBeliefsV.begin(), _oldBeliefsV.end(), newBeliefsV.begin(), 
            beliefsVDiff.begin(), dist_functor()
        );
        Real diff = thrust::reduce(
            beliefsVDiff.begin(), beliefsVDiff.end(), 0.0, 
            thrust::maximum<Real>()
        );
        maxDiff = std::max(diff, maxDiff);

        _oldBeliefsV = newBeliefsV;
    }

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    if (props.verbose >= 1) {
        if (maxDiff > props.tol) {
            if (props.verbose == 1) {
                std::cerr << std::endl;
            }
            std::cerr << "BP::run:  WARNING: not converged after " << _iters;
            std::cerr << " passes (" << toc() - tic;
            std::cerr << " seconds)...final maxdiff:" << maxDiff << std::endl;
        } else {
            if (props.verbose >= 3) {
                std::cerr << "BP::run:  ";
            }
            std::cerr << "converged in " << _iters << " passes (";
            std::cerr << toc() - tic << " seconds)." << std::endl;
        }
    }
    transferMessageToHost();
}

Real CausalBP::run(Real tolerance, size_t minIters, size_t maxIters, size_t histLength) {
    assert(0 < tolerance);
    assert(0 < histLength && histLength < minIters && minIters < maxIters);
    
    std::cerr << "Starting CausalBP"
                       << "...  tolerance: " << tolerance
                       << ". minIters: " << minIters
                       << ". maxIters: " << maxIters
                       << ". histLength: " << histLength 
                       << "." << std::endl;

    Real tic = toc();

    size_t numIters = 0;
    Real maxDiff = INFINITY;
    Real yetToConvergeFraction = 1.0;
    Real nodeFracTolerance = 0.0;
    std::vector<std::queue<Real>> beliefHist(nrVars());

    enum class RunReturnReason { ALL_CONVERGED, BIG_FRAC_CONVERGED, DIVERGED };
    RunReturnReason returnReason = RunReturnReason::DIVERGED;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);

    for (; true; numIters++, _iters++) {
        if (numIters >= minIters) {
            nodeFracTolerance = Real(numIters - minIters) / (maxIters - minIters);
        }

        if (maxDiff <= tolerance) {
            returnReason = RunReturnReason::ALL_CONVERGED;
            break;
        } else if (numIters > minIters && yetToConvergeFraction < nodeFracTolerance) {
            returnReason = RunReturnReason::BIG_FRAC_CONVERGED;
            break;
        } else if (numIters > maxIters) {
            returnReason = RunReturnReason::DIVERGED;
            break;
        } else if ((toc() - tic) > props.maxtime) {
            returnReason = RunReturnReason::DIVERGED;
            break;
        }

        if (props.updates == Properties::UpdateType::PARALL) {
            calcNewMessageAll();
        } else {
            if (props.updates == Properties::UpdateType::SEQRND) {
                std::shuffle(_updateSeq.begin(), _updateSeq.end(), engine);
                putUpdateSeqToKernel(
                    hostProp.h_row_ptr_fv, hostProp.h_row_ptr_vf, hostProp.h_head, 
                    hostProp.h_prob_default, hostProp.h_prob, hostProp.h_mask0, 
                    hostProp.h_mask1
                );
            }
            for (size_t i = 0; i < _updateSeqVF.size(); i++) {  
                calcNewMessage(i);
                updateMessage(i);
            }
        }

        maxDiff = -INFINITY;
        thrust::device_vector<Real> newBeliefsV(nrVars());
        thrust::device_vector<Real> beliefsVDiff(nrVars());
        calcBeliefsV(newBeliefsV);
        thrust::transform(
            _oldBeliefsV.begin(), _oldBeliefsV.end(), newBeliefsV.begin(), 
            beliefsVDiff.begin(), dist_functor()
        );

        _oldBeliefsV = newBeliefsV;
        Real diff = thrust::reduce(
            beliefsVDiff.begin(), beliefsVDiff.end(), 0.0, 
            thrust::maximum<Real>()
        );
        maxDiff = std::max(diff, maxDiff);
        thrust::device_vector<size_t> nonConverged(nrVars());
        thrust::transform(
            beliefsVDiff.begin(), beliefsVDiff.end(), nonConverged.begin(),
            count_functor(tolerance)
        );
        
        size_t nonConvergedElems = thrust::reduce(
            nonConverged.begin(), nonConverged.end(), 0, thrust::plus<size_t>()
        );

        // thrust::host_vector<Real> bel(_oldBeliefsV);
        // thrust::host_vector<Real> dist(beliefsVDiff);
        // // // calculate new beliefs and compare with old ones
        // // std::map<int, size_t> diffHistogram;
        // // const int minBucketIndex = -1;
        // // int maxBucketIndex = 0;

        // for( size_t i = 0; i < nrVars(); ++i ) {
        //     Factor b( beliefV(i) );
        //     Real iDist = dist[i];
        //     if (iDist == 1) {
        //         std::cerr << "you diff is one! fuck you asshole!    " << i  << ": " << bel[i] << std::endl;
        //     }

        //     // if (iDist == 0) {
        //     //     diffHistogram[minBucketIndex]++;
        //     // } else {
        //     //     int bucketIndex = std::max(int(ceil(log2(iDist) - log2(props.tol))), minBucketIndex);
        //     //     diffHistogram[bucketIndex]++;
        //     //     maxBucketIndex = std::max(maxBucketIndex, bucketIndex);
        //     // }

        //     // auto newBelief = newBeliefsV[i];
        //     // auto newBeliefType = fpclassify(newBelief);
        //     // if (newBeliefType == FP_NORMAL || newBeliefType == FP_SUBNORMAL || newBeliefType == FP_ZERO) {
        //     //     beliefHist[i].push(newBelief);
        //     // }
        //     // if (beliefHist[i].size() > histLength) {
        //     //     beliefHist[i].pop();
        //     // }
        // }

        yetToConvergeFraction = Real(nonConvergedElems) / nrVars();
        // break;

        // std::cerr << "CausalBP::run():  maxdiff: " << maxDiff
        //                              << ". numIters: " << numIters
        //                              << ". Time elapsed: " << toc() - tic << " seconds. "
        //                              << "yetToConvergeFraction: " << yetToConvergeFraction << "." << std::endl;
    }

    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    transferMessageToHost();
    switch (returnReason) {
    case RunReturnReason::ALL_CONVERGED:
        _lowPassBeliefs = std::vector<Real>(nrVars());
        for (size_t i = 0; i < nrVars(); i++) {
            _lowPassBeliefs[i] = _beliefsV[i];
        }
        break;
    case RunReturnReason::BIG_FRAC_CONVERGED:
    case RunReturnReason::DIVERGED:
        // _lowPassBeliefs = std::vector<Real>(nrVars());
        // for (size_t i = 0; i < nrVars(); i++) {
        //     assert(beliefHist[i].size() <= histLength);
        //     size_t denom = beliefHist[i].size();
        //     while (!beliefHist[i].empty()) {
        //         _lowPassBeliefs[i] += beliefHist[i].front();
        //         beliefHist[i].pop();
        //     }
        //     if (denom > 0) { _lowPassBeliefs[i] /= denom; }
        // }
        _lowPassBeliefs = std::vector<Real>(nrVars());
        for (size_t i = 0; i < nrVars(); i++) {
            _lowPassBeliefs[i] = _beliefsV[i];
        }
        break;
    }

    switch (returnReason) {
    case RunReturnReason::ALL_CONVERGED:
        std::cerr << "CausalBP::run:  converged in " << numIters << " passes and "
                           << toc() - tic << " seconds. Final maxdiff: " << maxDiff << std::endl;
        break;
    case RunReturnReason::BIG_FRAC_CONVERGED:
        std::cerr << "CausalBP::run:  Sufficiently big fraction " << yetToConvergeFraction
                                     << " of variables appeared to converge in " << numIters << " passes and "
                                     << toc() - tic << " seconds. Final maxDiff: " << maxDiff << std::endl;
        break;
    case RunReturnReason::DIVERGED:
        std::cerr << "CausalBP::run:  WARNING: not converged after " << numIters << " passes and "
                           << toc() - tic << " seconds. Final maxdiff: " << maxDiff << std::endl;
        break;
    }

    return yetToConvergeFraction;
}

} // namespace lbp 
