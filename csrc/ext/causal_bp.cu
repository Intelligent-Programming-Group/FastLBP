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

thread_local rnd_gen_type rnd_gen(42U);

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
        props.logdomain = false;
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
        } else if (updates == "TOPO") {
            props.updates = Properties::UpdateType::TOPO;
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

void CausalBP::buildFixedUpdateSeq() {
    _edgePreorderRank.clear();
    _updateSeq.clear();
    _updateSeq.reserve(nrEdges());

    std::vector<int> numbers(nrFactors());
    for(int i = 0; i < nrFactors(); ++i) {
        numbers[i] = i;
    }
    std::shuffle(numbers.begin(), numbers.end(), rnd_gen);
    for (size_t I = 0; I < nrFactors(); I++) {
        for (auto &i: nbF(numbers[I])) {
            _updateSeq.push_back(Edge(i, i.dual));
        }
    }
}

void CausalBP::buildTopoUpdateSeq() {
    Size factorOffset = nrVars();
    Size nodeCount = nrVars() + nrFactors();
    std::vector<std::vector<Size>> adjacency(nodeCount);

    auto addUndirectedEdge = [&](Size lhs, Size rhs) {
        if (lhs >= nodeCount || rhs >= nodeCount || lhs == rhs) {
            return;
        }
        adjacency[lhs].push_back(rhs);
        adjacency[rhs].push_back(lhs);
    };

    for (size_t I = 0; I < nrFactors(); I++) {
        Size factorNode = factorOffset + I;
        for (const auto &i: nbF(I)) {
            addUndirectedEdge(factorNode, static_cast<Size>(i));
        }
    }

    for (auto &neighbors: adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    std::queue<Size> ready;
    std::vector<Size> nodeOrder(nodeCount, nodeCount);
    Size order = 0;
    for (Size root = 0; root < nodeCount; root++) {
        if (nodeOrder[root] != nodeCount) {
            continue;
        }
        ready.push(root);
        nodeOrder[root] = order++;
        while (!ready.empty()) {
            Size i = ready.front();
            ready.pop();
            for (Size next: adjacency[i]) {
                if (nodeOrder[next] != nodeCount) {
                    continue;
                }
                nodeOrder[next] = order++;
                ready.push(next);
            }
        }
    }

    for (Size i = 0; i < nodeCount; i++) {
        if (nodeOrder[i] == nodeCount) {
            nodeOrder[i] = order++;
        }
    }

    _edgePreorderRank.clear();
    _edgePreorderRank.resize(nrVars());
    for (Size i = 0; i < nrVars(); i++) {
        _edgePreorderRank[i].assign(nbV(i).size(), 0);
    }

    struct RankedEdge {
        Size rank;
        Size sourceRank;
        Size factor;
        Size var;
        Size dual;
        Edge edge;
    };

    std::vector<RankedEdge> rankedEdges;
    rankedEdges.reserve(nrEdges());
    for (Size I = 0; I < nrFactors(); I++) {
        Size factorNode = factorOffset + I;
        for (auto &i: nbF(I)) {
            Size varNode = static_cast<Size>(i);
            Size sourceRank = std::min(nodeOrder[factorNode], nodeOrder[varNode]);
            Size targetRank = std::max(nodeOrder[factorNode], nodeOrder[varNode]);
            Edge edge(i, i.dual);
            rankedEdges.push_back({
                targetRank, sourceRank, I, varNode, static_cast<Size>(i.dual), edge
            });
        }
    }

    std::stable_sort(
        rankedEdges.begin(), rankedEdges.end(),
        [](const RankedEdge &lhs, const RankedEdge &rhs) {
            if (lhs.rank != rhs.rank) return lhs.rank < rhs.rank;
            if (lhs.sourceRank != rhs.sourceRank) return lhs.sourceRank < rhs.sourceRank;
            if (lhs.factor != rhs.factor) return lhs.factor < rhs.factor;
            if (lhs.var != rhs.var) return lhs.var < rhs.var;
            return lhs.dual < rhs.dual;
        }
    );

    _updateSeq.clear();
    _updateSeq.reserve(rankedEdges.size());
    for (Size order = 0; order < rankedEdges.size(); order++) {
        const auto &rankedEdge = rankedEdges[order];
        _edgePreorderRank[rankedEdge.edge.first][rankedEdge.edge.second] = order;
        _updateSeq.push_back(rankedEdge.edge);
    }
}

bool CausalBP::edgeStrictlyPrecedes(const Edge &lhs, const Edge &rhs) const {
    return _edgePreorderRank[lhs.first][lhs.second] < _edgePreorderRank[rhs.first][rhs.second];
}

void CausalBP::construct() {
    // initialize device vectors
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

    // prob/prob_default
    thrust::host_vector<Real> h_prob(nrFactors());
    thrust::host_vector<Real> h_prob_default(nrFactors());
    for (size_t I = 0; I < nrFactors(); I++) {
        h_prob[I] = factor(I).prob();
        h_prob_default[I] = factor(I).prob_default();
    }

    // indices
    thrust::host_vector<Size> h_row(message_len);
    thrust::host_vector<Size> h_col(message_len);
    thrust::host_vector<Size> h_row_ptr_vf(nrFactors() + 1);
    thrust::host_vector<Size> h_col_ind_vf(message_len);
    thrust::host_vector<Size> h_head(nrFactors());
    size_t idx = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        size_t head_label = factor(I).head().label();
        for (auto &i: nbF(I)) {
            h_row[idx] = I;
            h_col[idx] = i;
            if (var(i).label() == head_label) {
                h_head[I] = idx;
            }
            idx++;
        }
    }
    coo2csr(h_row, h_col, h_row_ptr_vf, h_col_ind_vf);

    idx = 0;
    thrust::host_vector<Size> h_row_ptr_fv(nrVars() + 1);
    thrust::host_vector<Size> h_col_ind_fv(message_len);
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

    if (props.updates == Properties::UpdateType::PARALL) {
        putParallUpdateSeqToKernelFused(
            h_row_ptr_fv, h_row_ptr_vf, h_head, 
            h_prob_default, h_prob, h_mask0, h_mask1
        );
    } else if (props.updates == Properties::UpdateType::SEQFIX) {
        buildFixedUpdateSeq();
        putUpdateSeqToKernelFused(
            h_row_ptr_fv, h_row_ptr_vf, h_head, 
            h_prob_default, h_prob, h_mask0, h_mask1
        );
    } else if (props.updates == Properties::UpdateType::TOPO) {
        buildTopoUpdateSeq();
        putUpdateSeqToKernelFused(
            h_row_ptr_fv, h_row_ptr_vf, h_head,
            h_prob_default, h_prob, h_mask0, h_mask1,
            true
        );
    } else {
        buildFixedUpdateSeq();
        putUpdateSeqToKernelFused(
            h_row_ptr_fv, h_row_ptr_vf, h_head, 
            h_prob_default, h_prob, h_mask0, h_mask1
        );
    }
}

void CausalBP::calcNewMessageFusedAll() {
    // std::cerr << "Iteration " << i << std::endl;

    // f -> v
    kernel::calcMessageFVFusedAll(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(_updateSeqAndClampedBU[0].data()),
        _updateSeqAndClampedBU[0].size()
    );

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
}

void CausalBP::calcNewMessageFused(size_t i) {
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
    kernel::calcMessageFVFused(
        thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.message_vf_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        thrust::raw_pointer_cast(_updateSeqFV[i].data()),
        _updateSeqFV[i].size()
    );
}

void CausalBP::updateMessage(size_t i) {

}

void CausalBP::calcBeliefsV(thrust::device_vector<Real> &newBeliefsV) {
    // kernel::calcBeliefsV(
    //     thrust::raw_pointer_cast(edgePropKernel.beliefs_0.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.message_fv_0.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.row_ptr_fv.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.col_ind_fv.data()),
    //     edgePropKernel.beliefs_0.size()
    // );
    // kernel::calcBeliefsV(
    //     thrust::raw_pointer_cast(edgePropKernel.beliefs_1.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.message_fv_1.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.row_ptr_fv.data()),
    //     thrust::raw_pointer_cast(edgePropKernel.col_ind_fv.data()),
    //     edgePropKernel.beliefs_1.size()
    // );
    kernel::getBeliefsV(
        thrust::raw_pointer_cast(edgePropKernel.beliefs_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_0.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_0.data()),
        nrVars()
    );
    kernel::getBeliefsV(
        thrust::raw_pointer_cast(edgePropKernel.beliefs_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.prod_fv_1.data()),
        thrust::raw_pointer_cast(edgePropKernel.num_zeros_fv_1.data()),
        nrVars()
    );
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

void CausalBP::transferBeliefToHost() {
    _beliefsV = _oldBeliefsV;
}

void CausalBP::coo2csr(
    thrust::host_vector<Size> &rows, thrust::host_vector<Size> &cols, 
    thrust::host_vector<Size> &h_row_ptr, 
    thrust::host_vector<Size> &h_col_ind
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

void CausalBP::putUpdateSeqToKernelFused(
    thrust::host_vector<Size> &h_row_ptr_fv, 
    thrust::host_vector<Size> &h_row_ptr_vf, 
    thrust::host_vector<Size> &h_head, 
    thrust::host_vector<Real> &h_prob_default, 
    thrust::host_vector<Real> &h_prob, 
    thrust::host_vector<Real> &h_mask0, 
    thrust::host_vector<Real> &h_mask1,
    bool respectPreorder
) {
    _updateSeqVF.clear();
    _updateSeqFV.clear();

    thrust::host_vector<Index3> update_seq_vf;
    thrust::host_vector<Index7Real4> update_seq_fv;
    std::set<Edge> fv_set;
    std::set<Edge> vf_set;
    std::set<Size> v_set;
    for (const auto &e: _updateSeq) {
        Size i = e.first;
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
                            if (fv_set.find(fv_e) != fv_set.end() &&
                                (!respectPreorder || edgeStrictlyPrecedes(fv_e, e))) {
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
            _updateSeqFV.push_back(update_seq_fv);

            v_set.clear();
            fv_set.clear();
            vf_set.clear();
            update_seq_vf.clear();
            update_seq_fv.clear();
        }

        v_set.insert(i);
        fv_set.insert(e);
        Size ind = h_row_ptr_fv[i] + e.second;
        Size src_ind = h_row_ptr_vf[I] + I.dual;
        Size start_ind = h_row_ptr_vf[I], end_ind = h_row_ptr_vf[I + 1];
        Size head = h_head[I];
        Real prob_default = h_prob_default[I], prob = h_prob[I];
        if (factor(I).type() == 0) {
            update_seq_fv.push_back({
                ind, i, src_ind, start_ind, end_ind, head, 0,
                prob_default, prob, h_mask0[I], h_mask1[I]
            });
        } else if (factor(I).type() == 1) {
            if (factor(I).head_clamped) {
                if (factor(I).head().label() == var(i).label()) {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 3,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                } else {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 7,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                }
            } else {
                if (factor(I).head().label() == var(i).label()) {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 1,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                } else {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 5,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                }
            }
        } else {
            if (factor(I).head_clamped) {
                if (factor(I).head().label() == var(i).label()) {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 4,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                } else {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 8,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                }
            } else {
                if (factor(I).head().label() == var(i).label()) {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 2,
                        prob_default, prob, h_mask0[I], h_mask1[I]
                    });
                } else {
                    update_seq_fv.push_back({
                        ind, i, src_ind, start_ind, end_ind, head, 6,
                        prob_default, prob, h_mask0[I], h_mask1[I]
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
                        (Size)(h_row_ptr_vf[I] + j.iter), (Size)j, (Size)(h_row_ptr_fv[j] + j.dual)
                    });
                }
            }
        }
    }
    _updateSeqVF.push_back(update_seq_vf);
    _updateSeqFV.push_back(update_seq_fv);
}

void CausalBP::putParallUpdateSeqToKernelFused(
    thrust::host_vector<Size> &h_row_ptr_fv, 
    thrust::host_vector<Size> &h_row_ptr_vf, 
    thrust::host_vector<Size> &h_head, 
    thrust::host_vector<Real> &h_prob_default, 
    thrust::host_vector<Real> &h_prob, 
    thrust::host_vector<Real> &h_mask0, 
    thrust::host_vector<Real> &h_mask1
) {
    thrust::host_vector<Index3> update_seq_vf;
    thrust::host_vector<Index6Real4> update_seq_and_clamped_bu;
    
    size_t message_len = edgePropKernel.message_fv_0.size();
    update_seq_vf.reserve(message_len);
    update_seq_and_clamped_bu.reserve(message_len);

    for (Size i = 0; i < nrVars(); i++) {
        for (auto &I: nbV(i)) {
            Size ind = h_row_ptr_fv[i] + I.iter;
            Size src_ind = h_row_ptr_vf[I] + I.dual;
            Size start_ind = h_row_ptr_vf[I], end_ind = h_row_ptr_vf[I + 1];
            Size head = h_head[I];
            Real prob_default = h_prob_default[I], prob = h_prob[I];
            if (factor(I).type() == 0) {
                // std::cout << factor(I).head().label() << " prob = " << h_prob[I] << std::endl;
                update_seq_and_clamped_bu.push_back({
                    0, i, src_ind, start_ind, end_ind, head, 
                    prob_default, prob, h_mask0[I], h_mask1[I]
                });
            } else if (factor(I).type() == 1) {
                if (factor(I).head_clamped) {
                    if (factor(I).head().label() == var(i).label()) {
                        update_seq_and_clamped_bu.push_back({
                            3, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    } else {
                        update_seq_and_clamped_bu.push_back({
                            7, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    }
                } else {
                    if (factor(I).head().label() == var(i).label()) {
                        update_seq_and_clamped_bu.push_back({
                            1, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    } else {
                        update_seq_and_clamped_bu.push_back({
                            5, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    }
                }
            } else {
                if (factor(I).head_clamped) {
                    if (factor(I).head().label() == var(i).label()) {
                        update_seq_and_clamped_bu.push_back({
                            4, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    } else {
                        update_seq_and_clamped_bu.push_back({
                            8, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    }
                } else {
                    if (factor(I).head().label() == var(i).label()) {
                        update_seq_and_clamped_bu.push_back({
                            2, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    } else {
                        update_seq_and_clamped_bu.push_back({
                            6, i, src_ind, start_ind, end_ind, head, 
                            prob_default, prob, h_mask0[I], h_mask1[I]
                        });
                    }
                }
            }
        }
    }
    for (size_t I = 0; I < nrFactors(); I++) {
        for (auto &i: nbF(I)) {
            update_seq_vf.push_back({
                (Size)(h_row_ptr_vf[I] + i.iter), (Size)i, (Size)(h_row_ptr_fv[i] + i.dual)
            });
        }
    }
    _updateSeqVF.push_back(update_seq_vf);
    _updateSeqAndClampedBU.push_back(update_seq_and_clamped_bu);
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

    Real tic = toc();
    Real maxDiff = INFINITY;

    for (; _iters < props.maxiter && maxDiff > props.tol && (toc() - tic) < props.maxtime; _iters++) {
        if (props.updates == Properties::UpdateType::PARALL) {
            calcNewMessageFusedAll();
        } else {
            for (size_t i = 0; i < _updateSeqVF.size(); i++) {  
                calcNewMessageFused(i);
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
            std::cerr << "CausalBP::run:  WARNING: not converged after " << _iters;
            std::cerr << " passes (" << toc() - tic;
            std::cerr << " seconds)...final maxdiff:" << maxDiff << std::endl;
        } else {
            if (props.verbose >= 3) {
                std::cerr << "CausalBP::run:  ";
            }
            std::cerr << "converged in " << _iters << " passes (";
            std::cerr << toc() - tic << " seconds)." << std::endl;
        }
    }
    transferBeliefToHost();
}

Real CausalBP::run(Real tolerance, size_t maxIters, size_t histLength) {
    assert(0 < tolerance);
    // assert(0 < histLength && minIters < maxIters);
    
    // std::cerr << "Starting CausalBP"
    //                    << "...  tolerance: " << tolerance
    //                    << ". minIters: " << minIters
    //                    << ". maxIters: " << maxIters
    //                    << ". histLength: " << histLength 
    //                    << "." << std::endl;

    // Real tic = toc();

    size_t numIters = 0;
    // Real maxDiff = INFINITY;
    Real yetToConvergeFraction = 1.0;
    // Real nodeFracTolerance = 0.0;
    std::vector<std::queue<Real>> beliefHist(nrVars());

    enum class RunReturnReason { ALL_CONVERGED, BIG_FRAC_CONVERGED, DIVERGED };
    RunReturnReason returnReason = RunReturnReason::DIVERGED;

    for (; true; numIters++, _iters++) {
        if (numIters >= maxIters) {
            returnReason = RunReturnReason::DIVERGED;
            break;
        }

        if (props.updates == Properties::UpdateType::PARALL) {
            calcNewMessageFusedAll();
        } else {
            for (size_t i = 0; i < _updateSeqVF.size(); i++) {  
                calcNewMessageFused(i);
                updateMessage(i);
            }
        }

        // std::cerr << "CausalBP::run():  maxdiff: " << maxDiff
        //                              << ". numIters: " << numIters
        //                              << ". Time elapsed: " << toc() - tic << " seconds. "
        //                              << "yetToConvergeFraction: " << yetToConvergeFraction << "." << std::endl;
    }

    thrust::device_vector<Real> newBeliefsV(nrVars());
    calcBeliefsV(newBeliefsV);
    _oldBeliefsV = newBeliefsV;

    transferBeliefToHost();
    switch (returnReason) {
    case RunReturnReason::ALL_CONVERGED:
        // transferBeliefToHost();
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
        // std::cerr << "CausalBP::run:  converged in " << numIters << " passes and "
        //                    << toc() - tic << " seconds. Final maxdiff: " << maxDiff << std::endl;
        break;
    case RunReturnReason::BIG_FRAC_CONVERGED:
        // std::cerr << "CausalBP::run:  Sufficiently big fraction " << yetToConvergeFraction
        //                              << " of variables appeared to converge in " << numIters << " passes and "
        //                              << toc() - tic << " seconds. Final maxDiff: " << maxDiff << std::endl;
        break;
    case RunReturnReason::DIVERGED:
        // std::cerr << "CausalBP::run:  WARNING: not converged after " << numIters << " passes and "
        //                    << toc() - tic << " seconds. Final maxdiff: " << maxDiff << std::endl;
        break;
    }

    return yetToConvergeFraction;
}

Real CausalBP::run(Real tolerance, size_t minIters, size_t maxIters, size_t histLength) {
    assert(0 < tolerance);
    assert(0 < histLength && histLength < minIters && minIters < maxIters);
    
    // std::cerr << "Starting CausalBP"
    //                    << "...  tolerance: " << tolerance
    //                    << ". minIters: " << minIters
    //                    << ". maxIters: " << maxIters
    //                    << ". histLength: " << histLength 
    //                    << "." << std::endl;

    Real tic = toc();

    size_t numIters = 0;
    Real maxDiff = INFINITY;
    Real yetToConvergeFraction = 1.0;
    Real nodeFracTolerance = 0.0;
    std::vector<std::queue<Real>> beliefHist(nrVars());

    enum class RunReturnReason { ALL_CONVERGED, BIG_FRAC_CONVERGED, DIVERGED };
    RunReturnReason returnReason = RunReturnReason::DIVERGED;

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
            calcNewMessageFusedAll();
        } else {
            for (size_t i = 0; i < _updateSeqVF.size(); i++) {  
                // calcNewMessage(i);
                calcNewMessageFused(i);
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

        thrust::host_vector<Real> bel(_oldBeliefsV);
        for( size_t i = 0; i < nrVars(); ++i ) {
            auto newBelief = bel[i];
            auto newBeliefType = fpclassify(newBelief);
            if (newBeliefType == FP_NORMAL || newBeliefType == FP_SUBNORMAL || newBeliefType == FP_ZERO) {
                beliefHist[i].push(newBelief);
            }
            if (beliefHist[i].size() > histLength) {
                beliefHist[i].pop();
            }
        }

        yetToConvergeFraction = Real(nonConvergedElems) / nrVars();
        // break;

        // std::cerr << "CausalBP::run():  maxdiff: " << maxDiff
        //                              << ". numIters: " << numIters
        //                              << ". Time elapsed: " << toc() - tic << " seconds. "
        //                              << "yetToConvergeFraction: " << yetToConvergeFraction << "." << std::endl;
    }

    if( maxDiff > _maxdiff )
        _maxdiff = maxDiff;

    transferBeliefToHost();
    switch (returnReason) {
    case RunReturnReason::ALL_CONVERGED:
        // transferBeliefToHost();
        _lowPassBeliefs = std::vector<Real>(nrVars());
        for (size_t i = 0; i < nrVars(); i++) {
            _lowPassBeliefs[i] = _beliefsV[i];
        }
        break;
    case RunReturnReason::BIG_FRAC_CONVERGED:
    case RunReturnReason::DIVERGED:
        _lowPassBeliefs = std::vector<Real>(nrVars());
        for (size_t i = 0; i < nrVars(); i++) {
            assert(beliefHist[i].size() <= histLength);
            size_t denom = beliefHist[i].size();
            while (!beliefHist[i].empty()) {
                _lowPassBeliefs[i] += beliefHist[i].front();
                beliefHist[i].pop();
            }
            if (denom > 0) { _lowPassBeliefs[i] /= denom; }
        }
        // _lowPassBeliefs = std::vector<Real>(nrVars());
        // for (size_t i = 0; i < nrVars(); i++) {
        //     _lowPassBeliefs[i] = _beliefsV[i];
        // }
        break;
    }

    switch (returnReason) {
    case RunReturnReason::ALL_CONVERGED:
        // std::cerr << "CausalBP::run:  converged in " << numIters << " passes and "
        //                    << toc() - tic << " seconds. Final maxdiff: " << maxDiff << std::endl;
        break;
    case RunReturnReason::BIG_FRAC_CONVERGED:
        // std::cerr << "CausalBP::run:  Sufficiently big fraction " << yetToConvergeFraction
        //                              << " of variables appeared to converge in " << numIters << " passes and "
        //                              << toc() - tic << " seconds. Final maxDiff: " << maxDiff << std::endl;
        break;
    case RunReturnReason::DIVERGED:
        // std::cerr << "CausalBP::run:  WARNING: not converged after " << numIters << " passes and "
        //                    << toc() - tic << " seconds. Final maxdiff: " << maxDiff << std::endl;
        break;
    }

    return yetToConvergeFraction;
}

void CausalBP::transferMessagesToHost() {
    thrust::host_vector<Real> h_message_fv_0 = edgePropKernel.message_fv_0;
    thrust::host_vector<Real> h_message_fv_1 = edgePropKernel.message_fv_1;
    size_t idx = 0;
    _edges.clear();
    _edges.reserve(nrVars());
    for (size_t i = 0; i < nrVars(); i++) {
        std::vector<EdgeProp> props;
        props.reserve(nbV(i).size());
        for (auto &I : nbV(i)) {
            std::vector<Real> v = {h_message_fv_0[idx], h_message_fv_1[idx]};
            props.push_back({Prob(v)});
            idx++;
        }
        _edges.push_back(props);
    }
}

} // namespace lbp 
