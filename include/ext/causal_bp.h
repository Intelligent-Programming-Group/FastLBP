/// @file causal_bp.h
/// @brief Defines class `CausalBP`, which implements (loopy) belief 
/// propagation with local structure.
///
/// @date 2025-1-14

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <ext/causal_fg.h>
#include <lbp/daialg.h>
#include <lbp/properties.h>
#include <utils/utils.h>
#include <kernel/causal_bp_seq.h>

namespace lbp {

class CausalBP: public DAIAlg<CausalFactorGraph> {
private:
    /* data */
    /// @brief Parameters for BP
    struct Properties {
        /// @brief Enumeration of possible update schedules
        /// The following update schedules have been defined:
        ///  - SEQFIX sequential updates using a fixed sequence
        ///  - PARALL parallel updates
        ///  - TOPO topological preorder over a tree-shaped factor graph
        enum class UpdateType {
            SEQFIX,
            PARALL,
            TOPO
        };

        /// @brief Verbosity (amount of output sent to stderr)
        size_t verbose;
        /// @brief Maximum number of iterations
        size_t maxiter;
        /// @brief Maximum time (in seconds)
        Real maxtime;
        /// @brief Tolerance for convergence test
        Real tol;
        /// @brief Whether updates should be done in logarithmic domain or not
        bool logdomain;
        /// @brief Message update schedule
        UpdateType updates;
    } props;
    Real _maxdiff;
    /// @brief Number of iterations needed
    size_t _iters;
    /// @brief 
    struct EdgeProp {
        /// @brief Old message living on this edge
        Prob message;
    };
    /// @brief Stores all edge properties
    std::vector<std::vector<EdgeProp>> _edges;

    struct EdgePropKernel {
        thrust::device_vector<Size> row_ptr_fv;
        thrust::device_vector<Size> col_ind_fv;

        thrust::device_vector<Real> prod_fv_0;
        thrust::device_vector<Real> prod_fv_1;
        thrust::device_vector<Size> num_zeros_fv_0;
        thrust::device_vector<Size> num_zeros_fv_1;
        thrust::device_vector<Real> message_fv_0;
        thrust::device_vector<Real> message_fv_1;
        thrust::device_vector<Real> message_vf_0;
        thrust::device_vector<Real> message_vf_1;

        thrust::device_vector<Real> beliefs_0;
        thrust::device_vector<Real> beliefs_1;
    } edgePropKernel;

    struct HostProp {
        thrust::host_vector<Size> h_row_ptr_fv;
        thrust::host_vector<Size> h_row_ptr_vf;
        thrust::host_vector<Size> h_head; 
        thrust::host_vector<Real> h_prob_default;
        thrust::host_vector<Real> h_prob; 
        thrust::host_vector<Real> h_mask0;
        thrust::host_vector<Real> h_mask1;
    } hostProp;

    thrust::device_vector<Real> _oldBeliefsV;
    thrust::device_vector<Real> _oldBeliefsF;

    thrust::host_vector<Real> _beliefsV;
    std::vector<Real> _lowPassBeliefs;
    std::vector<Edge> _updateSeq;
    std::vector<std::vector<Size>> _edgePreorderRank;
    std::vector<thrust::device_vector<Index3>> _updateSeqVF;
    std::vector<thrust::device_vector<Index2Real>> _updateSeqI;
    std::vector<thrust::device_vector<Index5Real2>> _updateSeqAndTD;
    std::vector<thrust::device_vector<Index5Real2>> _updateSeqOrTD;
    std::vector<thrust::device_vector<Index5Real4>> _updateSeqAndClampedTD;
    std::vector<thrust::device_vector<Index5Real4>> _updateSeqOrClampedTD;
    std::vector<thrust::device_vector<Index6Real2>> _updateSeqAndBU;
    std::vector<thrust::device_vector<Index6Real2>> _updateSeqOrBU;
    std::vector<thrust::device_vector<Index6Real4>> _updateSeqAndClampedBU;
    std::vector<thrust::device_vector<Index6Real4>> _updateSeqOrClampedBU;
    std::vector<thrust::device_vector<Index7Real4>> _updateSeqFV;

    /// @brief 
    /// @param opts 
    void setProperties(const PropertySet &opts);
    /// @brief Helper function for constructors
    void construct();
    /// @brief Build the default fixed edge order used by sequential updates.
    void buildFixedUpdateSeq();
    /// @brief Build a topological preorder over a tree-shaped factor graph.
    void buildTopoUpdateSeq();
    /// @brief Return true iff lhs is a strict predecessor of rhs in the active preorder.
    bool edgeStrictlyPrecedes(const Edge &lhs, const Edge &rhs) const;
    /// @brief Parallelly calculate all messages over the network
    void calcNewMessageFusedAll();
    /// @brief Parallelly calculate all messages over the network
    void calcNewMessageFused(size_t i);
    /// @brief Replace the "old" message from the neighbors of variables to 
    /// variables by the "new" (updated) message
    void updateMessage(size_t i);
    /// @brief Calculates normalized beliefs of all variables
    /// @param newBeliefsV 
    void calcBeliefsV(thrust::device_vector<Real> &newBeliefsV);
    /// @brief Calculates normalized beliefs of all factors
    /// @param newBeliefsF 
    void calcBeliefsF(thrust::device_vector<Real> &newBeliefsF);
    /// @brief 
    /// @param i 
    /// @param p 
    void calcBeliefV(size_t i, Prob &p) const;

    /// @brief 
    /// @param rows 
    /// @param cols 
    /// @param d_row_ptr 
    /// @param d_col_ind 
    void coo2csr(
        thrust::host_vector<Size> &rows, thrust::host_vector<Size> &cols, 
        thrust::host_vector<Size> &h_row_ptr, 
        thrust::host_vector<Size> &h_col_ind
    );
    /// @brief 
    void transferBeliefToHost();
    /// @brief 
    void putUpdateSeqToKernelFused(
        thrust::host_vector<Size> &h_row_ptr_fv, 
        thrust::host_vector<Size> &h_row_ptr_vf, 
        thrust::host_vector<Size> &h_head, 
        thrust::host_vector<Real> &h_prob_default, 
        thrust::host_vector<Real> &h_prob, 
        thrust::host_vector<Real> &h_mask0, 
        thrust::host_vector<Real> &h_mask1,
        bool respectPreorder = false
    );
    /// @brief 
    void putParallUpdateSeqToKernelFused(
        thrust::host_vector<Size> &h_row_ptr_fv, 
        thrust::host_vector<Size> &h_row_ptr_vf, 
        thrust::host_vector<Size> &h_head, 
        thrust::host_vector<Real> &h_prob_default, 
        thrust::host_vector<Real> &h_prob, 
        thrust::host_vector<Real> &h_mask0, 
        thrust::host_vector<Real> &h_mask1
    );
public:
    /// @brief Construct from `CausalFactorGraph` `fg`
    /// @param fg 
    /// @param opts 
    CausalBP(const CausalFactorGraph &fg, const PropertySet &opts);

    Factor belief(const VarSet &vs) const;
    Factor beliefV(size_t i) const;
    /// @brief 
    void init();
    /// @brief 
    void run();

    Real run(Real tolerance, size_t maxIters, size_t histLength);
    Real run(Real tolerance, size_t minIters, size_t maxIters, size_t histLength);
    Real newBelief(size_t varIndex) const {
        return _lowPassBeliefs[varIndex];
    }

    void transferMessagesToHost();

    /// @brief 
    /// @param i 
    /// @param _I 
    /// @return 
    const Prob &newMessage(size_t i, size_t _I) const;
};
    
} // namespace lbp 
