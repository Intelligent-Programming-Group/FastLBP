/// @file bp.h
/// @brief Defines class BP, which implements (Loopy) Belief Propagation
///
/// @date 2024-07-25

#pragma once

#include <lbp/daialg.h>
#include <lbp/factorgraph.h>
#include <lbp/properties.h>

namespace lbp {

class BP: public DAIAlgFG {
private:
    /* data */
    /// @brief Parameters for BP
    struct Properties {
        /// @brief Verbosity (amount of output sent to stderr)
        size_t verbose;
        /// @brief Maximum number of iterations
        size_t maxiter;
        /// @brief Maximum time (in seconds)
        Real maxtime;
        /// @brief Tolerance for convergence test
        Real tol;
    } props;

    struct EdgePropKernel {
        Real *newMessageVF;
        Real *messageFV;
        Real *newMessageFV;
        Real *factor;
        size_t **nbV;
        size_t **nbF;
        size_t **index;
        size_t *numNbV;
        size_t *numNbF;
        size_t *numIndex;
        size_t size;
        size_t sizeProd;
        size_t sizeVecs;
        size_t *offsets;
        size_t *lengths;
    } edgePropKernel;
    /// @brief Type used for index cache
    typedef std::vector<size_t> ind_t;
    /// @brief Type used for storing edge properties
    struct EdgeProp {
        /// @brief Index cached for this edge
        ind_t index;
        /// @brief Old message living on this edge
        Prob message;
    };
    /// @brief Stores all edge properties
    std::vector<std::vector<EdgeProp>> _edges;
    /// @brief Number of iterations needed
    size_t _iters;
    /// @brief Stores variable beliefs of previous iteration
    Real *_oldBeliefsV;
    /// @brief Stores factor beliefs of previous iteration
    Real *_oldBeliefsF;
    /// @brief 
    size_t sizeBV;
    /// @brief 
    size_t sizeBF;
    /// @brief 
    size_t **nbBV;
    /// @brief 
    size_t **nbBF;
    /// @brief 
    size_t *numNbBV;
    /// @brief 
    size_t *numNbBF;
    /// @brief 
    Real *factorB;
    size_t *offsetsBV;
    size_t *lengthsBV;
    size_t *offsetsBF;
    size_t *lengthsBF;

    /// @brief 
    /// @param i 
    /// @param I 
    /// @return constant reference to cached index for the edge between 
    /// variable \a i and its \a _I 'th neighbor
    const ind_t &index(size_t i, size_t I) const;
    const Prob & message(size_t i, size_t _I) const;
    /// @brief 
    /// @param i 
    /// @param _I 
    /// @return constant reference to updated message from the \a _I 'th 
    /// neighbor of variable \a i to variable \a i
    const Prob & newMessage(size_t i, size_t _I) const;
    /// @brief 
    /// @param opts 
    void setProperties(const PropertySet &opts);
    /// @brief Helper function for constructors
    void construct();
    /// @brief Parallelly calculate all messages over the network
    void calcNewMessage();
    /// @brief Replace the "old" message from the neighbors of variables to 
    /// variables by the "new" (updated) message
    void updateMessage();
    /// @brief Calculates normalized beliefs of all variables
    /// @param newBeliefsV 
    void calcBeliefsV(Real *newBeliefsV);
    /// @brief Calculates normalized beliefs of all factors
    /// @param newBeliefsF 
    void calcBeliefsF(Real *newBeliefsF);
    /// @brief Calculates unnormalized belief of variable \a i
    /// @param i 
    /// @param p 
    void calcBeliefV(size_t i, Prob &p) const;
    /// @brief Calculates unnormalized belief of factor \a I
    /// @param I 
    /// @param p 
    void calcBeliefF( size_t I, Prob &p ) const;

    /// @brief 
    void transferMessageToHost();
public:
    /// @brief Default constructor
    BP(/* args */) = default;

    /// @brief Construct from `FactorGraph` `fg`
    /// @param fg Factor graph
    BP(const FactorGraph &fg, const PropertySet &opts);

    ~BP();

    Prob calcIncomingMessageProduct( size_t I, bool without_i, size_t i ) const;
    Factor belief(const VarSet &vs) const;
    Factor beliefV(size_t i) const;
    Factor beliefF( size_t I ) const;
    void init();
    void run();
};

} // namespace lbp 
