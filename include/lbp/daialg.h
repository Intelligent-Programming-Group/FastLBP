/// @file daialg.h
///
/// @brief
/// @date 2024-09-04

#pragma once

#include <lbp/factorgraph.h>

namespace lbp {

class InfAlg {
private:
    /* data */
public:
    InfAlg(/* args */) = default;
    ~InfAlg() = default;
};

template <class GRM>
class DAIAlg: public InfAlg, public GRM {
private:
    /* data */
public:
    /// @brief Default constructor
    DAIAlg(/* args */) = default;

    /// @brief Construct from GRM
    /// @param grm 
    DAIAlg(const GRM &grm): InfAlg(), GRM(grm) {}
    
    ~DAIAlg() = default;
};

typedef DAIAlg<FactorGraph> DAIAlgFG;
    
} // namespace lbp 
