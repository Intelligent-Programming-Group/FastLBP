/// @file bp.cpp
///
/// @date 2024-08-23

#include <cuda_runtime.h>
#include <math.h>
#include <kernel/bp.h>
#include <lbp/bp.h>
#include <lbp/index.h>
#include <utils/cuda_utils.h>
#include <utils/utils.h>

namespace lbp
{

const BP::ind_t &BP::index(size_t i, size_t _I) const {
    return _edges[i][_I].index;
}

const Prob &BP::message(size_t i, size_t _I) const {
    return _edges[i][_I].message;
}

const Prob &BP::newMessage(size_t i, size_t _I) const {
    return _edges[i][_I].message;
}

void BP::setProperties(const PropertySet &opts) {
    if (opts.hasKey("tol")) {
        props.tol = opts.getStringAs<Real>("tol");
    } else {
        props.tol = 1e-9;
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
    if (opts.hasKey("verbose")) {
        props.verbose = opts.getStringAs<size_t>("verbose");
    } else {
        props.verbose = 0;
    }
}

void BP::construct() {
    // create edge properties
    _edges.clear();
    _edges.reserve(nrVars());
    for (size_t i = 0; i < nrVars(); i++) {
        _edges.push_back(std::vector<EdgeProp>());
        _edges[i].reserve(nbV(i).size());
        for (auto &I: nbV(i)) {
            EdgeProp newEP;
            newEP.index.reserve(factor(I).nrStates());
            for (IndexFor k(var(i), factor(I).vars()); k.valid(); ++k) {
                newEP.index.push_back(k);
            }
            _edges[i].push_back(newEP);
        }
    }

    std::vector<size_t> startIdxV(nrVars());
    std::vector<ind_t> startIdxF(nrFactors());
    std::vector<size_t> startIdxP(nrFactors());
    std::vector<size_t> hostOffsets;
    std::vector<size_t> hostLengths;
    for (size_t i = 0; i < nrVars(); i++) {
        startIdxV[i] = edgePropKernel.size;
        for (size_t j = 0; j < nbV(i).size(); j++) {
            hostOffsets.push_back(edgePropKernel.size + j * var(i).states());
            hostLengths.push_back(var(i).states());
            edgePropKernel.sizeVecs++;
        }
        edgePropKernel.size += var(i).states() * nbV(i).size();
    }
    size_t acc = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        startIdxF[I].reserve(nbF(I).size());
        for (auto &i: nbF(I)) {
            startIdxF[I].push_back(acc);
            acc += var(i).states();
        }
        startIdxP[I] = edgePropKernel.sizeProd;
        edgePropKernel.sizeProd += factor(I).nrStates() * nbF(I).size();
    }
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.newMessageVF, edgePropKernel.size * sizeof(Real)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.messageFV, edgePropKernel.size * sizeof(Real)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.newMessageFV, edgePropKernel.size * sizeof(Real)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.factor, edgePropKernel.sizeProd * sizeof(Real)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.nbV, edgePropKernel.size * sizeof(size_t *)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.nbF, edgePropKernel.sizeProd * sizeof(size_t *)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.index, edgePropKernel.size * sizeof(size_t *)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.numNbV, edgePropKernel.size * sizeof(size_t)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.numNbF, edgePropKernel.sizeProd * sizeof(size_t)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.numIndex, edgePropKernel.size * sizeof(size_t)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.offsets, edgePropKernel.sizeVecs * sizeof(size_t)
    ));
    CUDA_CHECK_ERROR(cudaMalloc(
        &edgePropKernel.lengths, edgePropKernel.sizeVecs * sizeof(size_t)
    ));

    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.offsets, hostOffsets.data(), 
        edgePropKernel.sizeVecs * sizeof(size_t), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.lengths, hostLengths.data(), 
        edgePropKernel.sizeVecs * sizeof(size_t), cudaMemcpyHostToDevice
    ));

    // Compute edgePropKernel.nbV
    std::vector<size_t *> nbVPtrs(edgePropKernel.size);
    std::vector<size_t> hostNumNbV(edgePropKernel.size);
    size_t i = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        for (auto &j: nbF(I)) {
            size_t j_states = var(j).states();
            for (size_t k = 0; k < j_states; k++, i++) {
                size_t size = nbV(j).size() - 1;
                size_t *deviceNbV;
                CUDA_CHECK_ERROR(cudaMalloc(&deviceNbV, size * sizeof(size_t)));
                std::vector<size_t> hostNbV;
                for (auto &J: nbV(j)) {
                    if (J != I) {
                        hostNbV.push_back(startIdxV[j] + J.iter * j_states + k);
                    }
                }
                CUDA_CHECK_ERROR(cudaMemcpy(
                    deviceNbV, hostNbV.data(), hostNbV.size() * sizeof(size_t), 
                    cudaMemcpyHostToDevice
                ));
                nbVPtrs[i] = deviceNbV;
                hostNumNbV[i] = size;
            }
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.nbV, nbVPtrs.data(), 
        edgePropKernel.size * sizeof(size_t *), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.numNbV, hostNumNbV.data(), 
        edgePropKernel.size * sizeof(size_t), cudaMemcpyHostToDevice
    ));

    // Compute edgePropKernel.nbF
    std::vector<size_t *> nbFPtrs(edgePropKernel.sizeProd);
    std::vector<size_t> hostNumNbF(edgePropKernel.sizeProd);
    i = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        for (auto &j: nbF(I)) {
            for (size_t r = 0; r < factor(I).nrStates(); r++, i++) {
                size_t size = nbF(I).size() - 1;
                size_t *deviceNbF;
                CUDA_CHECK_ERROR(cudaMalloc(&deviceNbF, size * sizeof(size_t)));
                std::vector<size_t> hostNbF;
                for (auto &j_prime: nbF(I)) {
                    if (j_prime != j) {
                        size_t _I = j_prime.dual;
                        const ind_t &ind = index(j_prime, _I);
                        hostNbF.push_back(startIdxF[I][j_prime.iter] + ind[r]);
                    }
                }
                CUDA_CHECK_ERROR(cudaMemcpy(
                    deviceNbF, hostNbF.data(), hostNbF.size() * sizeof(size_t), 
                    cudaMemcpyHostToDevice
                ));
                nbFPtrs[i] = deviceNbF;
                hostNumNbF[i] = size;
            }
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.nbF, nbFPtrs.data(), 
        edgePropKernel.sizeProd * sizeof(size_t *), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.numNbF, hostNumNbF.data(), 
        edgePropKernel.sizeProd * sizeof(size_t), cudaMemcpyHostToDevice
    ));

    // Compute edgePropKernel.index
    std::vector<size_t *> indexPtrs(edgePropKernel.size);
    std::vector<size_t> hostNumIndex(edgePropKernel.size);
    i = 0;
    for (size_t j = 0; j < nrVars(); j++) {
        for (auto &I: nbV(j)) {
            size_t j_states = var(j).states();
            size_t size = factor(I).nrStates() / j_states;
            std::vector<size_t *> deviceIndex(j_states);
            for (size_t k = 0; k < j_states; k++) {
                CUDA_CHECK_ERROR(cudaMalloc(&deviceIndex[k], size * sizeof(size_t)));
            }
            std::vector<ind_t> hostIndex(j_states);

            size_t _I = I.iter;
            const ind_t &ind = index(j, _I);
            for (size_t r = 0; r < factor(I).nrStates(); r++) {
                hostIndex[ind[r]].push_back(startIdxP[I] + factor(I).nrStates() * I.dual + r);
            }
            for (size_t k = 0; k < j_states; k++, i++) {
                CUDA_CHECK_ERROR(cudaMemcpy(
                    deviceIndex[k], hostIndex[k].data(), 
                    hostIndex[k].size() * sizeof(size_t), cudaMemcpyHostToDevice
                ));
                indexPtrs[i] = deviceIndex[k];
                hostNumIndex[i] = size;
            }
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.index, indexPtrs.data(), 
        edgePropKernel.size * sizeof(size_t *), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.numIndex, hostNumIndex.data(),
        edgePropKernel.size * sizeof(size_t), cudaMemcpyHostToDevice
    ));

    // create old beliefs
    sizeBV = 0;
    std::vector<Real> hostBeliefsV;
    for (size_t i = 0; i < nrVars(); i++) {
        sizeBV += var(i).states();
        auto v = Factor(var(i)).toVec();
        hostBeliefsV.insert(hostBeliefsV.end(), v.begin(), v.end());
    }
    CUDA_CHECK_ERROR(cudaMalloc(&_oldBeliefsV, sizeBV * sizeof(Real)));
    CUDA_CHECK_ERROR(cudaMemcpy(
        _oldBeliefsV, hostBeliefsV.data(), sizeBV * sizeof(Real),
        cudaMemcpyHostToDevice
    ));
    
    sizeBF = 0;
    std::vector<Real> hostBeliefsF;
    for (size_t I = 0; I < nrFactors(); I++) {
        sizeBF += factor(I).nrStates();
        auto v = Factor(factor(I).vars()).toVec();
        hostBeliefsF.insert(hostBeliefsF.end(), v.begin(), v.end());
    }
    CUDA_CHECK_ERROR(cudaMalloc(&_oldBeliefsF, sizeBF * sizeof(Real)));
    CUDA_CHECK_ERROR(cudaMemcpy(
        _oldBeliefsF, hostBeliefsF.data(), sizeBF * sizeof(Real),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK_ERROR(cudaMalloc(&nbBV, sizeBV * sizeof(size_t *)));
    CUDA_CHECK_ERROR(cudaMalloc(&numNbBV, sizeBV * sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&nbBF, sizeBF * sizeof(size_t *)));
    CUDA_CHECK_ERROR(cudaMalloc(&numNbBF, sizeBF * sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&factorB, sizeBF * sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&offsetsBV, nrVars() * sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&lengthsBV, nrVars() * sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&offsetsBF, nrFactors() * sizeof(size_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&lengthsBF, nrFactors() * sizeof(size_t)));
    std::vector<size_t *> nbBVPtrs(sizeBV);
    std::vector<size_t> hostNumNbBV(sizeBV);
    i = 0;
    for (size_t j = 0; j < nrVars(); j++) {
        size_t size = nbV(j).size();
        size_t j_states = var(j).states();
        for (size_t k = 0; k < j_states; k++, i++) {
            size_t *deviceNbBV;
            CUDA_CHECK_ERROR(cudaMalloc(&deviceNbBV, size * sizeof(size_t)));
            std::vector<size_t> hostNbBV;
            for (auto &J: nbV(j)) {
                hostNbBV.push_back(startIdxV[j] + J.iter * j_states + k);
            }
            CUDA_CHECK_ERROR(cudaMemcpy(
                deviceNbBV, hostNbBV.data(), hostNbBV.size() * sizeof(size_t), 
                cudaMemcpyHostToDevice
            ));
            nbBVPtrs[i] = deviceNbBV;
            hostNumNbBV[i] = size;
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        nbBV, nbBVPtrs.data(), sizeBV * sizeof(size_t *), 
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        numNbBV, hostNumNbBV.data(), sizeBV * sizeof(size_t), 
        cudaMemcpyHostToDevice
    ));

    std::vector<size_t *> nbBFPtrs(sizeBF);
    std::vector<size_t> hostNumNbBF(sizeBF);
    i = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        for (size_t r = 0; r < factor(I).nrStates(); r++, i++) {
            size_t size = nbF(I).size();
            size_t *deviceNbBF;
            CUDA_CHECK_ERROR(cudaMalloc(&deviceNbBF, size * sizeof(size_t)));
            std::vector<size_t> hostNbBF;
            for (auto &j: nbF(I)) {
                size_t _I = j.dual;
                const ind_t &ind = index(j, _I);
                hostNbBF.push_back(startIdxF[I][j.iter] + ind[r]);
            }
            CUDA_CHECK_ERROR(cudaMemcpy(
                deviceNbBF, hostNbBF.data(), hostNbBF.size() * sizeof(size_t), 
                cudaMemcpyHostToDevice
            ));
            nbBFPtrs[i] = deviceNbBF;
            hostNumNbBF[i] = size;
        }
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        nbBF, nbBFPtrs.data(), sizeBF * sizeof(size_t *), 
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        numNbBF, hostNumNbBF.data(), sizeBF * sizeof(size_t), 
        cudaMemcpyHostToDevice
    ));

    std::vector<size_t> hostOffsetsBV;
    std::vector<size_t> hostLengthsBV;
    std::vector<size_t> hostOffsetsBF;
    std::vector<size_t> hostLengthsBF;
    size_t offset = 0;
    for (size_t i = 0; i < nrVars(); i++) {
        hostOffsetsBV.push_back(offset);
        hostLengthsBV.push_back(var(i).states());
        offset += var(i).states();
    }
    offset = 0;
    for (size_t I = 0; I < nrFactors(); I++) {
        hostOffsetsBF.push_back(offset);
        hostLengthsBF.push_back(factor(I).nrStates());
        offset += factor(I).nrStates();
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        offsetsBV, hostOffsetsBV.data(), nrVars() * sizeof(size_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        lengthsBV, hostLengthsBV.data(), nrVars() * sizeof(size_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        offsetsBF, hostOffsetsBF.data(), nrFactors() * sizeof(size_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        lengthsBF, hostLengthsBF.data(), nrFactors() * sizeof(size_t),
        cudaMemcpyHostToDevice
    ));
}

void BP::calcNewMessage() {
    kernel::initArray(edgePropKernel.newMessageVF, 1.0, edgePropKernel.size);
    kernel::calcProduct(
        edgePropKernel.newMessageVF, edgePropKernel.messageFV, 
        edgePropKernel.nbV, edgePropKernel.numNbV, edgePropKernel.size
    );
    
    Real *product;
    CUDA_CHECK_ERROR(cudaMalloc(
        &product, edgePropKernel.sizeProd * sizeof(Real)
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        product, edgePropKernel.factor, 
        edgePropKernel.sizeProd * sizeof(Real), cudaMemcpyDeviceToDevice
    ));

    kernel::calcProduct(
        product, edgePropKernel.newMessageVF, edgePropKernel.nbF, 
        edgePropKernel.numNbF, edgePropKernel.sizeProd
    );
    kernel::initArray(edgePropKernel.newMessageFV, 0.0, edgePropKernel.size);
    kernel::marginalize(
        edgePropKernel.newMessageFV, product, edgePropKernel.index, 
        edgePropKernel.numIndex, edgePropKernel.size
    );
    kernel::normalize(
        edgePropKernel.newMessageFV, edgePropKernel.offsets, 
        edgePropKernel.lengths, edgePropKernel.sizeVecs
    );
    CUDA_CHECK_ERROR(cudaFree(product));
}

void BP::updateMessage() {
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.messageFV, edgePropKernel.newMessageFV, 
        edgePropKernel.size * sizeof(Real), cudaMemcpyDeviceToDevice
    ));
}

void BP::calcBeliefsV(Real *newBeliefsV) {
    kernel::initArray(newBeliefsV, 1.0, sizeBV);
    kernel::calcProduct(
        newBeliefsV, edgePropKernel.newMessageFV, nbBV, numNbBV, sizeBV
    );
    kernel::normalize(newBeliefsV, offsetsBV, lengthsBV, nrVars());
}

void BP::calcBeliefsF(Real *newBeliefsF) {
    kernel::initArray(edgePropKernel.newMessageVF, 1.0, edgePropKernel.size);
    kernel::calcProduct(
        edgePropKernel.newMessageVF, edgePropKernel.messageFV, 
        edgePropKernel.nbV, edgePropKernel.numNbV, edgePropKernel.size
    );
    CUDA_CHECK_ERROR(cudaMemcpy(
        newBeliefsF, factorB, sizeBF * sizeof(Real), cudaMemcpyDeviceToDevice
    ));
    kernel::calcProduct(
        newBeliefsF, edgePropKernel.newMessageVF, nbBF, numNbBF, sizeBF
    );
    kernel::normalize(newBeliefsF, offsetsBF, lengthsBF, nrFactors());
}

void BP::transferMessageToHost() {
    std::vector<Real> hostMessage(edgePropKernel.size);
    CUDA_CHECK_ERROR(cudaMemcpy(
        hostMessage.data(), edgePropKernel.messageFV, 
        edgePropKernel.size * sizeof(Real), cudaMemcpyDeviceToHost
    ));

    size_t offset = 0;
    for (size_t i = 0; i < nrVars(); i++) {
        for (size_t k = 0; k < nbV(i).size(); k++) {
            std::vector<Real> v;
            v.reserve(var(i).states());
            v.insert(
                v.end(), hostMessage.begin() + offset, 
                hostMessage.begin() + offset + var(i).states()
            );
            _edges[i][k].message = Prob(v);
            offset += var(i).states();
        }
    }
}

BP::BP(const FactorGraph &fg, const PropertySet &opts): DAIAlgFG(fg), _iters(0), edgePropKernel() {
    setProperties(opts);
    construct();
}

BP::~BP() {
    std::vector<size_t *> hostNbV(edgePropKernel.size);
    CUDA_CHECK_ERROR(cudaMemcpy(
        hostNbV.data(), edgePropKernel.nbV, 
        edgePropKernel.size * sizeof(size_t *), cudaMemcpyDeviceToHost
    ));
    std::vector<size_t *> hostNbF(edgePropKernel.sizeProd);
    CUDA_CHECK_ERROR(cudaMemcpy(
        hostNbF.data(), edgePropKernel.nbF, 
        edgePropKernel.sizeProd * sizeof(size_t *), cudaMemcpyDeviceToHost
    ));
    std::vector<size_t *> hostIndex(edgePropKernel.size);
    CUDA_CHECK_ERROR(cudaMemcpy(
        hostIndex.data(), edgePropKernel.index, 
        edgePropKernel.size * sizeof(size_t *), cudaMemcpyDeviceToHost
    ));
    std::vector<size_t *> hostNbBV(sizeBV);
    CUDA_CHECK_ERROR(cudaMemcpy(
        hostNbBV.data(), nbBV, sizeBV * sizeof(size_t *), cudaMemcpyDeviceToHost
    ));
    std::vector<size_t *> hostNbBF(sizeBF);
    CUDA_CHECK_ERROR(cudaMemcpy(
        hostNbBF.data(), nbBF, sizeBF * sizeof(size_t *), cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.newMessageVF));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.messageFV));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.newMessageFV));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.factor));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.numNbV));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.numNbF));
    for (size_t i = 0; i < edgePropKernel.size; i++) {
        CUDA_CHECK_ERROR(cudaFree(hostNbV[i]));
    }
    for (size_t i = 0; i < edgePropKernel.sizeProd; i++) {
        CUDA_CHECK_ERROR(cudaFree(hostNbF[i]));
    }
    for (size_t i = 0; i < edgePropKernel.size; i++) {
        CUDA_CHECK_ERROR(cudaFree(hostIndex[i]));
    }
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.nbV));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.nbF));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.numIndex));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.offsets));
    CUDA_CHECK_ERROR(cudaFree(edgePropKernel.lengths));
    CUDA_CHECK_ERROR(cudaFree(_oldBeliefsV));
    CUDA_CHECK_ERROR(cudaFree(_oldBeliefsF));
    for (size_t i = 0; i < sizeBV; i++) {
        CUDA_CHECK_ERROR(cudaFree(hostNbBV[i]));
    }
    for (size_t i = 0; i < sizeBF; i++) {
        CUDA_CHECK_ERROR(cudaFree(hostNbBF[i]));
    }
    CUDA_CHECK_ERROR(cudaFree(nbBV));
    CUDA_CHECK_ERROR(cudaFree(nbBF));
    CUDA_CHECK_ERROR(cudaFree(numNbBV));
    CUDA_CHECK_ERROR(cudaFree(numNbBF));
    CUDA_CHECK_ERROR(cudaFree(factorB));
}

void BP::init() {
    std::cerr << "Initializing message" << std::endl;
    kernel::initArray(edgePropKernel.messageFV, 1.0, edgePropKernel.size);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Initialize `edgePropKernel.factor` and `factorB`
    std::vector<Real> hostFactor;
    std::vector<Real> hostFactorB;
    hostFactor.reserve(edgePropKernel.sizeProd);
    hostFactorB.reserve(sizeBF);
    for (size_t I = 0; I < nrFactors(); I++) {
        auto &v = factor(I).toVec();
        for (size_t j = 0; j < nbF(I).size(); j++) {
            hostFactor.insert(hostFactor.end(), v.begin(), v.end());
        }
        hostFactorB.insert(hostFactorB.end(), v.begin(), v.end());
    }
    CUDA_CHECK_ERROR(cudaMemcpy(
        edgePropKernel.factor, hostFactor.data(), 
        edgePropKernel.sizeProd * sizeof(Real), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK_ERROR(cudaMemcpy(
        factorB, hostFactorB.data(), sizeBF * sizeof(Real), 
        cudaMemcpyHostToDevice
    ));
}

void BP::run() {
    if (props.verbose >= 1) {
        std::cerr << "Starting ..." << std::endl;
    }

    Real tic = toc();

    // do several passes over the network until maximum number of iterations has
    // been reached or until the maximum belief difference is smaller than 
    // tolerance
    Real maxDiff = INFINITY;
    for (; _iters < props.maxiter && maxDiff > props.tol && (toc() - tic) < props.maxtime; _iters++) {
        // Parallel updates
        calcNewMessage();
        updateMessage();

        // calculate new beliefs and compare with old ones
        maxDiff = -INFINITY;
        Real *newBeliefsV;
        CUDA_CHECK_ERROR(cudaMalloc(&newBeliefsV, sizeBV * sizeof(Real)));
        calcBeliefsV(newBeliefsV);
        kernel::dist(_oldBeliefsV, newBeliefsV, sizeBV);
        maxDiff = std::max(maxDiff, kernel::calcMax(_oldBeliefsV, sizeBV));

        CUDA_CHECK_ERROR(cudaMemcpy(
            _oldBeliefsV, newBeliefsV, sizeBV * sizeof(Real), 
            cudaMemcpyDeviceToDevice
        ));
        CUDA_CHECK_ERROR(cudaFree(newBeliefsV));
        Real *newBeliefsF;
        CUDA_CHECK_ERROR(cudaMalloc(&newBeliefsF, sizeBF * sizeof(Real)));
        calcBeliefsF(newBeliefsF);
        kernel::dist(_oldBeliefsF, newBeliefsF, sizeBF);
        maxDiff = std::max(maxDiff, kernel::calcMax(_oldBeliefsF, sizeBF));
        CUDA_CHECK_ERROR(cudaMemcpy(
            _oldBeliefsF, newBeliefsF, sizeBF * sizeof(Real), 
            cudaMemcpyDeviceToDevice
        ));
        CUDA_CHECK_ERROR(cudaFree(newBeliefsF));

        if (props.verbose >= 3) {
            std::cerr << "BP::run:  maxdiff " << maxDiff << " after ";
            std::cerr << _iters + 1  << " passes" << std::endl;
        }
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

Prob BP::calcIncomingMessageProduct(size_t I, bool without_i, size_t i ) const {
    Factor Fprod(factor(I));
    Prob &prod = Fprod.p();

    for (auto &j: nbF(I)) {
        if (!(without_i && (j == i))) {
            Prob prod_j(var(j).states(), 1.0);
            for (auto &J: nbV(j)) {
                if (J != I) {
                    prod_j *= message(j, J.iter);
                }
            }

            size_t _I = j.dual;
            const ind_t &ind = index(j, _I);
            for (size_t r = 0; r < prod.size(); r++) {
                prod.set(r, prod[r] * prod_j[ind[r]]);
            }
        }
    }
    return prod;
}

void BP::calcBeliefV(size_t i, Prob &p) const {
    p = Prob(var(i).states(), 1.0);
    for (auto &I: nbV(i)) {
        p *= newMessage(i, I.iter);
    }
}

void BP::calcBeliefF( size_t I, Prob &p ) const {
    p = calcIncomingMessageProduct( I, false, 0 );
}

Factor BP::beliefV(size_t i) const {
    Prob p;
    calcBeliefV(i, p);
    p.normalize();
    return Factor(var(i), p);
}

Factor BP::beliefF( size_t I ) const {
    Prob p;
    calcBeliefF( I, p );
    p.normalize();
    return( Factor( factor(I).vars(), p ) );
}

Factor BP::belief(const VarSet &vs) const {
    if (vs.size() == 0) {
        return Factor();
    } else if (vs.size() == 1) {
        return beliefV(findVar(*(vs.begin())));
    } else {
        size_t I;
        for (I = 0; I < nrFactors(); I++) {
            if (factor(I).vars() >> vs) {
                break;
            }
        }
        if (I == nrFactors()) {
            std::cerr << "Belief not available!" << std::endl;
        }
        return beliefF(I).marginal(vs);
    }
}

} // namespace lbp
