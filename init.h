#pragma once
#include <iostream>
#include <fstream>
#include <typeinfo>

#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "state.h"
#include "constants.h"

using namespace std;

template<typename FT, typename IT>
class Init {
public:
    /* Mutable State */
    typedef Matrix<FT,Dynamic,Dynamic,RowMajor> MatrixW;
    typedef Matrix<FT,Dynamic,Dynamic,RowMajor> MatrixWff;
    typedef Matrix<IT,Dynamic,Dynamic,RowMajor> MatrixIncomingSpikes;
    typedef Matrix<IT,1,NBNEUR,RowMajor> VectorFirings;

    static MatrixW initW() {
        MatrixW w = MatrixW::Zero(NBNEUR,NBNEUR); //MatrixXf::Zero(NBNEUR, NBNEUR);
        w.bottomRows(NBI).leftCols(NBE).setRandom(); // Inhbitory neurons receive excitatory inputs from excitatory neurons
        w.rightCols(NBI).setRandom(); // Everybody receives inhibition (including inhibitory neurons)
        w.bottomRows(NBI).rightCols(NBI) =  -w.bottomRows(NBI).rightCols(NBI).cwiseAbs() * WII_MAX;
        w.topRows(NBE).rightCols(NBI) = -w.topRows(NBE).rightCols(NBI).cwiseAbs() * WIE_MAX;
        w.bottomRows(NBI).leftCols(NBE) = w.bottomRows(NBI).leftCols(NBE).cwiseAbs() * WEI_MAX;
        w = w - w.cwiseProduct(MatrixW::Identity(NBNEUR,NBNEUR)); // Diagonal lateral weights are 0 (no autapses !)
        return w;
    }

    static MatrixWff initWff() {
        MatrixWff wff = MatrixWff::Zero(NBNEUR,FFRFSIZE);
        wff = (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixWff::Random(NBNEUR,FFRFSIZE).cwiseAbs().array()).cwiseMin(MAXW);
        wff.bottomRows(NBI).setZero();
        return wff;
    }

    static MatrixIncomingSpikes initIncomingSpikes() {
        MatrixIncomingSpikes incomingSpikes = MatrixIncomingSpikes::Zero(NBNEUR,NBNEUR);
        return incomingSpikes;
    }

    static VectorFirings initFirings() {
        VectorFirings firings = VectorFirings::Zero();
        return firings;
    }

    /* Static State */
    typedef Matrix<IT,Dynamic,Dynamic,RowMajor> MatrixDelays;

    static MatrixDelays initDelays() {
        MatrixDelays delays = MatrixDelays::Zero(NBNEUR,NBNEUR);
        return delays;
    }

    /* Buffers */
    typedef Matrix<FT,Dynamic,Dynamic,RowMajor> MatrixLgnFiringsBuffer;
    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorNeuronInputsBuffer;

    static MatrixLgnFiringsBuffer initLgnFiringsBuffer() {
        MatrixLgnFiringsBuffer lgnfiringsBuffer =
            MatrixLgnFiringsBuffer::Zero(NBSTEPSSTIM,FFRFSIZE);
        return lgnfiringsBuffer;
    }

    static VectorNeuronInputsBuffer initNeuronInputsBuffer() {
        VectorNeuronInputsBuffer neuronInputsBuffer =
            VectorNeuronInputsBuffer::Zero();
        return neuronInputsBuffer;
    }


private:
    BOOST_STATIC_ASSERT(is_same<float,FT>::value || is_same<double,FT>::value);
    BOOST_STATIC_ASSERT(is_same<int,IT>::value || is_same<long,IT>::value);
};

/* Mutable State */
template <typename FT, typename IT>
    using MatrixW = typename Init<FT,IT>::MatrixW;

template <typename FT, typename IT>
    using MatrixWff = typename Init<FT,IT>::MatrixWff;

template <typename FT, typename IT>
    using MatrixIncomingSpikes = typename Init<FT,IT>::MatrixIncomingSpikes;

template <typename FT, typename IT>
    using VectorFirings = typename Init<FT,IT>::VectorFirings;

/* Static State */
template <typename FT, typename IT>
    using MatrixDelays = typename Init<FT,IT>::MatrixDelays;

/* Buffers */
template <typename FT, typename IT>
    using MatrixLgnFiringsBuffer = typename Init<FT,IT>::MatrixLgnFiringsBuffer;

template <typename FT, typename IT>
    using VectorNeuronInputsBuffer = typename Init<FT,IT>::VectorNeuronInputsBuffer;
