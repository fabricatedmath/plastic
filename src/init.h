#pragma once
#include <iostream>
#include <fstream>
#include <typeinfo>

#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "state.h"
#include "dataset.h"
#include "constants.h"

using namespace std;

template<typename F, typename I>
class Init {
public:
    /* Mutable State */
    static MatrixRX<F> initW() {
        MatrixRX<F> w = MatrixRX<F>::Zero(NBNEUR,NBNEUR); //MatrixXf::Zero(NBNEUR, NBNEUR);
        w.bottomRows(NBI).leftCols(NBE).setRandom(); // Inhbitory neurons receive excitatory inputs from excitatory neurons
        w.rightCols(NBI).setRandom(); // Everybody receives inhibition (including inhibitory neurons)
        w.bottomRows(NBI).rightCols(NBI) =  -w.bottomRows(NBI).rightCols(NBI).cwiseAbs() * WII_MAX;
        w.topRows(NBE).rightCols(NBI) = -w.topRows(NBE).rightCols(NBI).cwiseAbs() * WIE_MAX;
        w.bottomRows(NBI).leftCols(NBE) = w.bottomRows(NBI).leftCols(NBE).cwiseAbs() * WEI_MAX;
        w = w - w.cwiseProduct(MatrixRX<F>::Identity(NBNEUR,NBNEUR)); // Diagonal lateral weights are 0 (no autapses !)
        return w;
    }

    static MatrixRX<F> initWff() {
        MatrixRX<F> wff = MatrixRX<F>::Zero(NBNEUR,FFRFSIZE);
        wff = (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixRX<F>::Random(NBNEUR,FFRFSIZE).cwiseAbs().array()).cwiseMin(MAXW);
        wff.bottomRows(NBI).setZero();
        return wff;
    }

    static VectorX<F> initV() {
        return VectorX<F>::Constant(NBNEUR,IZHREST);
    }

    static VectorX<F> initVThresh() {
        return VectorX<F>::Constant(NBNEUR,VTREST);
    }

    static VectorX<F> initVLongtrace() {
        VectorX<F> v = initV();
        return (v.array() - THETAVLONGTRACE).cwiseMax(0);
    }

    static VectorX<F> initVPos() {
        return initV();
    }

    static VectorX<F> initVNeg() {
        return initV();
    }

    static VectorX<F> initXPlastFF() {
        return VectorX<F>::Zero(FFRFSIZE);
    }

    static VectorX<F> initXPlastLat() {
        return VectorX<F>::Zero(NBNEUR);
    }

    static VectorX<F> initWadap() {
        return VectorX<F>::Zero(NBNEUR);
    }

    static VectorX<F> initZ() {
        return VectorX<F>::Zero(NBNEUR);
    }

    static MutableState<F,I> initMutableState() {
        MutableState<F,I> mutableState;

        mutableState.w = Init<F,I>::initW();
        mutableState.wff = Init<F,I>::initWff();

        mutableState.vthresh = Init<F,I>::initVThresh();
        mutableState.vlongtrace = Init<F,I>::initVLongtrace();
        mutableState.vpos = Init<F,I>::initVPos();
        mutableState.vneg = Init<F,I>::initVNeg();

        mutableState.xplastLat = Init<F,I>::initXPlastLat();
        mutableState.xplastFF = Init<F,I>::initXPlastFF();

        mutableState.wadap = Init<F,I>::initWadap();
        mutableState.z = Init<F,I>::initZ();

        return mutableState;
    }

    /* Static State */
    static MatrixRX<I> initDelays() {
        MatrixRX<I> delays = MatrixRX<I>::Zero(NBNEUR,NBNEUR);

        //TODO

        return delays;
    }

    static VectorX<F> initALTDS() {
        //BASEALTD
        //RANDALTD
        //RAND_MAX
        //TODO
        return VectorX<F>::Zero(NBNEUR);
    }

    static StaticState<F,I> initStaticState() {
        StaticState<F,I> staticState;

        staticState.input = Dataset<F>::retrieveTransformedDataset();
        staticState.delays = Init<F,I>::initDelays();
        staticState.altds = Init<F,I>::initALTDS();

        return staticState;
    }

    /* Buffers */
    static MatrixRX<I> initIncomingSpikes() {
        return MatrixRX<I>::Zero(NBNEUR,NBNEUR);
    }

    static VectorX<I> initFirings() {
        return VectorX<I>::Zero(NBNEUR);
    }

    static MatrixRX<I> initLgnFiringsBuffer() {
        return MatrixRX<I>::Zero(NBSTEPSSTIM,FFRFSIZE);
    }

    static MatrixRX<I> initPoissonNoiseBuffer() {
        return MatrixRX<I>::Zero(NBSTEPSPERPRES,NBNEUR);
    }

    static VectorX<F> initNeuronInputsBuffer() {
        return VectorX<F>::Zero(NBNEUR);
    }

    static VectorX<F> initEachNeurLTD() {
        return VectorX<F>::Zero(NBNEUR);
    }

    static VectorX<F> initEachNeurLTP() {
        return VectorX<F>::Zero(NBNEUR);
    }

    static Buffers<F,I> initBuffers() {
        Buffers<F,I> buffers;

        buffers.incomingSpikes = Init<F,I>::initIncomingSpikes();
        buffers.firings = Init<F,I>::initFirings();

        buffers.lgnfirings = Init<F,I>::initLgnFiringsBuffer();
        buffers.poissonNoise = Init<F,I>::initPoissonNoiseBuffer();
        buffers.neuronInputs = Init<F,I>::initNeuronInputsBuffer();

        buffers.eachNeurLTD = Init<F,I>::initEachNeurLTD();
        buffers.eachNeurLTP = Init<F,I>::initEachNeurLTP();

        return buffers;
    }

private:
    BOOST_STATIC_ASSERT(is_same<float,F>::value || is_same<double,F>::value);
    BOOST_STATIC_ASSERT(is_same<int,I>::value || is_same<long,I>::value);
};
