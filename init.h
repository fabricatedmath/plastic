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

    typedef Matrix<FT,Dynamic,Dynamic,RowMajor> MatrixWff;
    static MatrixWff initWff() {
        MatrixWff wff = MatrixWff::Zero(NBNEUR,FFRFSIZE);
        wff = (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixWff::Random(NBNEUR,FFRFSIZE).cwiseAbs().array()).cwiseMin(MAXW);
        wff.bottomRows(NBI).setZero();
        return wff;
    }

    typedef Matrix<IT,Dynamic,Dynamic,RowMajor> MatrixIncomingSpikes;
    static MatrixIncomingSpikes initIncomingSpikes() {
        return MatrixIncomingSpikes::Zero(NBNEUR,NBNEUR);
    }

    typedef Matrix<IT,1,NBNEUR,RowMajor> VectorFirings;
    static VectorFirings initFirings() {
        return VectorFirings::Zero();
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorV;
    static VectorV initV() {
        return VectorV::Constant(IZHREST);
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorVPrev;
    static VectorVPrev initVPrev() {
        return initV();
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorVThresh;
    static VectorVThresh initVThresh() {
        return VectorVThresh::Constant(VTREST);
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorVLongtrace;
    static VectorVLongtrace initVLongtrace() {
        VectorV v = initV();
        return (v.array() - THETAVLONGTRACE).cwiseMax(0);
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorVPos;
    static VectorVPos initVPos() {
        return initV();
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorVNeg;
    static VectorVNeg initVNeg() {
        return initV();
    }

    typedef Matrix<FT,1,FFRFSIZE,RowMajor> VectorXPlastFF;
    static VectorXPlastFF initXPlastFF() {
        return VectorXPlastFF::Zero();
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorXPlastLat;
    static VectorXPlastLat initXPlastLat() {
        return VectorXPlastLat::Zero();
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorWadap;
    static VectorWadap initWadap() {
        return VectorWadap::Zero();
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorZ;
    static VectorZ initZ() {
        return VectorZ::Zero();
    }

    typedef Matrix<IT,1,NBNEUR,RowMajor> VectorIsSpiking;
    static VectorIsSpiking initIsSpiking() {
        return VectorIsSpiking::Zero();
    }

    /* Static State */
    typedef Matrix<IT,Dynamic,Dynamic,RowMajor> MatrixDelays;
    static MatrixDelays initDelays() {
        MatrixDelays delays = MatrixDelays::Zero(NBNEUR,NBNEUR);

        //TODO

        return delays;
    }

    typedef Matrix<FT,1,Dynamic,RowMajor> VectorALTDS;
    static VectorALTDS initALTDS() {
        //BASEALTD
        //RANDALTD
        //RAND_MAX
        //TODO
        return VectorALTDS::Zero();
    }

    /* Buffers */
    typedef Matrix<FT,Dynamic,Dynamic,RowMajor> MatrixLgnFiringsBuffer;
    static MatrixLgnFiringsBuffer initLgnFiringsBuffer() {
        MatrixLgnFiringsBuffer lgnfiringsBuffer =
            MatrixLgnFiringsBuffer::Zero(NBSTEPSSTIM,FFRFSIZE);
        return lgnfiringsBuffer;
    }

    typedef Matrix<FT,1,NBNEUR,RowMajor> VectorNeuronInputsBuffer;
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
