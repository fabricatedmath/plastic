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

template<typename T>
class Init {
public:
    typedef Matrix<T,NBNEUR,NBNEUR,RowMajor> MatrixW;
    typedef Matrix<T,NBNEUR,FFRFSIZE,RowMajor> MatrixWff;
    typedef Matrix<T,Dynamic,Dynamic,RowMajor> MatrixLgnFiringsBuffer;

    static MatrixW initW() {
        MatrixW w = MatrixW::Zero(); //MatrixXf::Zero(NBNEUR, NBNEUR);
        w.bottomRows(NBI).leftCols(NBE).setRandom(); // Inhbitory neurons receive excitatory inputs from excitatory neurons
        w.rightCols(NBI).setRandom(); // Everybody receives inhibition (including inhibitory neurons)
        w.bottomRows(NBI).rightCols(NBI) =  -w.bottomRows(NBI).rightCols(NBI).cwiseAbs() * WII_MAX;
        w.topRows(NBE).rightCols(NBI) = -w.topRows(NBE).rightCols(NBI).cwiseAbs() * WIE_MAX;
        w.bottomRows(NBI).leftCols(NBE) = w.bottomRows(NBI).leftCols(NBE).cwiseAbs() * WEI_MAX;
        w = w - w.cwiseProduct(MatrixW::Identity()); // Diagonal lateral weights are 0 (no autapses !)
        return w;
    }

    static MatrixWff initWff() {
        MatrixWff wff = MatrixWff::Zero();
        wff = (WFFINITMIN + (WFFINITMAX-WFFINITMIN) * MatrixWff::Random().cwiseAbs().array()).cwiseMin(MAXW);
        wff.bottomRows(NBI).setZero();
        return wff;
    }

    static MatrixLgnFiringsBuffer initLgnFiringsBuffer() {
        MatrixLgnFiringsBuffer lgnfiringsBuffer =
            MatrixLgnFiringsBuffer::Zero(NBSTEPSSTIM,FFRFSIZE);
        return lgnfiringsBuffer;
    }

private:
    BOOST_STATIC_ASSERT(is_same<float,T>::value || is_same<double,T>::value);
};

template <typename T>
using MatrixW = typename Init<T>::MatrixW;

template <typename T>
using MatrixWff = typename Init<T>::MatrixWff;

template <typename T>
using MatrixLgnFiringsBuffer = typename Init<T>::MatrixLgnFiringsBuffer;
