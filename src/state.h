#pragma once

#include "eigen.h"

template<typename F, typename I>
struct MutableState {
    MatrixRX<F> w;
    MatrixRX<F> wff;

    VectorX<F> vthresh;
    VectorX<F> vlongtrace;
    VectorX<F> vpos;
    VectorX<F> vneg;

    VectorX<F> xplastLat;
    VectorX<F> xplastFF;

    VectorX<F> wadap;
    VectorX<F> z;
};

template<typename F, typename I>
struct StaticState {
    MatrixRX<F> input;
    MatrixRX<I> delays;
    VectorX<F> altds;
};

template<typename F, typename I>
struct Buffers {
    MatrixRX<I> incomingSpikes;
    VectorX<I> firings;

    MatrixRX<I> lgnfirings;
    MatrixRX<I> poissonNoise;
    VectorX<F> neuronInputs;

    VectorX<F> eachNeurLTD;
    VectorX<F> eachNeurLTP;
};

template<typename F>
struct RandomHistorical {
    MatrixRX<F> uniformMatrix;
    MatrixRX<unsigned int> posPoissonMatrix;
    MatrixRX<unsigned int> negPoissonMatrix;
};
