#pragma once

#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixRXf;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatrixRXi;

template<typename T>
using MatrixRX = Matrix<T, Dynamic, Dynamic, RowMajor>;

template<typename T>
using VectorX = Matrix<T, Dynamic, 1>;

template<typename F, typename I>
struct MutableState {
    MatrixRX<F> w;
    MatrixRX<F> wff;
    MatrixRX<I> incomingSpikes;
    VectorX<I> firings;

    VectorX<F> v;
    VectorX<F> vprev;
    VectorX<F> vthresh;
    VectorX<F> vlongtrace;
    VectorX<F> vpos;
    VectorX<F> vneg;

    VectorX<F> xplastLat;
    VectorX<F> xplastFF;

    VectorX<F> wadap;
    VectorX<F> z;

    VectorX<I> isSpiking;
};

template<typename F, typename I>
struct StaticState {
    MatrixRX<F> input;
    MatrixRX<I> delays;
    VectorX<F> altds;
};

template<typename F, typename I>
struct Buffers {
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
