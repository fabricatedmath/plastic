#pragma once

#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixRXf;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatrixRXi;

struct MutableState {
    MatrixRXf w;
    MatrixRXf wff;
    MatrixRXi incomingSpikes;
    VectorXi firings;
};

struct StaticState {
    MatrixRXf input;
    MatrixRXi delays;
};

struct Buffers {
    MatrixRXf lgnfirings;
    VectorXf neuronInputs;
};
