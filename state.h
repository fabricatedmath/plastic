#pragma once

#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixRXf;

struct MutableState {
    MatrixRXf w;
    MatrixRXf wff;
};

struct StaticState {
    MatrixRXf input;
};

struct Buffers {

};
