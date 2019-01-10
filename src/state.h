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

    VectorXf v;
    VectorXf vprev;
    VectorXf vthresh;
    VectorXf vlongtrace;
    VectorXf vpos;
    VectorXf vneg;

    VectorXf xplastLat;
    VectorXf xplastFF;

    VectorXf wadap;
    VectorXf z;

    VectorXi isSpiking;
};

struct StaticState {
    MatrixRXf input;
    MatrixRXi delays;
    VectorXf altds;
};

struct Buffers {
    MatrixRXi lgnfirings;
    MatrixRXi poissonNoise;
    VectorXf neuronInputs;

    VectorXf eachNeurLTD;
    VectorXf eachNeurLTP;
};
