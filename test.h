#pragma once
#include <Eigen/Dense>

using namespace Eigen;

struct Test {
    MatrixXf m;
    VectorXf v;
};

struct StaticState {
    MatrixXf images;
};

struct Buffers {
    MatrixXf buf;
};

void wrapper(Test test);
