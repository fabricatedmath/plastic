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

void wrapper(Test test);
