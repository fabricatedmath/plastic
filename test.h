#pragma once
#include <Eigen/Dense>

using namespace Eigen;

struct Test {
    MatrixXf m;
    VectorXf v;
};

void wrapper(Test test);
