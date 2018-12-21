#pragma once
#include <Eigen/Dense>

MatrixXd initW(int numNeurons) {
    return MatrixXd::Random(numNeurons, numNeurons);
}
