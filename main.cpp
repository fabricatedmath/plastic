#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "state.h"
#include "test.h"
#include "init.h"
#include "dataset.h"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    MatrixXf m = MatrixXf::Random(50,50);
    VectorXf v = VectorXf::Ones(5);

    MatrixW<float> w = Init<float>::initW();
    MatrixWff<float> wff = Init<float>::initWff();

    MutableState mutableState;
    mutableState.w = w;
    mutableState.wff = wff;

    MatrixTransformedDataset<float> transformedDataset =
        Dataset<float>::retrieveTransformedDataset();
    cout << transformedDataset.row(0).head(10) << endl;

    wrapper2();
}
