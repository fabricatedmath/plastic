#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "type.h"
#include "test.h"
#include "init.h"
#include "dataset.h"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    MatrixXf m = MatrixXf::Random(50,50);
    VectorXf v = VectorXf::Ones(5);
    Test test;
    test.m = m;
    test.v = v;

    MatrixW<float> w = Init<float>::initW();
    MatrixWff<float> wff = Init<float>::initWff();

    MatrixTransformedDataset<float> transformedDataset =
        Dataset<float>::retrieveTransformedDataset();
    cout << transformedDataset.row(0).head(10) << endl;


    wrapper2();
}
