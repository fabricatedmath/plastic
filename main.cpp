#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "test.h"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    cout << "dogs" << endl;
    MatrixXf m = MatrixXf::Identity(5,5);
    VectorXf v = VectorXf::Ones(5);
    Test test;
    test.m = m;
    test.v = v;
    cout << v << endl;
    cout << "dogs" << endl;
    cout << test.v << endl;
    wrapper(test);
}
