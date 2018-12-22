#include <iostream>
#include <Eigen/Dense>
#include "eigen_boost_serialization.hpp"
#include "type.h"
#include "test.h"
#include "inits.h"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[]) {
    MatrixXf m = MatrixXf::Random(50,50);
    VectorXf v = VectorXf::Ones(5);
    Test test;
    test.m = m;
    test.v = v;
    //cout << v << endl;
    //cout << test.v << endl;
    //cout << test.m << endl;
    cout << initW() << endl;
    cout << endl << endl;
    cout << initWff() << endl;

    MatrixTransformedDataset transformedDataset = retrieveTransformedDataset();
    cout << transformedDataset.row(0).head(10) << endl;


}
