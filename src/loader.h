#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "eigen_boost_serialization.hpp"

using namespace std;
using namespace Eigen;

template<typename T>
using VectorX = Matrix<T, Dynamic, 1>;

template<typename T>
using MatrixRX = Matrix<T, Dynamic, Dynamic, RowMajor>;

template<typename T>
using MatrixX = Matrix<T, Dynamic, Dynamic, ColMajor>;

template<typename T>
void storeVector(string filenamePrefix, VectorX<T> vector) {
    string filename = filenamePrefix.append(".").append(typeid(T).name());
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);
    oa << vector;
}

template<typename T>
VectorX<T> loadVector(string filenamePrefix) {
    string filename = filenamePrefix.append(".").append(typeid(T).name());
    VectorX<T> vector;
    std::ifstream ifs(filename);
    boost::archive::binary_iarchive ia(ifs);
    ia & vector;
    return vector;
}

template<typename T>
void storeMatrix(string filenamePrefix, MatrixRX<T> matrix) {
    string filename = filenamePrefix.append(".").append(typeid(T).name());
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);
    oa << matrix;
}

template<typename T>
MatrixRX<T> loadMatrix(string filenamePrefix) {
    string filename = filenamePrefix.append(".").append(typeid(T).name());
    MatrixRX<T> matrix;
    std::ifstream ifs(filename);
    boost::archive::binary_iarchive ia(ifs);
    ia & matrix;
    return matrix;
}

template<typename T>
void storeMatrix(string filenamePrefix, MatrixX<T> matrix) {
    string filename = filenamePrefix.append(".").append(typeid(T).name());
    MatrixRX<T> matrix2 = matrix;
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);
    oa << matrix2;
}
