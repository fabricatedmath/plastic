#pragma once

template<typename T>
using MatrixRX = Matrix<T, Dynamic, Dynamic, RowMajor>;

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
