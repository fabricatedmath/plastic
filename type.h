#pragma once

using namespace Eigen;

typedef Matrix<int8_t, Dynamic, Dynamic> MatrixXu;
typedef Matrix<int8_t, Dynamic, Dynamic, RowMajor> MatrixRXu;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatrixRXi;
typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixRXd;
typedef Matrix<int8_t, 1, Dynamic> VectorXu;

struct CudaMatrixXf {
    float* data;
    size_t pitch;
};

struct CudaVectorXf {
    float* data;
};

struct CudaTest {
    CudaMatrixXf m;
    CudaVectorXf v;
};

struct CudaStaticState {
    CudaMatrixXf images;
};

struct CudaBuffers {
    CudaMatrixXf buf;
};

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
