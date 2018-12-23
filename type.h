#pragma once

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixRXf;

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
    MatrixRXf m;
    VectorXf v;
};

struct StaticState {
    MatrixRXf images;
};

struct Buffers {
    MatrixRXf buf;
};
