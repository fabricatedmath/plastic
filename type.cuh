#pragma once

const int rows = 50;

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
