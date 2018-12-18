#pragma once

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
