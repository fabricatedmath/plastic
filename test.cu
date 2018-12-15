#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"

using namespace std;
using namespace Eigen;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

__device__ float* getRowPtr(CudaMatrixXf,int);

__global__ void test_kernel(CudaTest test, unsigned long long* time) {
    unsigned long long startTime = clock();
    float v = test.v.data[threadIdx.x];
    test.v.data[threadIdx.x] = v+1;
    for (int row = 0; row < 5; row++) {
        float* rowPtr = getRowPtr(test.m, row);
        float i = rowPtr[threadIdx.x];
//        printf("%d:%f:%d\n",row, i, threadIdx.x);
        rowPtr[threadIdx.x] = i+1;
    }
    unsigned long long endTime = clock();
    *time = (endTime - startTime);
}

cudaError_t cudaMalloc(VectorXf* v, CudaVectorXf* cv) {
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(float));
}

cudaError_t memcpyHostToDevice(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(float), cudaMemcpyDeviceToHost);
}

cudaError_t cudaMalloc(MatrixXf* m, CudaMatrixXf* cm) {
   return cudaMallocPitch((void**)&cm->data, &cm->pitch, m->cols() * sizeof(float), m->rows());
}

cudaError_t memcpyHostToDevice(MatrixXf* m, CudaMatrixXf* cm) {
    return cudaMemcpy2D(cm->data, cm->pitch, m->data(), m->cols() * sizeof(float), m->cols() * sizeof(float), m->rows(), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(MatrixXf* m, CudaMatrixXf* cm) {
    return cudaMemcpy2D(m->data(), m->cols() * sizeof(float), cm->data, cm->pitch, m->cols() * sizeof(float), m->rows(), cudaMemcpyDeviceToHost);
}

float* getRowPtr(CudaMatrixXf cm, int row) {
    return (float*)((char*)cm.data + row*cm.pitch);
}

void wrapper(Test test) {
    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    CudaTest cudaTest;
    gpuErrchk( cudaMalloc(&test.v, &cudaTest.v) );
    gpuErrchk( memcpyHostToDevice(&test.v, &cudaTest.v) );
    gpuErrchk( cudaMalloc(&test.m, &cudaTest.m) );
    gpuErrchk( memcpyHostToDevice(&test.m, &cudaTest.m) );
    for (int i = 0; i < 1; i++) {
        test_kernel<<<1,5>>>(cudaTest, d_time);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( memcpyDeviceToHost(&test.v, &cudaTest.v) );
        gpuErrchk( memcpyDeviceToHost(&test.m, &cudaTest.m) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << time << endl;
        cout << test.v << endl;
        cout << test.m << endl;
    }
}
