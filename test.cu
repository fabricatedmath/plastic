#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"

using namespace std;
using namespace Eigen;

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    cout << x << endl; \
    exit(-1); \
    }} while(0)

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

__global__ void test_kernel(CudaTest test, unsigned long long* time) {
    unsigned long long startTime = clock();
    float v = test.v.data[threadIdx.x];
    printf("dogs2:%f\n",v);
    test.v.data[threadIdx.x] = v+1;
    unsigned long long endTime = clock();
    *time = (endTime - startTime);
}

__global__ void test_kernel2(unsigned long long* time) {
    unsigned long long startTime = clock();
    printf("dogs2\n");
    unsigned long long endTime = clock();
    *time = (endTime - startTime);
}
/*
__global__ void test_kernel2(flotest) {
    float v = test.v.data[0];
    printf("dogs2\n%d",v);
    }*/

cudaError_t cudaMalloc(VectorXf* v, CudaVectorXf* cv) {
    cout << v->size() << endl;
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(float));
}

cudaError_t memcpyHostToDevice(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(float), cudaMemcpyDeviceToHost);
}

void cudaMalloc(MatrixXf m, CudaMatrixXf cm) {
    cudaMallocPitch((void**)&cm.data, &cm.pitch, m.cols() * sizeof(float), m.rows());
}

void wrapper(Test test) {
    cout << "cats" << endl;
    //cout << test.v << endl;
    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    CudaTest cudaTest;
    CudaVectorXf cv;
    gpuErrchk( cudaMalloc(&test.v, &cudaTest.v) );
    gpuErrchk( memcpyHostToDevice(&test.v, &cudaTest.v) );
    //gpuErrchk( cudaMalloc((void**)&cv.data, 5 * sizeof(float)) );
    test_kernel<<<1,5>>>(cudaTest, d_time);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( memcpyDeviceToHost(&test.v, &cudaTest.v) );
    cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cout << time << endl;
    cout << test.v << endl;
    
//    float* data = (float*)malloc(5*sizeof(float));
//    cudaMemcpy(data, cudaTest.v.data, 5 * sizeof(float), cudaMemcpyDeviceToHost);
//    cout << data[0] << endl;
}
