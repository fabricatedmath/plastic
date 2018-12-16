#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"
#include "err.cuh"
#include "mem.cuh"

using namespace std;
using namespace Eigen;

__global__ void test_kernel(CudaTest test, unsigned long long* time) {
    unsigned long long startTime = clock();
    float v = test.v.data[threadIdx.x];
    test.v.data[threadIdx.x] = v+1;
    for (int row = 0; row < 5; row++) {
        float* rowPtr = getRowPtr(test.m, row);
        float i = rowPtr[threadIdx.x];
        rowPtr[threadIdx.x] = i+1;
    }
    unsigned long long endTime = clock();
    *time = (endTime - startTime);
}

void wrapper(Test test) {
    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    CudaTest cudaTest;
    gpuErrchk( cudaMalloc(&test,&cudaTest) );
    gpuErrchk( memcpyHostToDevice(&test,&cudaTest) );

    for (int i = 0; i < 10; i++) {
        test_kernel<<<1,5>>>(cudaTest, d_time);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( memcpyDeviceToHost(&test, &cudaTest) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << time << endl;
        cout << test.v << endl;
        cout << test.m << endl;
    }
}
