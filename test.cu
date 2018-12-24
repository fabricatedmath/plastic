#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"
#include "err.cuh"
#include "randGen.cuh"
#include "cuda_state.cuh"

using namespace std;
using namespace Eigen;

const int rows = 50;

__global__ void test_kernel(CudaMutableState cudaMutableState, unsigned long long* time) {
    unsigned long long startTime = clock();
//    float w = cudaMutableState.w.data[threadIdx.x];
//    cudaMutableState.w.data[threadIdx.x] = w+1;
    for (int row = 0; row < rows; row++) {
        float* rowPtr = getRowPtr(cudaMutableState.w, row);
        float i = rowPtr[threadIdx.x];
        rowPtr[threadIdx.x] = i+1;
    }
    unsigned long long endTime = clock();
    *time = (endTime - startTime);
}

void wrapper2() {
    typedef RandomGen<curandState,1,2> Rgen;
    Rgen d_rgen(1.1,1.8);
    //Rgen* d_rgen = new Rgen(1.8,1.1);
}

void wrapper(MutableState mutableState, StaticState staticState) {
    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    CudaMutableState cudaMutableState;
    gpuErrchk( cudaMalloc(&mutableState,&cudaMutableState) );
    gpuErrchk( memcpyHostToDevice(&mutableState,&cudaMutableState) );

    for (int i = 0; i < 10; i++) {
        test_kernel<<<1,5>>>(cudaMutableState, d_time);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( memcpyDeviceToHost(&mutableState, &cudaMutableState) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << time << endl;
    }
}
