#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"
#include "err.cuh"
#include "randGen.cuh"
#include "cuda_state.cuh"
#include "helper_cuda.h"
#include "cuda_utility.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

using namespace std;
using namespace Eigen;
namespace cg = cooperative_groups;

const int rows = 50;

__global__ void test_kernel(CudaMutableState cudaMutableState, unsigned long long* time) {
    unsigned long long startTime = clock64();
    unsigned long long hz = 2100000000;
    unsigned long long seconds = 5*2;
    unsigned long long thresh = hz * seconds;
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        //        printf("%ul\n", thresh);
        //printf("%ul\n", thresh2);
    }
    do {
        //        printf("%d\n", idx);
//    float w = cudaMutableState.w.data[threadIdx.x];
//    cudaMutableState.w.data[threadIdx.x] = w+1;
            /*
            for (int row = 0; row < rows; row++) {
                float* rowPtr = getRowPtr(cudaMutableState.w, row);
                float i = rowPtr[threadIdx.x];
                rowPtr[threadIdx.x] = i+1;
                //printf("%d",threadIdx.x);
            }
            */
        //        cg::sync(grid);
    } while ((clock64() - startTime) < thresh);
    unsigned long long endTime = clock64();
    if (idx == 0) {
        *time = (endTime - startTime);
    }
    
}

void wrapper2() {
    typedef RandomGen<curandState,1,2> Rgen;
    Rgen d_rgen(1.1,1.8);
    //Rgen* d_rgen = new Rgen(1.8,1.1);
}

void something() {
    char array[] = "dogs";
    char *p = array;
    const char *arg = p;
    const char *argv[] = {"dogs"};
    
    int argc = 0;
    int device = findCudaDevice(argc, argv);
    cudaDeviceProp prop = { 0 };
    cout << gpuGetMaxGflopsDeviceId() << endl;
    gpuErrchk( cudaSetDevice(device) );
    gpuErrchk( cudaGetDeviceProperties(&prop, device) );
    cout << prop.multiProcessorCount << endl;
    cout << prop.maxThreadsPerMultiProcessor << endl;
    cout << prop.maxThreadsPerBlock << endl;
    cout << prop.clockRate << endl;
    cout << endl;

    int numThreads = 120; //prop.maxThreadsPerMultiProcessor;
    int maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    int numSms = prop.multiProcessorCount;
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, test_kernel, numThreads, 0));
    cout << numBlocksPerSm << endl;
    cout << numBlocksPerSm * numSms << endl;
}

void wrapper(MutableState mutableState, StaticState staticState) {
    something();
//    gpuErrchk( cudaOccupancyMaxActiveBlocksPerMultiprocessor( );
    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    
    CudaMutableState cudaMutableState;
    gpuErrchk( cudaMalloc(&mutableState,&cudaMutableState) );
    gpuErrchk( memcpyHostToDevice(&mutableState,&cudaMutableState) );

    CudaStaticState cudaStaticState;
    gpuErrchk( cudaMalloc(&staticState,&cudaStaticState) );
    gpuErrchk( memcpyHostToDevice(&staticState,&cudaStaticState) );

    void *kernelArgs[] = {
        (void*)&cudaMutableState,
        (void*)&d_time
    };

    int numBlocks = 544;
    int numThreads = 120;

    dim3 dimBlock(numThreads,1,1);
    dim3 dimGrid(numBlocks,1,1);

    const int smemSize = 0;
    for (int i = 0; i < 10; i++) {
        gpuErrchk( cudaLaunchCooperativeKernel((void*)test_kernel, dimGrid, dimBlock, kernelArgs, smemSize, NULL) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( memcpyDeviceToHost(&mutableState, &cudaMutableState) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << time << endl;
    }
}


