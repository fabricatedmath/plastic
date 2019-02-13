#include <iostream>
#include <vector>
#include <cooperative_groups.h>

#include "eigen.h"
#include "err.cuh"
#include "cuda_eigen.cuh"

namespace cg = cooperative_groups;
using namespace std;

struct State {
    VectorX<float> f;
};

struct CudaState {
    CudaVectorX<float> f;
};

cudaError_t cudaMalloc(State* s, CudaState* cs) {
    errRet( cudaMalloc(&s->f,&cs->f) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(State* s, CudaState* cs) {
    errRet( memcpyHostToDevice(&s->f,&cs->f) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(State* s, CudaState* cs) {
    errRet( memcpyDeviceToHost(&s->f,&cs->f) );
    return cudaSuccess;
}

__global__ void test(int* const __restrict__ d1,
                     int* const __restrict__ d2) {
    cg::multi_grid_group myMultiGrid = cg::this_multi_grid();
    cg::grid_group myGrid = cg::this_grid();
    cg::thread_block myBlock = cg::this_thread_block();

    if (myGrid.thread_rank() == 0) {
        printf("%d\n", myMultiGrid.grid_rank());
    }

    cg::sync(myMultiGrid);

    if (myMultiGrid.thread_rank() == 0) {
        printf("%d : %d \n", *d1, *d2);
    }

    cg::sync(myMultiGrid);

    if(myMultiGrid.grid_rank() == 0) {
        *d1 = 1;
    }

    if(myMultiGrid.grid_rank() == 1) {
        *d2 = 1;
    }
    
    cg::sync(myMultiGrid);

    if (myMultiGrid.thread_rank() == 0) {
        printf("%d : %d \n", *d1, *d2);
    }
}

const int numDevices = 2;

void call() {

    std::vector<int*> thisD(numDevices);

    gpuErrchk( cudaSetDevice(0) );
    gpuErrchk( cudaDeviceEnablePeerAccess(1,0) );

    gpuErrchk( cudaSetDevice(1) );
    gpuErrchk( cudaDeviceEnablePeerAccess(0,0) );

    for (int i = 0; i < numDevices; i++) {
        gpuErrchk( cudaSetDevice(i) );
        gpuErrchk( cudaMalloc((void **)&thisD[i], numDevices*sizeof(int)) );
        gpuErrchk( cudaMemset(thisD[i], 0, numDevices*sizeof(int)) );
    }

    const dim3 dimBlock(32,1,1);
    const dim3 dimGrid(1,1,1);
    const int smemSize = 0;
    cudaLaunchParams launchParamsList[numDevices];
    for (int i = 0; i < numDevices; i++) {
        gpuErrchk( cudaSetDevice(i) );
        launchParamsList[i].func      = (void *)test;
        launchParamsList[i].blockDim  = dimBlock;
        launchParamsList[i].gridDim   = dimGrid;
        launchParamsList[i].args      = (void **)malloc(2 * sizeof(void *));
        {
            launchParamsList[i].args[0] = &thisD[0];
            launchParamsList[i].args[1] = &thisD[1];
        }
        launchParamsList[i].sharedMem = smemSize;

        gpuErrchk( cudaStreamCreate(&launchParamsList[i].stream) );
    }

    gpuErrchk( cudaLaunchCooperativeKernelMultiDevice(launchParamsList, 2, cudaCooperativeLaunchMultiDeviceNoPreSync |
          cudaCooperativeLaunchMultiDeviceNoPostSync) );
    
    for (int i = 0; i < numDevices; i++) {
        gpuErrchk( cudaSetDevice(i) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
}

