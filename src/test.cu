#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"
#include "err.cuh"
#include "randGen.cuh"
#include "constants.h"
#include "cuda_state.cuh"
#include "helper_cuda.h"
#include "kernels/fast_kernel/kernel.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

using namespace std;
using namespace Eigen;
namespace cg = cooperative_groups;

const int rows = 50;
const int numThreadsConstant = 128;

__global__ void spin_kernel() {
    unsigned long long startTime = clock64();
    unsigned long long hz = 2100000000;
    unsigned long long seconds = 100;
    unsigned long long thresh = hz * seconds;
    do {
    } while ((clock64() - startTime) < thresh);
}

void printSetBlockGridStats(int* thisNumBlocks, int* thisNumThreads) {
    const char *argv[] = {""};
    
    int argc = 0;
    int device = findCudaDevice(argc, argv);
    cudaDeviceProp prop = { 0 };
    //    cout << gpuGetMaxGflopsDeviceId() << endl;
    gpuErrchk( cudaSetDevice(device) );
    gpuErrchk( cudaGetDeviceProperties(&prop, device) );

//    int numThreads = 128;
    int maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    int numSms = prop.multiProcessorCount;
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, test_kernel<numThreadsConstant>, numThreadsConstant, 0));
    int numBlocks = numBlocksPerSm * numSms;
    
    cout << "--------Grid/Block Statistics-----------------------------------------" << endl;
    cout << endl;
    cout << "\t\tGlobally" << endl;
    printf("\tSMs:\t\t\t%d\n", numSms);
    printf("\tMax Blocks:\t\t%d\n", maxBlocks);
    cout << endl;
    cout << "\t\tProgram" << endl;
    printf("\tBlocks per SM:\t\t%d\n", numBlocksPerSm);
    printf("\tBlocks:\t\t\t%d\n", numBlocks);
    printf("\tThreads per Block:\t%d\n", numThreadsConstant);
    printf("\tThreads:\t\t%d\n", numThreadsConstant * numBlocks);
    cout << endl;
    cout << "----------------------------------------------------------------------" << endl;

    *thisNumBlocks = numBlocks;
    *thisNumThreads = numThreadsConstant;
}

void wrapper(MutableState mutableState, StaticState staticState, Buffers buffers) {
    int numBlocks;
    int numThreads;
    
    printSetBlockGridStats(&numBlocks,&numThreads);

    size_t usedBeforeAllocation; 
    {
        size_t freeBytes;
        size_t totalBytes;
        size_t usedBytes;
        gpuErrchk( cudaMemGetInfo(&freeBytes,&totalBytes) );
        usedBytes = totalBytes - freeBytes;
        usedBeforeAllocation = usedBytes;
        cout << "--------Memory Pre-Allocation-----------------------------------------" << endl;
        cout << endl;
        printf("\tfree:\t\t%lu MB\n", freeBytes / (1024*1024));
        printf("\tused:\t\t%lu MB\n", usedBytes / (1024*1024));
        printf("\ttotal:\t\t%lu MB\n", totalBytes / (1024*1024));
        cout << endl;
        cout << "----------------------------------------------------------------------" << endl;
    }

    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    
    CudaMutableState cudaMutableState;
    gpuErrchk( cudaMalloc(&mutableState,&cudaMutableState) );
    gpuErrchk( memcpyHostToDevice(&mutableState,&cudaMutableState) );

    CudaStaticState cudaStaticState;
    gpuErrchk( cudaMalloc(&staticState,&cudaStaticState) );
    gpuErrchk( memcpyHostToDevice(&staticState,&cudaStaticState) );

    CudaBuffers cudaBuffers;
    gpuErrchk( cudaMalloc(&buffers,&cudaBuffers) );
    gpuErrchk( memcpyHostToDevice(&buffers,&cudaBuffers) );

    numBlocks = min(NBNEUR,numBlocks);
    cout << "num blocks: " << numBlocks << endl;
    typedef RandomGen<curandState> Rgen;
    Rgen cudaRgen(numBlocks, numThreads, 1.1, 1.8);

    size_t usedAfterAllocation;
    size_t allocated;
    {
        size_t freeBytes;
        size_t totalBytes;
        size_t usedBytes;
        gpuErrchk( cudaMemGetInfo(&freeBytes,&totalBytes) );
        usedBytes = totalBytes - freeBytes;
        usedAfterAllocation = usedBytes;
        allocated = usedAfterAllocation - usedBeforeAllocation;
        cout << "--------Memory Post-Allocation----------------------------------------" << endl;
        cout << endl;
        printf("\tfree:\t\t%lu MB\n", freeBytes / (1024*1024));
        printf("\tused:\t\t%lu MB\n", usedBytes / (1024*1024));
        printf("\ttotal:\t\t%lu MB\n", totalBytes / (1024*1024));
        cout << endl;
        printf("\tallocated:\t%lu MB\n", allocated / (1024*1024));
        cout << endl;
        cout << "----------------------------------------------------------------------" << endl;
    }

    void *kernelArgs[] = {
        (void*)&cudaMutableState,
        (void*)&cudaStaticState,
        (void*)&cudaBuffers,
        (void*)&cudaRgen,
        (void*)&d_time
    };

    void *spinKernelArgs[] = {
    };

    const dim3 dimBlock(numThreads,1,1);
    const dim3 dimGrid(numBlocks,1,1);
    
    const int smemSize = 0;
    for (int i = 0; i < 10; i++) {
        gpuErrchk( cudaLaunchCooperativeKernel((void*)(test_kernel<numThreadsConstant>),  dimGrid, dimBlock, kernelArgs, smemSize, NULL) );
        //gpuErrchk( cudaLaunchCooperativeKernel((void*)spin_kernel, dimGrid, dimBlock, spinKernelArgs, smemSize, NULL) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( memcpyDeviceToHost(&mutableState, &cudaMutableState) );
        gpuErrchk( memcpyDeviceToHost(&buffers, &cudaBuffers) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << time << endl;
/*
        for (int row = 0; row < buffers.lgnfirings.rows(); row++) {
            cout << buffers.lgnfirings.row(row) << endl;
        }
        */
    }
}


