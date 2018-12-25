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

typedef RandomGen<curandState> Rgen;

__global__ void spin_kernel() {
    unsigned long long startTime = clock64();
    unsigned long long hz = 2100000000;
    unsigned long long seconds = 100;
    unsigned long long thresh = hz * seconds;
    do {
    } while ((clock64() - startTime) < thresh);
}

__global__ void test_kernel(CudaMutableState cudaMutableState,
                            CudaStaticState cudaStaticState,
                            CudaBuffers buffers,
                            Rgen rgen,
                            unsigned long long* time) {
    
    unsigned long long startTime = clock64();
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int inputRow = 0;
    float* rowPtr = getRowPtr(cudaStaticState.input, inputRow);
    if (id < 10) {
        curandState g = rgen.get(id);
        printf("%d:%f\n",id,rgen.sampleUniform(id,&g));
        printf("%d:%f\n",id,rowPtr[id]);
        rgen.put(id, g);
        
        //        printf("%ul\n", thresh);
        //printf("%ul\n", thresh2);
    }
    unsigned long long endTime = clock64();
    if (id == 0) {
        *time = (endTime - startTime);
    }   
}

void printSetBlockGridStats(int* thisNumBlocks, int* thisNumThreads) {
    const char *argv[] = {""};
    
    int argc = 0;
    int device = findCudaDevice(argc, argv);
    cudaDeviceProp prop = { 0 };
    //    cout << gpuGetMaxGflopsDeviceId() << endl;
    gpuErrchk( cudaSetDevice(device) );
    gpuErrchk( cudaGetDeviceProperties(&prop, device) );

    int numThreads = 120;
    int maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    int numSms = prop.multiProcessorCount;
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, test_kernel, numThreads, 0));
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
    printf("\tThreads per Block:\t%d\n", numThreads);
    printf("\tThreads:\t\t%d\n", numThreads * numBlocks);
    cout << endl;
    cout << "----------------------------------------------------------------------" << endl;

    *thisNumBlocks = numBlocks;
    *thisNumThreads = numThreads;
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
        printf("\tfree:\t\t%d MB\n", freeBytes / (1024*1024));
        printf("\tused:\t\t%d MB\n", usedBytes / (1024*1024));
        printf("\ttotal:\t\t%d MB\n", totalBytes / (1024*1024));
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
        printf("\tfree:\t\t%d MB\n", freeBytes / (1024*1024));
        printf("\tused:\t\t%d MB\n", usedBytes / (1024*1024));
        printf("\ttotal:\t\t%d MB\n", totalBytes / (1024*1024));
        cout << endl;
        printf("\tallocated:\t%d MB\n", allocated / (1024*1024));
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

    const dim3 dimBlock(numThreads,1,1);
    const dim3 dimGrid(numBlocks,1,1);
    
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


