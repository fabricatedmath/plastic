#include <iostream>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"
#include "err.cuh"
#include "randGen.cuh"
#include "constants.h"
#include "cuda_state.cuh"
#include "input.cuh"
#include "helper_cuda.h"
//#include "robust.cuh"
#include "cuda_utility.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

using namespace std;
using namespace Eigen;
namespace cg = cooperative_groups;

const int rows = 50;
const int numThreadsConstant = 128;

typedef RandomGen<curandState> Rgen;

__global__ void spin_kernel() {
    unsigned long long startTime = clock64();
    unsigned long long hz = 2100000000;
    unsigned long long seconds = 100;
    unsigned long long thresh = hz * seconds;
    do {
    } while ((clock64() - startTime) < thresh);
}

template <int numThreads>
__global__ void test_kernel(CudaMutableState ms,
                            CudaStaticState ss,
                            CudaBuffers b,
                            Rgen rgen,
                            unsigned long long* time) {
    unsigned long long startTime = clock64();
    
    const int id = threadIdx.x;
    const int row = blockIdx.x;
    
    __shared__ float sv[NBNEUR];
    __shared__ float svprev[NBNEUR];
    __shared__ float svthresh[NBNEUR];
    __shared__ float svlongtrace[NBNEUR];
    __shared__ float svneg[NBNEUR];
    __shared__ float svpos[NBNEUR];
    
    __shared__ float sw[NBNEUR];
    __shared__ float sxplastLat[NBNEUR];
    
    __shared__ int sincomingSpikes[NBNEUR];
    __shared__ int sfirings[NBNEUR];
    
    __shared__ int sdelays[NBNEUR];
    __shared__ int saltds[NBNEUR];
    
    __shared__ float swff[FFRFSIZE];
    __shared__ float slgnfirings[FFRFSIZE];
    __shared__ float sxplastFF[FFRFSIZE];
    
    for (int i = id; i < NBNEUR; i += blockDim.x) {
        sv[i] = ms.v.data[i];
        svprev[i] = ms.vprev.data[i];
        svthresh[i] = ms.vthresh.data[i];
        svlongtrace[i] = ms.vlongtrace.data[i];
        svneg[i] = ms.vneg.data[i];
        svpos[i] = ms.vpos.data[i];
        sxplastLat[i] = ms.xplastLat.data[i];

        saltds[i] = ss.altds.data[i];
        
        const int* delaysRowPtr = getRowPtr(ss.delays, row);
        sdelays[i] = delaysRowPtr[i];

        const float* wRowPtr = getRowPtr(ms.w, row);
        sw[i] = wRowPtr[i];
    }

    for (int i = id; i < FFRFSIZE; i += blockDim.x) {
        sxplastFF[i] = ms.xplastFF.data[i];
        
        const float* wffRowPtr = getRowPtr(ms.wff, row);
        swff[i] = wffRowPtr[i];
    }

    for (int inputRow = 0; inputRow < 100; inputRow++) {
        /* Clear State (that needs it) */
        for (int i = id; i < NBNEUR; i++) {
            sincomingSpikes[i] = 0;
            sfirings[i] = 0;
        }
        
        for (int numStepsThisPres = 0; numStepsThisPres < 1000; numStepsThisPres++) {
            for (int i = id; i < NBNEUR; i += blockDim.x) {
                sv[i] += 1;
                svprev[i] += 1;
                svthresh[i] += 1;
                svlongtrace[i] += 1;
                svneg[i] += 1;
                svpos[i] += 1;
                sxplastLat[i] += 1;
                sw[i] += 1;
            }

            for (int i = id; i < FFRFSIZE; i += blockDim.x) {
                sxplastFF[i] += 1;
                swff[i] += 1;
            }
        }
    }

    for (int i = id; i < NBNEUR; i += blockDim.x) {
        ms.v.data[i] = sv[i];
        ms.vprev.data[i] = svprev[i];
        ms.vthresh.data[i] = svthresh[i];
        ms.vlongtrace.data[i] = svlongtrace[i];
        ms.vneg.data[i] = svneg[i];
        ms.vpos.data[i] = svpos[i];
        ms.xplastLat.data[i] = sxplastLat[i];

        float* wRowPtr = getRowPtr(ms.w, row);
        wRowPtr[i] = sw[i];
    }

    for (int i = id; i < FFRFSIZE; i += blockDim.x) {
        ms.xplastFF.data[i] = sxplastFF[i];
        
        float* wffRowPtr = getRowPtr(ms.wff, row);
        wffRowPtr[i] = swff[i];
    }
    
    unsigned long long endTime = clock64();
    if (id == 0) {
        *time = endTime - startTime;
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


