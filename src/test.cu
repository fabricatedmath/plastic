#include <iostream>
#include <unistd.h>
#include <cuda.h>
#include <Eigen/Dense>
#include "test.h"
#include "err.cuh"
#include "randGen.cuh"
#include "constants.h"
#include "cuda_state.cuh"
#include "helper_cuda.h"
#include "kernels/robust_kernel/kernel.cuh"
#include "stats.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std::chrono;

using namespace std;
using namespace Eigen;
namespace cg = cooperative_groups;

const int numThreadsConstant = NUMTHREADS;

template<typename F, typename I>
void wrapper(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers) {
    int numThreads = numThreadsConstant;

    typedef RandomGen<F,curandState> Rgen;
    auto func = test_kernel<F,I,Rgen,numThreadsConstant>;
    
    int numBlocks = printSetBlockGridStats(func, numThreadsConstant);
    
    Rgen cudaRgen(numBlocks, numThreads, 1.1, 1.8);

    unsigned long long time;
    unsigned long long* d_time;
    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );
    
    CudaMutableState<F,I> cudaMutableState;
    gpuErrchk( cudaMalloc(&mutableState,&cudaMutableState) );
    gpuErrchk( memcpyHostToDevice(&mutableState,&cudaMutableState) );

    CudaStaticState<F,I> cudaStaticState;
    gpuErrchk( cudaMalloc(&staticState,&cudaStaticState) );
    gpuErrchk( memcpyHostToDevice(&staticState,&cudaStaticState) );

    CudaBuffers<F,I> cudaBuffers;
    gpuErrchk( cudaMalloc(&buffers,&cudaBuffers) );
    gpuErrchk( memcpyHostToDevice(&buffers,&cudaBuffers) );

    printNvidiaSmi();

    numBlocks = min(NBNEUR,numBlocks);
    cout << "Num Blocks: " << numBlocks << endl;

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
        auto start = high_resolution_clock::now();
        gpuErrchk( cudaLaunchCooperativeKernel((void*)func,  dimGrid, dimBlock, kernelArgs, smemSize, NULL) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Duration: " << duration.count() << " us" << endl;

        gpuErrchk( memcpyDeviceToHost(&mutableState, &cudaMutableState) );
        gpuErrchk( memcpyDeviceToHost(&buffers, &cudaBuffers) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << "Clocks: " << time << endl;
    }
}

template void
wrapper<float,int>(MutableState<float,int> mutableState, StaticState<float,int> staticState, Buffers<float,int> buffers);

template void
wrapper<double,int>(MutableState<double,int> mutableState, StaticState<double,int> staticState, Buffers<double,int> buffers);

