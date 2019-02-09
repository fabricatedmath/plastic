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

const int numThreads = NUMTHREADS;

template<typename F, typename I>
MutableState<F,I> run(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers) {
    typedef RandomGen<F,curandState> Rgen;
    auto func = test_kernel<F,I,Rgen,numThreads>;
    int numBlocks = printSetBlockGridStats(func, numThreads);
    numBlocks = min(NBNEUR,numBlocks);
    Rgen cudaRgen(numBlocks, numThreads, POSNOISERATE, NEGNOISERATE);
    return wrapper(mutableState, staticState, buffers, func, cudaRgen, numBlocks);
}

template<typename F, typename I>
MutableState<F,I> run(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers, RandomHistorical<F> randomHistorical) {
    typedef RandomGenHistorical<F> Rgen;
    auto func = test_kernel<F,I,Rgen,numThreads>;
    int numBlocks = printSetBlockGridStats(func, numThreads);
    numBlocks = min(NBNEUR,numBlocks);
    Rgen cudaRgen(randomHistorical);
    return wrapper(mutableState, staticState, buffers, func, cudaRgen, numBlocks);
}

template<class T, typename F, typename I, typename Rgen>
MutableState<F,I> wrapper(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers, T func, Rgen cudaRgen, const int numBlocks) {
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

    cout << "Num Blocks Used: " << numBlocks << endl << endl;

    int inputRow = 0;
    int numInputRows = 110000;
    int numPresThisLaunch = 1;

    void *kernelArgs[] = {
        (void*)&cudaMutableState,
        (void*)&cudaStaticState,
        (void*)&cudaBuffers,
        (void*)&cudaRgen,
        (void*)&inputRow,
        (void*)&numInputRows,
        (void*)&numPresThisLaunch,
        (void*)&d_time
    };

    const dim3 dimBlock(numThreads,1,1);
    const dim3 dimGrid(numBlocks,1,1);
    const int smemSize = 0;
    
    for (int i = 0; i < 1; i++) {
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
        cout << "Clocks: " << time << endl << endl;
    }

    return mutableState;
}

template MutableState<float,int>
run<float,int>(MutableState<float,int> mutableState, StaticState<float,int> staticState, Buffers<float,int> buffers);

template MutableState<double,int>
run<double,int>(MutableState<double,int> mutableState, StaticState<double,int> staticState, Buffers<double,int> buffers);

template MutableState<float,int>
run<float,int>(MutableState<float,int> mutableState, StaticState<float,int> staticState, Buffers<float,int> buffers, RandomHistorical<float> randomHistorical);

template MutableState<double,int>
run<double,int>(MutableState<double,int> mutableState, StaticState<double,int> staticState, Buffers<double,int> buffers, RandomHistorical<double> randomHistorical);

