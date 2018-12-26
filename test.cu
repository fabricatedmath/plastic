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
#include "cuda_utility.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

using namespace std;
using namespace Eigen;
namespace cg = cooperative_groups;

const int rows = 50;
const int numThreads = 128;

typedef RandomGen<curandState> Rgen;

__global__ void spin_kernel() {
    unsigned long long startTime = clock64();
    unsigned long long hz = 2100000000;
    unsigned long long seconds = 100;
    unsigned long long thresh = hz * seconds;
    do {
    } while ((clock64() - startTime) < thresh);
}

__device__ void fillLgnFiringsBuffer(CudaMatrixXf input, CudaMatrixXf lgnfirings, Rgen rgen, int inputRow) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float* rowPtr = getRowPtr(input, inputRow);
    curandState g = rgen.get(id);
    float* lgnfiringsRowPtr;
    const unsigned int tid = threadIdx.x;
    for (int row = blockIdx.x; row < NBSTEPSSTIM; row += gridDim.x) {
        lgnfiringsRowPtr = getRowPtr(lgnfirings, row);
        for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
            float rand = rgen.sampleUniform(tid,&g);
            lgnfiringsRowPtr[i] = rand < rowPtr[i];
        }
    }
    rgen.put(id,g);
}

__global__ void test_kernel(CudaMutableState cudaMutableState,
                            CudaStaticState cudaStaticState,
                            CudaBuffers buffers,
                            Rgen rgen,
                            unsigned long long* time) {
    
    unsigned long long startTime = clock64();
    __shared__ float sdata[numThreads];
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    cg::grid_group grid = cg::this_grid();

    const unsigned int tid = block.thread_rank();
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int inputRow = 0; inputRow < 100; inputRow++) {
        fillLgnFiringsBuffer(cudaStaticState.input, buffers.lgnfirings, rgen, inputRow);
        cg::sync(grid);
        for (int numStepsThisPres = 0; numStepsThisPres < NBSTEPSPERPRES; numStepsThisPres++) {
            for(int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
                float iff = 0;
                if (numStepsThisPres < NBSTEPSSTIM) {
                    iff = VSTIM * computeIFFNeuron(sdata, block, tile32, tid, cudaMutableState.wff, buffers.lgnfirings, numStepsThisPres, row);
                }

                float ilat = LATCONNMULT * VSTIM * computeILATNeuron(sdata, block, tile32, tid, cudaMutableState.w, cudaMutableState.incomingSpikes, cudaMutableState.firings, cudaStaticState.delays, row);
                
                curandState g = rgen.get(id);
                float posNoise = rgen.samplePosPoisson(id,&g);
                float negNoise = rgen.sampleNegPoisson(id,&g);
                rgen.put(id,g);
                

                /*
                curandState g = rgen.get(id);
                float posNoise = rgen.sampleUniform(id,&g);
                float negNoise = rgen.sampleUniform(id,&g);
                rgen.put(id,g);
                */

                //float posNoise = 0;
//                float negNoise = 0;

                buffers.neuronInputs.data[row] = iff + ilat + posNoise + negNoise;
                /*
                if (row == 0 && threadIdx.x == 0) {
                    printf("%0.3f\n", iff);
                }
                */
            }
            cg::sync(grid);
        }
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

//    int numThreads = 128;
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

    numBlocks = 120;
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
        gpuErrchk( cudaLaunchCooperativeKernel((void*)test_kernel, dimGrid, dimBlock, kernelArgs, smemSize, NULL) );
        //gpuErrchk( cudaLaunchCooperativeKernel((void*)spin_kernel, dimGrid, dimBlock, spinKernelArgs, smemSize, NULL) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        gpuErrchk( memcpyDeviceToHost(&mutableState, &cudaMutableState) );
        gpuErrchk( memcpyDeviceToHost(&buffers, &cudaBuffers) );
        gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
        cout << time << endl;

        cout << buffers.lgnfirings.rows() << endl;

        /*
        for (int row = 0; row < buffers.lgnfirings.rows(); row++) {
            cout << buffers.lgnfirings.row(row) << endl;
        }
        */
    }
}


