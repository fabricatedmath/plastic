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

__device__ void fillBuffers(const CudaMatrixXf input, CudaMatrixXf lgnfirings, CudaMatrixXi poissonNoise, Rgen rgen, int inputRow) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    const float* rowPtr = getRowPtr(input, inputRow);
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

    int* poissonNoiseRowPtr; 
    for (int row = blockIdx.x; row < NBSTEPSPERPRES; row += gridDim.x) {
        poissonNoiseRowPtr = getRowPtr(poissonNoise, row);
        for (int i = tid; i < NBNEUR; i += blockDim.x) {
            int rand1 = rgen.samplePosPoisson(tid,&g);
            int rand2 = rgen.sampleNegPoisson(tid,&g);
            poissonNoiseRowPtr[i] = rand1 + rand2;
        }
    }
    rgen.put(id,g);
}

__global__ void test_kernel(CudaMutableState ms,
                            const CudaStaticState ss,
                            CudaBuffers b,
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
        fillBuffers(ss.input, b.lgnfirings, b.poissonNoise, rgen, inputRow);
//        fillBuffers(ss.input, b.lgnfirings, rgen, inputRow);
        cg::sync(grid);
        for (int numStepsThisPres = 0; numStepsThisPres < NBSTEPSPERPRES; numStepsThisPres++) {
            /* Calculate Inputs with block per Neuron */
            for(int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
                float iff = 0;
                if (numStepsThisPres < NBSTEPSSTIM) {
                    iff = VSTIM * computeIFFNeuron(sdata, block, tile32, tid, ms.wff, b.lgnfirings, numStepsThisPres, row);
                }

                float ilat = LATCONNMULT * VSTIM * computeILATNeuron(sdata, block, tile32, tid, ms.w, ms.incomingSpikes, ms.firings, ss.delays, row);

                int* noiseRowPtr = getRowPtr(b.poissonNoise, numStepsThisPres);
                float noise = noiseRowPtr[row];
                b.neuronInputs.data[row] = iff + ilat + noise;
            }
            /* Sync blocks from Input calculation */
            cg::sync(grid);

            /* Neuron per thread stuff */
            for (int neuron = id; neuron < NBNEUR; neuron += gridDim.x) {
                float v = ms.v.data[id];
                float vprev = v;
                float vthresh = ms.vthresh.data[id];
                float input = b.neuronInputs.data[id];
                float wadap = ms.wadap.data[id];
                float z = ms.z.data[id];
                int isSpiking = ms.isSpiking.data[id];
                int firing = ms.firings.data[id];
                float vlongtrace = ms.vlongtrace.data[id];
                float xplastLat = ms.xplastLat.data[id];
                float xplastFF = ms.xplastFF.data[id];
                
                float lgnfirings = 0;
                if (numStepsThisPres < NBSTEPSSTIM) {
                    float* rowLgnFirings = getRowPtr(b.lgnfirings, numStepsThisPres);
                    lgnfirings = rowLgnFirings[id];
                }

                float vneg = ms.vneg.data[id];
                float vpos = ms.vpos.data[id];

                /* PRE-SPIKE UPDATE */
                v += (DT/CONSTC) * (-GLEAK * (v - ELEAK) + GLEAK * DELTAT * expf((v-vthresh) / DELTAT) + z - wadap) + input;

                if (isSpiking > 1) {
                    v = VPEAK-0.001;
                }

                if (isSpiking == 1) {
                    v = VRESET;
                    z = ISP;
                    vthresh = VTMAX;
                    wadap += CONSTB;
                }
                isSpiking = max(0,isSpiking - 1);
                v = max(v,MINV);

                /* SPIKE UPDATE */
                firing = 0;
                if (v > VPEAK) {
                    firing = 1;
                    v = VPEAK;
                    isSpiking = NBSPIKINGSTEPS;
                }

                /* POST-SPIKE UPDATE */
                wadap = wadap + (DT / TAUADAP) * (CONSTA * (v - ELEAK) - wadap);
                z = z + (DT / TAUZ) * (-1.0) * z;
                vthresh = vthresh + (DT / TAUVTHRESH) * (-1.0 * vthresh + VTREST);
                vlongtrace = vlongtrace + (DT / TAUVLONGTRACE) * (max(0.0,(vprev - THETAVLONGTRACE)) - vlongtrace);

                xplastLat = xplastLat + firing / TAUXPLAST - (DT / TAUXPLAST) * xplastLat;
                xplastFF = xplastFF + lgnfirings / TAUXPLAST - (DT / TAUXPLAST) * xplastFF;

                float altds = ss.altds.data[id];

                /* PLASTICITY */
                
                b.eachNeurLTD.data[id] = DT * (-altds / VREF2) * vlongtrace * vlongtrace * max(0.0,vneg - THETAVNEG);
                b.eachNeurLTP.data[id] = DT * ALTP * ALTPMULT * max(0.0, vpos - THETAVNEG) * max(0.0, v - THETAVPOS);

                
                ms.v.data[id] = v;
            }

            /* Plasticity */
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
        gpuErrchk( cudaLaunchCooperativeKernel((void*)test_kernel, dimGrid, dimBlock, kernelArgs, smemSize, NULL) );
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


