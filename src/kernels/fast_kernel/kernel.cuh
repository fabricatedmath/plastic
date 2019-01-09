#pragma once

#include "input.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

typedef RandomGen<curandState> Rgen;

template <int numThreads>
__global__ void test_kernel(CudaMutableState ms,
                            CudaStaticState ss,
                            CudaBuffers b,
                            Rgen rgen,
                            unsigned long long* time) {
    unsigned long long startTime = clock64();
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    cg::grid_group grid = cg::this_grid();
    
    const int bid = block.thread_rank();
    
    const int row = blockIdx.x;
    
    __shared__ float sdata[numThreads];
    
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

    /* Init Shared for Block */
    for (int i = bid; i < NBNEUR; i += blockDim.x) {
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

    for (int i = bid; i < FFRFSIZE; i += blockDim.x) {
        sxplastFF[i] = ms.xplastFF.data[i];
        
        const float* wffRowPtr = getRowPtr(ms.wff, row);
        swff[i] = wffRowPtr[i];
    }
    cg::sync(block);
    /* End Init Shared for Block */

    /* Iters */
    for (int inputRow = 0; inputRow < 100; inputRow++) {
        cg::sync(grid);
        
        /* Clear State (that needs it) */
        for (int i = bid; i < NBNEUR; i++) {
            sincomingSpikes[i] = 0;
            sfirings[i] = 0;
        }

        //Init buffers
        {
            // Cache lgnfirings (apply random noise to for each stimulus presentation step)
            const float* rowPtr = getRowPtr(ss.input, inputRow);
            const int id = blockIdx.x * blockDim.x + threadIdx.x;
        
            curandState g = rgen.get(id);
            for (int row = blockIdx.x; row < NBSTEPSSTIM; row += gridDim.x) {
                float* lgnfiringsRowPtr = getRowPtr(b.lgnfirings, row);
                for (int i = bid; i < FFRFSIZE; i += blockDim.x) {
                    float rand = rgen.sampleUniform(bid,&g);
                    lgnfiringsRowPtr[i] = rand < rowPtr[i];
                }
            }
    
            //Generate poisson noise
            for (int row = blockIdx.x; row < NBSTEPSPERPRES; row += gridDim.x) {
                int* poissonNoiseRowPtr = getRowPtr(b.poissonNoise, row);
                for (int i = bid; i < NBNEUR; i += blockDim.x) {
                    int rand1 = rgen.samplePosPoisson(bid,&g);
                    int rand2 = rgen.sampleNegPoisson(bid,&g);
                    poissonNoiseRowPtr[i] = rand1 + rand2;
                }
            }
            rgen.put(id,g);
        }
        
        cg::sync(grid);
        cg::sync(block);
        
        /* Show image */
        for (int numStepsThisPres = 0; numStepsThisPres < NBSTEPSPERPRES; numStepsThisPres++) {
            for (int i = bid; i < FFRFSIZE; i += blockDim.x) {
                if (numStepsThisPres < NBSTEPSSTIM) {
                    const float* rowLgnFirings = getRowPtr(b.lgnfirings, numStepsThisPres);
                    slgnfirings[i] =  rowLgnFirings[i];
                } else {
                    slgnfirings[i] = 0;
                }
            }
            float iff = 0;
            {
                float acc = 0;

                #pragma unroll
                for (int i = bid; i < FFRFSIZE; i += block.size()) {
                    float lgnfiring = slgnfirings[i];
                    float wff = swff[i];
                    acc += wff * lgnfiring;
                }

                cg::sync(block);
                
                #pragma unroll
                for (int i = tile32.size() / 2; i > 0; i >>= 1) {
                    acc += tile32.shfl_down(acc,i);
                }
                
                sdata[bid] = acc;
                
                cg::sync(block);
                
                if (block.thread_rank() == 0) {
                    acc = 0;
                    #pragma unroll
                    for(int i = 0; i < block.size(); i += tile32.size()) {
                        acc += sdata[i];
                    }
                    iff = acc;
                }
            }
            
            for (int i = bid; i < NBNEUR; i += blockDim.x) {
                sv[i] += 1;
                svprev[i] += 1;
                svthresh[i] += 1;
                svlongtrace[i] += 1;
                svneg[i] += 1;
                svpos[i] += 1;
                sxplastLat[i] += 1;
                sw[i] += 1;
            }

            for (int i = bid; i < FFRFSIZE; i += blockDim.x) {
                sxplastFF[i] += 1;
                swff[i] += 1;
            }
        }
    }

    /* Copy back from shared to global */
    for (int i = bid; i < NBNEUR; i += blockDim.x) {
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

    for (int i = bid; i < FFRFSIZE; i += blockDim.x) {
        ms.xplastFF.data[i] = sxplastFF[i];
        
        float* wffRowPtr = getRowPtr(ms.wff, row);
        wffRowPtr[i] = swff[i];
    }
    
    unsigned long long endTime = clock64();
    if (bid == 0) {
        *time = endTime - startTime;
    }
}
