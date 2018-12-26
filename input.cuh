#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//Use Kahan Summation?
__device__ float computeIFFNeuron
(
    float* sdata,
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    const CudaMatrixXf wff,
    const CudaMatrixXf lgnFiringsBuffer,
    const int inputRow,
    const int row
)
{
    float acc = 0;

    float* rowLgnFirings = getRowPtr(lgnFiringsBuffer, inputRow);
    float* rowWff = getRowPtr(wff, row);
    #pragma unroll
    for (int i = tid; i < FFRFSIZE; i+=block.size()) {
        float a = rowLgnFirings[i];
        float m = rowWff[i];
        acc += a*m;
    }

    cg::sync(block);

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        acc += tile32.shfl_down(acc,i);
    }
    sdata[tid] = acc;

    cg::sync(block);
    
    if (block.thread_rank() == 0) {
        acc = 0;
        #pragma unroll
        for(int i = 0; i < block.size(); i += tile32.size()) {
            acc += sdata[i];
        }
    }
    return acc;
}

__device__ float computeILATNeuron
(
    float* sdata,
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    CudaMatrixXf w,
    CudaMatrixXi incomingSpikes,
    CudaVectorXi firings,
    CudaMatrixXi delays,
    const int row
)
{
    int* incomingSpikesRow = getRowPtr(incomingSpikes, row);
    int* delaysRow = getRowPtr(delays, row);
    float* wRow = getRowPtr(w, row);

    float acc = 0;

    #pragma unroll
    for (int i = tid; i < NBNEUR; i += block.size()) {
        int incomingSpike = incomingSpikesRow[i];
        float wVal = wRow[i];
        int delay = delaysRow[i];
        int firing = firings.data[i];
        
        incomingSpike = incomingSpike | (firing << (delay-1));
        incomingSpikesRow[i] = incomingSpike;

        if (1 & incomingSpike == 1) {
            acc += wVal;
        }
    }
    
    cg::sync(block);

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        acc += tile32.shfl_down(acc,i);
    }
    sdata[tid] = acc;

    cg::sync(block);
    
    if (block.thread_rank() == 0) {
        acc = 0;
        #pragma unroll
        for(int i = 0; i < block.size(); i += tile32.size()) {
            acc += sdata[i];
        }
    }
    return acc;   
}
