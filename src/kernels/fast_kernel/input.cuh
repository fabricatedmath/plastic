#pragma once

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<int numThreads>
__device__ float computeIffNeuron
(
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    const int* slgnfirings,
    const float* swff
)
{
    __shared__ float sdata[numThreads];
    
    float acc = 0;
    
    #pragma unroll
    for (int i = tid; i < FFRFSIZE; i += block.size()) {
        int lgnfiring = slgnfirings[i];
        float wff = swff[i];
        acc += wff * lgnfiring;
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

template<int numThreads>
__device__ float computeIlatNeuron
(
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    const float w,
    int& incomingSpike,
    const int firing,
    const int delay,
    const int row
)
{
    __shared__ float sdata[numThreads];
    
    float acc = 0;
    
    if (tid != row) {
        incomingSpike = incomingSpike | (firing << (delay-1));
    }
    
    incomingSpike = incomingSpike >> 1;
    
    if (1 & incomingSpike == 1) {
        acc += w;
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
