#pragma once

__device__ float computeIffNeuron
(
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    const float* slgnfirings,
    const float* swff
)
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
    }
    return acc;
}
