#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//Use Kahan Summation?

template<typename F, typename I>
__device__ F computeIFFNeuron
(
    F* sdata,
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    CudaMatrixX<F> wff,
    CudaMatrixX<I> lgnFiringsBuffer,
    const int inputRow,
    const int row
)
{
    F acc = 0;

    const I* rowLgnFirings = lgnFiringsBuffer.getRowPtr(inputRow);
    const F* rowWff = wff.getRowPtr(row);
    #pragma unroll
    for (int i = tid; i < FFRFSIZE; i += block.size()) {
        const I a = rowLgnFirings[i];
        const F m = rowWff[i];
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

template<typename F, typename I>
__device__ F computeILATNeuron
(
    F* sdata,
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    CudaMatrixX<F> w,
    CudaMatrixX<I> incomingSpikes,
    CudaVectorX<I> firings,
    CudaMatrixX<I> delays,
    const int row
)
{
    I* incomingSpikesRow = incomingSpikes.getRowPtr(row);
    const I* delaysRow = delays.getRowPtr(row);
    const F* wRow = w.getRowPtr(row);

    F acc = 0;

    #pragma unroll
    for (int i = tid; i < NBNEUR; i += block.size()) {
        I incomingSpike = incomingSpikesRow[i];
        
        if (i != row) {
            const I delay = delaysRow[i];
            const I firing = firings.data[i];
            incomingSpike = incomingSpike | (firing << (delay-1));
        }

        incomingSpikesRow[i] = incomingSpike >> 1;

        if (1 & incomingSpike == 1) {
            const F wVal = wRow[i];
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
