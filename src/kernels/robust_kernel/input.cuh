#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//Use Kahan Summation?

template<typename F, typename I, int numThreads>
__device__ F computeIFFNeuron
(
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    CudaMatrixX<F> wffM,
    CudaMatrixX<I> lgnFiringsBuffer,
    CudaVectorX<F> xplastFFV,
    const F neurLTP,
    const F neurLTD,
    const int numStepsThisPres,
    const int row
)
{
    F acc = 0;

    const I numStepsThisPresPrev = numStepsThisPres - 1;
    
    const I* rowLgnFirings = lgnFiringsBuffer.getRowPtr(numStepsThisPres);
    const I* rowLgnFiringsPrev = lgnFiringsBuffer.getRowPtr(numStepsThisPresPrev);
    
    F* rowWff = wffM.getRowPtr(row);

    #pragma unroll
    for (int i = tid; i < FFRFSIZE; i += block.size()) {
        F wff = rowWff[i];
        
        /* WFF-PLASTICITY */
        if (numStepsThisPresPrev >= 0) {
            const F xplastFF = xplastFFV.data[i];
            wff = wff + xplastFF * neurLTP;
            if (numStepsThisPresPrev < NBSTEPSSTIM) {
                const I lgnfirings = rowLgnFiringsPrev[i];
                wff = wff + lgnfirings * neurLTD * (1.0 + wff * WPENSCALE);
            }
            wff = min(MAXW,max(0.0,wff));
            rowWff[i] = wff;
        }
        /* END WFF-PLASTICITY */
        
        if (numStepsThisPres < NBSTEPSSTIM) {
            const I lgnfirings = rowLgnFirings[i];
            acc += lgnfirings*wff;
        }
    }

    if (numStepsThisPres >= NBSTEPSSTIM) {
        return 0;
    }

    cg::sync(block);

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        acc += tile32.shfl_down(acc,i);
    }

    __shared__ F sdata[numThreads];
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

template<typename F, typename I, int numThreads>
__device__ F computeILATNeuron
(
    const cg::thread_block block,
    const cg::thread_block_tile<32> tile32,
    const unsigned int tid,
    CudaMatrixX<F> wM,
    CudaMatrixX<I> incomingSpikes,
    CudaVectorX<I> firingsV,
    CudaMatrixX<I> delays,
    CudaVectorX<F> xplastLatV,
    const F neurLTP,
    const F neurLTD,
    const int row
)
{   
    I* incomingSpikesRow = incomingSpikes.getRowPtr(row);
    const I* delaysRow = delays.getRowPtr(row);
    F* rowW = wM.getRowPtr(row);

    F acc = 0;

    #pragma unroll
    for (int i = tid; i < NBNEUR; i += block.size()) {
        const I firing = firingsV.data[i];
        F w = rowW[i];

        /* W-PLASTICITY */
        const F xplastLat = xplastLatV.data[i];
        w = w + xplastLat * neurLTP;
        w = w + firing * neurLTD * (1.0 + w * WPENSCALE);
        if (row == i) {
            w = 0.0;
        }
        if (i < NBE) {
            w = max(0.0,w);
        } else {
            w = min(0.0,w);
        }
        w = min(MAXW,w);
        rowW[i] = w;
        /* END W-PLASTICITY */

        I incomingSpike = incomingSpikesRow[i];
        if (i != row) {
            const I delay = delaysRow[i];
            incomingSpike = incomingSpike | (firing << (delay-1));
        }

        if (1 & incomingSpike == 1) {
            acc += w;
        }
        
        incomingSpikesRow[i] = incomingSpike >> 1;
    }

    cg::sync(block);

    #pragma unroll
    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        acc += tile32.shfl_down(acc,i);
    }
    
    __shared__ F sdata[numThreads];
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
