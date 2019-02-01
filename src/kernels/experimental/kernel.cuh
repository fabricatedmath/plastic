#pragma once

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
    const unsigned int tid = block.thread_rank();

    int acc = tid;

    cg::sync(tile32);

    for (int i = 0; i < 100; i++) {
        float v = ms.v.data[tid];
        v = v+1;
        ms.v.data[tid] = v;
    }
    
    /*
    #pragma unroll
    for (int numSteps = 0; numSteps < 100; numSteps++) {
        acc = numSteps + tid;
        #pragma unroll
        for (int i = tile32.size() / 2; i > 0; i >>= 1) {
            acc += tile32.shfl_down(acc,i);
        }
    }
    */
    
    //if (tid == 0 && acc > 500000) {
//        printf("dog");
//    }
    
    unsigned long long endTime = clock64();
    if (tid == 0) {
        *time = endTime - startTime;
    }
}
