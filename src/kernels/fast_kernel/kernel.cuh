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
    
    const unsigned int tid = block.thread_rank();
    
    const int row = blockIdx.x;
    
    float vthresh = ms.v.data[tid];
    
    float vlongtrace = ms.vlongtrace.data[row]; //only for tid 0
    float vneg = ms.vneg.data[row]; //only for tid 0
    float vpos = ms.vpos.data[row]; //only for tid 0

    float z = ms.z.data[tid];
    float wadap = ms.wadap.data[tid];

    float w = getRowPtr(ms.w, row)[tid];
    
    const int delay = getRowPtr(ss.delays, row)[tid];
    const int altds = ss.altds.data[row];

    float xplastLat = ms.xplastLat.data[tid];

    __shared__ float sxplastFF[FFRFSIZE];
    __shared__ float swff[FFRFSIZE];
    
    for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
        sxplastFF[i] = ms.xplastFF.data[i];
        
        const float* wffRowPtr = getRowPtr(ms.wff, row);
        swff[i] = wffRowPtr[i];
    }

    cg::sync(block);
    
    /* Iters */
    for (int inputRow = 0; inputRow < 100; inputRow++) {
        cg::sync(grid);

        float v = ELEAK;
        int isSpiking = 0;
        
        int incomingSpike = 0;
        int firing = 0;

        //Init poisson noise for each step of stimulus this input
        __shared__ int spoissonNoise[NBSTEPSPERPRES];

        {
            // Cache lgnfirings (apply random noise to for each stimulus presentation step)
            const float* rowPtr = getRowPtr(ss.input, inputRow);
            const int id = blockIdx.x * blockDim.x + threadIdx.x;
        
            curandState g = rgen.get(id);
            for (int row = blockIdx.x; row < NBSTEPSSTIM; row += gridDim.x) {
                int* lgnfiringsRowPtr = getRowPtr(b.lgnfirings, row);
                for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
                    float rand = rgen.sampleUniform(tid,&g);
                    lgnfiringsRowPtr[i] = rand < rowPtr[i];
                }
            }
    
            //Generate poisson noise
            for (int i = tid; i < NBSTEPSPERPRES; i += blockDim.x) {
                int posPoisson = rgen.samplePosPoisson(tid,&g);
                int negPoisson = rgen.sampleNegPoisson(tid,&g);
                spoissonNoise[i] = posPoisson + negPoisson;
            }
            rgen.put(id,g);
        }
        
        cg::sync(grid);
        
        /* Show image */
        for (int numStepsThisPres = 0; numStepsThisPres < NBSTEPSPERPRES; numStepsThisPres++) {
            __shared__ int slgnfirings[FFRFSIZE];
            if (numStepsThisPres < NBSTEPSSTIM) {
                #pragma unroll
                for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
                    const int* rowLgnFirings = getRowPtr(b.lgnfirings, numStepsThisPres);
                    const int lgnfiring = rowLgnFirings[i];
                    const float xplastFF = sxplastFF[i];
                    slgnfirings[i] = lgnfiring;
                    sxplastFF[i] = xplastFF + lgnfiring * INVTAUXPLAST - (DT * INVTAUXPLAST) * xplastFF;
                }
            } else {
                #pragma unroll
                for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
                    slgnfirings[i] = 0;
                    const float xplastFF = sxplastFF[i];
                    sxplastFF[i] = xplastFF - (DT * INVTAUXPLAST) * xplastFF;
                }
            }

            cg::sync(block);

            {
                const float iff = computeIffNeuron<numThreads>(block,tile32,tid,slgnfirings,swff);
                const float ilat = computeIlatNeuron<numThreads>(block,tile32,tid,w,incomingSpike,firing,delay,row);
                
                if (tid == 0) {
                    const float poissonNoise = spoissonNoise[numStepsThisPres];
                    const float input = iff + ilat + poissonNoise;
                    b.neuronInputs.data[row] = input;
                }
            }
            
            cg::sync(grid);
            
            __shared__ float sneurLTD;
            
            if (tid == 0) {
                const float vprev = v;
                vlongtrace = vlongtrace + (DT * INVTAUVLONGTRACE) * (max(0.0,(vprev - THETAVLONGTRACE)) - vlongtrace);
                vneg = vneg + (DT * TAUVNEG) * (vprev - vneg);
                vpos = vpos + (DT * TAUVPOS) * (vprev - vpos);
                sneurLTD = DT * (-altds * INVVREF2) * vlongtrace * vlongtrace * max(0.0, vneg - THETAVNEG);
            }

            {
                const float input = b.neuronInputs.data[tid];
                v += (DT * INVCONSTC) * (-GLEAK * (v - ELEAK) + GLEAK * DELTAT * expf((v-vthresh) * INVDELTAT) + z - wadap) + input;
            }

            if (isSpiking > 1) {
                v = VPEAK - 0.001;
            }

            if (isSpiking == 1) {
                v = VRESET;
                z = ISP;
                vthresh = VTMAX;
                wadap += CONSTB;
            }
            
            isSpiking = max(0,isSpiking - 1);
            v = max(v,MINV);
            
            firing = 0;
            if (v > VPEAK) {
                firing = 1;
                v = VPEAK;
                isSpiking = NBSPIKINGSTEPS;
            }
            
            xplastLat = xplastLat + firing * INVTAUXPLAST - (DT * INVTAUXPLAST) * xplastLat;
            
            wadap = wadap + (DT * INVTAUADAP) * (CONSTA * (v - ELEAK) - wadap);
            z = z + (DT * INVTAUZ) * (-1.0) * z;
            vthresh = vthresh + (DT * INVTAUVTHRESH) * (-1.0 * vthresh + VTREST);

            __shared__ float sneurLTP;
            if (tid == 0) {
                sneurLTP = DT * ALTP * ALTPMULT * max(0.0, vpos - THETAVNEG) * max(0.0, v - THETAVPOS);
            }

            if (tid < NBE) {
                w = w + xplastLat * sneurLTP;
                w = w + firing * sneurLTD * (1.0 + w * WPENSCALE);
                if (row == tid) {
                    w = 0.0;
                }
                w = max(0.0,w);
                w = min(MAXW,w);
            }

            for (int i = tid; i < FFRFSIZE; i += block.size()) {
                const float lgnfiring = slgnfirings[tid];
                float wff = swff[tid];
                const float xplastFF = sxplastFF[tid];
                wff = wff + xplastFF * sneurLTP;
                wff = wff + lgnfiring * sneurLTD * (1.0 + wff * WPENSCALE);
                wff = min(MAXW, max(0.0,wff));
                swff[tid] = wff;
            }
            cg::sync(grid);
        }
    }

    /* Copy back from shared to global */
    if (tid == 0) {
        ms.vlongtrace.data[row] = vlongtrace;
        ms.vneg.data[row] = vneg;
        ms.vpos.data[row] = vpos;
    }
    
    ms.v.data[tid] = 0;
    ms.vthresh.data[tid] = vthresh;
    ms.xplastLat.data[tid] = xplastLat;

    {
        float* wRowPtr = getRowPtr(ms.w, row);
        wRowPtr[tid] = w;
    }

    for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
        ms.xplastFF.data[i] = sxplastFF[i];
        
        float* wffRowPtr = getRowPtr(ms.wff, row);
        wffRowPtr[i] = swff[i];
    }
    
    unsigned long long endTime = clock64();
    if (tid == 0) {
        *time = endTime - startTime;
    }
}
