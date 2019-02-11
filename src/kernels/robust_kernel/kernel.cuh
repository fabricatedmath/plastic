#pragma once

#include "input.cuh"

enum plasticity {};
enum noplasticity {};

template<typename F, typename I, typename Rgen>
__device__ void fillBuffers
(
    CudaMatrixX<F> input,
    CudaMatrixX<I> lgnfirings,
    CudaMatrixX<I> poissonNoise,
    CudaMatrixX<I> incomingSpikes,
    CudaVectorX<I> firingsV,
    Rgen rgen,
    const int inputRow
)
{
    const unsigned int tid = threadIdx.x;

    //Clear incoming spikes
    #pragma unroll
    for (int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
        volatile I* incomingSpikesRowPtr = incomingSpikes.getRowPtr(row);
        #pragma unroll
        for (int i = tid; i < NBNEUR; i += blockDim.x) {
            incomingSpikesRowPtr[i] = 0;
        }
    }

    if (blockIdx.x == 0) {
        for (int i = tid; i < NBNEUR; i+= blockDim.x) {
            volatile I* firings = firingsV.data;
            firings[i] = 0;
        }
    }

    //Cache lgnfirings (apply random noise to for each stimulus presentation step)
    const F* rowPtr = input.getRowPtr(inputRow);
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto g = rgen.get(id);
    #pragma unroll
    for (int row = blockIdx.x; row < NBSTEPSSTIM; row += gridDim.x) {
        volatile I* lgnfiringsRowPtr = lgnfirings.getRowPtr(row);
        #pragma unroll
        for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
            const F rand = rgen.sampleUniform(i,row,&g);
            lgnfiringsRowPtr[i] = rand < rowPtr[i];
        }
    }
    
    //Generate poisson noise
    #pragma unroll
    for (int row = blockIdx.x; row < NBSTEPSPERPRES; row += gridDim.x) {
        volatile I* poissonNoiseRowPtr = poissonNoise.getRowPtr(row);
        #pragma unroll
        for (int i = tid; i < NBNEUR; i += blockDim.x) {
            const int rand1 = rgen.samplePosPoisson(i,row,&g);
            const int rand2 = rgen.sampleNegPoisson(i,row,&g);
            poissonNoiseRowPtr[i] = rand1 + rand2;
        }
    }
    rgen.put(id,g);
}

template <typename F, typename I, class p, typename std::enable_if<std::is_same<p, plasticity>::value>::type* = nullptr>
inline __device__ void doPlasticity
(
     CudaMutableState<F,I> ms,
     CudaBuffers<F,I> b,
     const int numStepsThisPres,
     const int tid,
     const cg::thread_block block
) {}

template <typename F, typename I, class p, typename std::enable_if<std::is_same<p, noplasticity>::value>::type* = nullptr>
inline __device__ void doPlasticity
(
    CudaMutableState<F,I> ms,
    CudaBuffers<F,I> b,
    const int numStepsThisPres,
    const int tid,
    const cg::thread_block block
) {}

template <typename F, typename I, typename Rgen, int numThreads, class p>
__global__ void test_kernel(CudaMutableState<F,I> ms,
                            CudaStaticState<F,I> ss,
                            CudaBuffers<F,I> b,
                            Rgen rgen,
                            int inputRow,
                            const int numInputRows,
                            const int numPresThisLaunch,
                            unsigned long long* time) {
    const unsigned long long startTime = clock64();
    const cg::thread_block block = cg::this_thread_block();
    const cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    const cg::grid_group grid = cg::this_grid();
    const unsigned int tid = block.thread_rank();

    const int ffrfBlockOffset = NBNEUR / NUMTHREADS + 1;
    const int nid = threadIdx.x + blockDim.x * blockIdx.x;

    F vthresh, vlongtrace, vneg, vpos;
    F wadap, z, xplastLat, xplastFF;
    
    F altds;

    if (nid < NBNEUR) {
        vthresh = ms.vthresh.data[nid];
        vlongtrace = ms.vlongtrace.data[nid];
        vneg = ms.vneg.data[nid];
        vpos = ms.vpos.data[nid];
        
        wadap = ms.wadap.data[nid];
        z = ms.z.data[nid];
        
        xplastLat = ms.xplastLat.data[nid];
        xplastFF = ms.xplastFF.data[nid];
        
        altds = ss.altds.data[nid];
    }

    inputRow--;
    for (int numPres = 0; numPres < numPresThisLaunch; numPres++) {
        inputRow++;
        if (inputRow > numInputRows) {
            inputRow = 0;
        }
        
        fillBuffers(ss.input, b.lgnfirings, b.poissonNoise, b.incomingSpikes, b.firings, rgen, inputRow);
        cg::sync(grid);
        
        F v = ELEAK;
        int isSpiking = 0;
        for (int numStepsThisPres = 0; numStepsThisPres < NBSTEPSPERPRESRUN; numStepsThisPres++) {
            /* Calculate Inputs with block per Neuron */
            #pragma unroll
            for(int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
                const F neurLTP = b.eachNeurLTP.data[row];
                const F neurLTD = b.eachNeurLTD.data[row];
                
                const F iff = VSTIM * computeIFFNeuron<F,I,numThreads>(block, tile32, tid, ms.wff, b.lgnfirings, ms.xplastFF, neurLTP, neurLTD, numStepsThisPres, row);
                const F ilat = LATCONNMULT * VSTIM * computeILATNeuron<F,I,numThreads>(block, tile32, tid, ms.w, b.incomingSpikes, b.firings, ss.delays, ms.xplastLat, neurLTP, neurLTD, row);

                if (tid == 0) {
                    const I* noiseRowPtr = b.poissonNoise.getRowPtr(numStepsThisPres);
                    const I noise = noiseRowPtr[row];
                    const F input = iff + ilat + noise;
                    volatile F* neuronInputs = b.neuronInputs.data;
                    neuronInputs[row] = input;
                }
            }
            
            /* Sync blocks from Input calculation, threadfence for neuronInputs write */
            __threadfence();
            cg::sync(grid);

            /* Neuron per thread stuff */
            const int fid = threadIdx.x + blockDim.x * (blockIdx.x - ffrfBlockOffset);
            if (fid >= 0 && fid < FFRFSIZE) {
                I lgnfirings = 0;
                if (numStepsThisPres < NBSTEPSSTIM) {
                    const I* rowLgnFirings = b.lgnfirings.getRowPtr(numStepsThisPres);
                    lgnfirings = rowLgnFirings[fid];
                }
                xplastFF = xplastFF + lgnfirings / TAUXPLAST - (DT / TAUXPLAST) * xplastFF;
                ms.xplastFF.data[fid] = xplastFF;
            }

            if (nid < NBNEUR) {
                {
                    const F vprev = v;
                    vlongtrace = vlongtrace + (DT / TAUVLONGTRACE) * (max(0.0,(vprev - THETAVLONGTRACE)) - vlongtrace);
                    vneg = vneg + (DT / TAUVNEG) * (vprev - vneg);
                    vpos = vpos + (DT / TAUVPOS) * (vprev - vpos);
                }

                /* PRE-SPIKE UPDATE */

                const volatile F* neuronInputs = b.neuronInputs.data;
                const F input = neuronInputs[nid];

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
                I firing = 0;
                if (v > VPEAK) {
                    firing = 1;
                    v = VPEAK;
                    isSpiking = NBSPIKINGSTEPS;
                }
                xplastLat = xplastLat + firing / TAUXPLAST - (DT / TAUXPLAST) * xplastLat;
                int* firings = b.firings.data;
                firings[nid] = firing;
                ms.xplastLat.data[nid] = xplastLat;

                /* POST-SPIKE UPDATE */
                wadap = wadap + (DT / TAUADAP) * (CONSTA * (v - ELEAK) - wadap);
                z = z + (DT / TAUZ) * (-1.0) * z;
                vthresh = vthresh + (DT / TAUVTHRESH) * (-1.0 * vthresh + VTREST);
            }
            
            if (nid < NBE) {
                b.eachNeurLTD.data[nid] = DT * (-altds / VREF2) * vlongtrace * vlongtrace * max(0.0,vneg - THETAVNEG);
                b.eachNeurLTP.data[nid] = DT * ALTP * ALTPMULT * max(0.0, vpos - THETAVNEG) * max(0.0, v - THETAVPOS);
            }

            cg::sync(grid);
        }

        if (nid < NBNEUR) {
            ms.vthresh.data[nid] = vthresh;
            ms.vlongtrace.data[nid] = vlongtrace;
            ms.vneg.data[nid] = vneg;
            ms.vpos.data[nid] = vpos;
            
            ms.wadap.data[nid] = wadap;
            ms.z.data[nid] = z;
            
            ms.xplastLat.data[nid] = xplastLat;
            ms.xplastFF.data[nid] = xplastFF;
        }
    }
    unsigned long long endTime = clock64();
    if (nid == 0) {
        *time = (endTime - startTime);
    }   
}

