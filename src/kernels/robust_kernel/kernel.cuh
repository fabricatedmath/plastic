#pragma once

#include "input.cuh"

template<typename F, typename I, typename Rgen>
__device__ void fillBuffers
(
    CudaMatrixX<F> input,
    CudaMatrixX<I> lgnfirings,
    CudaMatrixX<I> poissonNoise,
    CudaMatrixX<I> incomingSpikes,
    CudaVectorX<I> firingsV,
    Rgen rgen,
    int inputRow
)
{
    const unsigned int tid = threadIdx.x;

    //Clear incoming spikes
    for (int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
        volatile I* incomingSpikesRowPtr = incomingSpikes.getRowPtr(row);
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
    for (int row = blockIdx.x; row < NBSTEPSSTIM; row += gridDim.x) {
        volatile I* lgnfiringsRowPtr = lgnfirings.getRowPtr(row);
        for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
            const F rand = rgen.sampleUniform(i,row,&g);
            lgnfiringsRowPtr[i] = rand < rowPtr[i];
        }
    }
    
    //Generate poisson noise
    for (int row = blockIdx.x; row < NBSTEPSPERPRES; row += gridDim.x) {
        volatile I* poissonNoiseRowPtr = poissonNoise.getRowPtr(row);
        for (int i = tid; i < NBNEUR; i += blockDim.x) {
            const int rand1 = rgen.samplePosPoisson(i,row,&g);
            const int rand2 = rgen.sampleNegPoisson(i,row,&g);
            poissonNoiseRowPtr[i] = rand1 + rand2;
        }
    }
    rgen.put(id,g);
}


template <typename F, typename I, typename Rgen, int numThreads>
__global__ void test_kernel(CudaMutableState<F,I> ms,
                            CudaStaticState<F,I> ss,
                            CudaBuffers<F,I> b,
                            Rgen rgen,
                            int inputRow,
                            const int numInputRows,
                            const int numPresThisLaunch,
                            unsigned long long* time) {
    const unsigned long long startTime = clock64();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);
    cg::grid_group grid = cg::this_grid();
    const unsigned int tid = block.thread_rank();

    const int ffrfBlockOffset = NBNEUR / NUMTHREADS + 1;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    F vthresh, vlongtrace, vneg, vpos;
    F wadap, z, xplastLat, xplastFF;
    
    F altds;

    if (id < NBNEUR) {
        vthresh = ms.vthresh.data[id];
        vlongtrace = ms.vlongtrace.data[id];
        vneg = ms.vneg.data[id];
        vpos = ms.vpos.data[id];
        
        wadap = ms.wadap.data[id];
        z = ms.z.data[id];
        
        xplastLat = ms.xplastLat.data[id];
        xplastFF = ms.xplastFF.data[id];
        
        altds = ss.altds.data[id];
    }

    inputRow--;
    for (int numPres = 0; numPres < numPresThisLaunch; numPres++) {
        inputRow++;
        if (inputRow > numInputRows) {
            inputRow = 0;
        }
        cg::sync(grid);
        fillBuffers(ss.input, b.lgnfirings, b.poissonNoise, b.incomingSpikes, b.firings, rgen, inputRow);
        cg::sync(grid);

        F v = ELEAK;
        int isSpiking = 0;
        for (int numStepsThisPres = 0; numStepsThisPres < 350; numStepsThisPres++) {
            /*
            if (id == 0) {
                printf("\nnumpres: %d\n\n", numStepsThisPres);
            }
            */
            cg::sync(grid);
            /* Calculate Inputs with block per Neuron */
            for(int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
                F iff = 0;
                if (numStepsThisPres < NBSTEPSSTIM) {
                    iff = VSTIM * computeIFFNeuron<F,I,numThreads>(block, tile32, tid, ms.wff, b.lgnfirings, numStepsThisPres, row);
                }
                
                const F ilat = LATCONNMULT * VSTIM * computeILATNeuron<F,I,numThreads>(block, tile32, tid, ms.w, b.incomingSpikes, b.firings, ss.delays, row);
                
                if (tid == 0) {
                    const I* noiseRowPtr = b.poissonNoise.getRowPtr(numStepsThisPres);
                    const I noise = noiseRowPtr[row];
                    const F input = iff + ilat + noise;
                    //printf("%d : %.15f %.15f %.15f\n", row, iff, ilat, input);
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
                if (fid == 326) {
                    printf("%d : %.15f\n", numStepsThisPres, xplastFF);
                }
            }

            //const int nid = threadIdx.x + blockDim.x * blockIdx.x;
            if (id < NBNEUR) {
                {
                    const F vprev = v;
                    vlongtrace = vlongtrace + (DT / TAUVLONGTRACE) * (max(0.0,(vprev - THETAVLONGTRACE)) - vlongtrace);
                    vneg = vneg + (DT / TAUVNEG) * (vprev - vneg);
                    vpos = vpos + (DT / TAUVPOS) * (vprev - vpos);
                }

                /* PRE-SPIKE UPDATE */

                const volatile F* neuronInputs = b.neuronInputs.data;
                const F input = neuronInputs[id];

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
                firings[id] = firing;
                ms.xplastLat.data[id] = xplastLat;

                /* POST-SPIKE UPDATE */
                wadap = wadap + (DT / TAUADAP) * (CONSTA * (v - ELEAK) - wadap);
                z = z + (DT / TAUZ) * (-1.0) * z;
                vthresh = vthresh + (DT / TAUVTHRESH) * (-1.0 * vthresh + VTREST);
            }

            const bool doPlasticity = false;
            if (doPlasticity) {
                /* PLASTICITY */
                if (id < NBE) {
                    b.eachNeurLTD.data[id] = DT * (-altds / VREF2) * vlongtrace * vlongtrace * max(0.0,vneg - THETAVNEG);
                    b.eachNeurLTP.data[id] = DT * ALTP * ALTPMULT * max(0.0, vpos - THETAVNEG) * max(0.0, v - THETAVPOS);
                }

                cg::sync(grid);

                for (int row = blockIdx.x; row < NBE; row += gridDim.x) {
                    const F neurLTP = b.eachNeurLTP.data[row];
                    const F neurLTD = b.eachNeurLTD.data[row];

                    const I* rowLgnFirings = b.lgnfirings.getRowPtr(numStepsThisPres);
                    F* rowWff = ms.wff.getRowPtr(row);

                    for (int i = tid; i < FFRFSIZE; i += block.size()) {
                        const F xplastFF = ms.xplastFF.data[i];
                        I lgnfirings = 0;
                        if (numStepsThisPres < NBSTEPSSTIM) {
                            lgnfirings = rowLgnFirings[i];
                        }
                        F wff = rowWff[i];
                        wff = wff + xplastFF * neurLTP;
                        wff = wff + lgnfirings * neurLTD * (1.0 + wff * WPENSCALE);
                        wff = min(MAXW,max(0.0,wff));
                        rowWff[i] = wff;
                    }

                    F* rowW = ms.w.getRowPtr(row);
                    for (int i = tid; i < NBE; i += block.size()) {
                        const F xplastLat = ms.xplastLat.data[i];    
                        const I firing = b.firings.data[i];
                        F w = rowW[i];
                        w = w + xplastLat * neurLTP;
                        w = w + firing * neurLTD * (1.0 + w * WPENSCALE);
                        if (row == i) {
                            w = 0.0;
                        }
                        if (i < NBE) {
                            //Excitory Pruning
                            w = max(0.0,w);
                        } else {
                            // Inhibitory Pruning
                            w = min(0.0,w);
                        }
                        w = min(MAXW,w);
                        rowW[i] = w;
                    }
                }
            }
        }

        if (id < NBNEUR) {
            ms.vthresh.data[id] = vthresh;
            ms.vlongtrace.data[id] = vlongtrace;
            ms.vneg.data[id] = vneg;
            ms.vpos.data[id] = vpos;
            
            ms.wadap.data[id] = wadap;
            ms.z.data[id] = z;
            
            ms.xplastLat.data[id] = xplastLat;
            ms.xplastFF.data[id] = xplastFF;
        }
    }
    unsigned long long endTime = clock64();
    if (id == 0) {
        *time = (endTime - startTime);
    }   
}
