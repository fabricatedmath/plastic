#pragma once

#include "input.cuh"

//typedef RandomGen<curandState> Rgen;

//todo move shared to per block declaration

template<typename F, typename I, typename Rgen>
__device__ void fillBuffers(CudaMatrixX<F> input, CudaMatrixX<I> lgnfirings, CudaMatrixX<I> poissonNoise, CudaMatrixX<I> incomingSpikes, CudaVectorX<I> firings, Rgen rgen, int inputRow) {

    const unsigned int tid = threadIdx.x;

    //Clear incoming spikes
    for (int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
        I* incomingSpikesRowPtr = incomingSpikes.getRowPtr(row);
        for (int i = tid; i < NBNEUR; i += blockDim.x) {
            incomingSpikesRowPtr[i] = 0;
        }
    }

    if (blockIdx.x == 0) {
        for (int i = tid; i < NBNEUR; i+= blockDim.x) {
            firings.data[i] = 0;
        }
    }

    //Cache lgnfirings (apply random noise to for each stimulus presentation step)
    const F* rowPtr = input.getRowPtr(inputRow);
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto g = rgen.get(id);
    for (int row = blockIdx.x; row < NBSTEPSSTIM; row += gridDim.x) {
        I* lgnfiringsRowPtr = lgnfirings.getRowPtr(row);
        for (int i = tid; i < FFRFSIZE; i += blockDim.x) {
            F rand = rgen.sampleUniform(tid,&g);
            lgnfiringsRowPtr[i] = rand < rowPtr[i];
        }
    }
    
    //Generate poisson noise
    for (int row = blockIdx.x; row < NBSTEPSPERPRES; row += gridDim.x) {
        I* poissonNoiseRowPtr = poissonNoise.getRowPtr(row);
        for (int i = tid; i < NBNEUR; i += blockDim.x) {
            int rand1 = rgen.samplePosPoisson(tid,&g);
            int rand2 = rgen.sampleNegPoisson(tid,&g);
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
                            unsigned long long* time) {
    const unsigned long long startTime = clock64();
    __shared__ F sdata[numThreads];
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

    for (int inputRow = 0; inputRow < 100; inputRow++) {
        
        fillBuffers(ss.input, b.lgnfirings, b.poissonNoise, ms.incomingSpikes, ms.firings, rgen, inputRow);
        
        cg::sync(grid);

        F v = ELEAK;
        int isSpiking = 0;
        for (int numStepsThisPres = 0; numStepsThisPres < NBSTEPSPERPRES; numStepsThisPres++) {
            /* Calculate Inputs with block per Neuron */
            for(int row = blockIdx.x; row < NBNEUR; row += gridDim.x) {
                F iff = 0;
                if (numStepsThisPres < NBSTEPSSTIM) {
                    iff = VSTIM * computeIFFNeuron(sdata, block, tile32, tid, ms.wff, b.lgnfirings, numStepsThisPres, row);
                }

                F ilat = LATCONNMULT * VSTIM * computeILATNeuron(sdata, block, tile32, tid, ms.w, ms.incomingSpikes, ms.firings, ss.delays, row);

                if (tid == 0) {
                    const I* noiseRowPtr = b.poissonNoise.getRowPtr(numStepsThisPres);
                    const F noise = noiseRowPtr[row];
                    b.neuronInputs.data[row] = iff + ilat + noise;
                }
            }
            /* Sync blocks from Input calculation */
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

            //const int nid = threadIdx.x + blockDim.x * blockIdx.x;
            if (id < NBNEUR) {
                { 
                    const F vprev = v;
                    vlongtrace = vlongtrace + (DT / TAUVLONGTRACE) * (max(0.0,(vprev - THETAVLONGTRACE)) - vlongtrace);
                    vneg = vneg + (DT / TAUVNEG) * (vprev - vneg);
                    vpos = vpos + (DT / TAUVPOS) * (vprev - vpos);
                }

                /* PRE-SPIKE UPDATE */
                {
                    const F input = b.neuronInputs.data[id];
                    v += (DT/CONSTC) * (-GLEAK * (v - ELEAK) + GLEAK * DELTAT * expf((v-vthresh) / DELTAT) + z - wadap) + input;
                }

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
                {
                    I firing = 0;
                    if (v > VPEAK) {
                        firing = 1;
                        v = VPEAK;
                        isSpiking = NBSPIKINGSTEPS;
                    }
                    xplastLat = xplastLat + firing / TAUXPLAST - (DT / TAUXPLAST) * xplastLat;
                    ms.firings.data[id] = firing;
                    ms.xplastLat.data[id] = xplastLat;
                }

                /* POST-SPIKE UPDATE */
                wadap = wadap + (DT / TAUADAP) * (CONSTA * (v - ELEAK) - wadap);
                z = z + (DT / TAUZ) * (-1.0) * z;
                vthresh = vthresh + (DT / TAUVTHRESH) * (-1.0 * vthresh + VTREST);
            }

            /* PLASTICITY */
            if (id < NBE) {
                b.eachNeurLTD.data[id] = DT * (-altds / VREF2) * vlongtrace * vlongtrace * max(0.0,vneg - THETAVNEG);
                b.eachNeurLTP.data[id] = DT * ALTP * ALTPMULT * max(0.0, vpos - THETAVNEG) * max(0.0, v - THETAVPOS);
            }

            cg::sync(grid);

            for (int row = blockIdx.x; row < NBE; row += gridDim.x) {
                const F neurLTP = b.eachNeurLTP.data[row];
                const F neurLTD = b.eachNeurLTD.data[row];
                {
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
                }
                {
                    F* rowW = ms.w.getRowPtr(row);
                    for (int i = tid; i < NBE; i += block.size()) {
                        const F xplastLat = ms.xplastLat.data[i];    
                        const I firing = ms.firings.data[i];
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
            
            cg::sync(grid);
        }
    }
    unsigned long long endTime = clock64();
    if (id == 0) {
        *time = (endTime - startTime);
    }   
}
