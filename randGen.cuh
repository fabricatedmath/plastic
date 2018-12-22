#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "err.cuh"

using namespace std;

struct RandomGenHistorical {
    float* arrUniform;
    float* arrPosPoisson;
    float* arrNegPoisson;
    __device__ void* get(int id) { return NULL; }
    __device__ float sampleUniform(int idx, void* empty) { return arrUniform[idx]; }
    __device__ float samplePosPoisson(int idx, void* empty) { return arrPosPoisson[idx]; }
    __device__ float sampleNegPoisson(int idx, void* empty) { return arrNegPoisson[idx]; }
    __device__ void put(int id, void* empty) {}
};

template <typename G>
__global__ void setup_kernel(G *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

template <typename G, int Dg, int Db>
struct RandomGen {
    G *states;
    curandDiscreteDistribution_t posPoisson;
    curandDiscreteDistribution_t negPoisson;

    RandomGen(const double posLambda, const double negLambda) {
        gpuErrchkCuRand( curandCreatePoissonDistribution(posLambda,&posPoisson) );
        gpuErrchkCuRand( curandCreatePoissonDistribution(negLambda,&negPoisson) );        
        gpuErrchk( cudaMalloc((void **)&states, Dg * Db * sizeof(G)) );
        setup_kernel<G><<<Dg,Db>>>(states);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    __device__ G get(int id) {
        return states[id];
    }
    __device__ float sampleUniform(int idx, G* localState) {
        return curand_uniform(localState);
    }
    __device__ float samplePosPoisson(int idx, G* localState) {
        return curand_discrete(localState, posPoisson);
    }
    __device__ float sampleNegPoisson(int idx, G* localState) {
        return curand_discrete(localState, negPoisson);
    }
    __device__ void put(int id, G localState) {
        states[id] = localState;
    }
};
