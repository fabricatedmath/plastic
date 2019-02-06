#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "err.cuh"

using namespace std;

template<typename F>
struct RandomGenHistorical {
    F* arrUniform;
    F* arrPosPoisson;
    F* arrNegPoisson;
    __device__ void* get(int id) { return NULL; }
    __device__ F sampleUniform(int idx, void* empty) { return arrUniform[idx]; }
    __device__ F samplePosPoisson(int idx, void* empty) { return arrPosPoisson[idx]; }
    __device__ F sampleNegPoisson(int idx, void* empty) { return arrNegPoisson[idx]; }
    __device__ void put(int id, void* empty) {}
};

template <typename G>
__global__ void setup_kernel(G *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, id, 0, &state[id]);
}

template<typename F, typename G>
__device__ F curand_uniform_internal(G* localState) {
    return curand_uniform(localState);
};

template <>
__device__ double curand_uniform_internal<double,curandState>(curandState* localState) {
    return curand_uniform_double(localState);
}

template <typename F, typename G>
struct RandomGen {
    G *states;
    curandDiscreteDistribution_t posPoisson;
    curandDiscreteDistribution_t negPoisson;

    RandomGen(const int numBlocks, const int numThreads, const double posLambda, const double negLambda) {
        gpuErrchkCuRand( curandCreatePoissonDistribution(posLambda,&posPoisson) );
        gpuErrchkCuRand( curandCreatePoissonDistribution(negLambda,&negPoisson) );
        gpuErrchk( cudaMalloc((void **)&states, numBlocks * numThreads * sizeof(G)) );
        setup_kernel<G><<<numBlocks,numThreads>>>(states);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    __device__ G get(int id) {
        return states[id];
    }
    __device__ F sampleUniform(int idx, G* localState) {
        return curand_uniform_internal<F,G>(localState);
    }
    __device__ unsigned int samplePosPoisson(int idx, G* localState) {
        return curand_discrete(localState, posPoisson);
    }
    __device__ unsigned int sampleNegPoisson(int idx, G* localState) {
        return curand_discrete(localState, negPoisson);
    }
    __device__ void put(int id, G localState) {
        states[id] = localState;
    }
};

