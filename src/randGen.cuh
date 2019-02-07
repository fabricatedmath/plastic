#pragma once

#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include "cuda_state.cuh"
#include "err.cuh"

using namespace std;

template<typename F>
struct RandomGenHistorical {
    CudaMatrixX<F> uniformCudaMatrix;
    CudaMatrixX<unsigned int> posPoissonCudaMatrix;
    CudaMatrixX<unsigned int> negPoissonCudaMatrix;

    RandomGenHistorical(RandomHistorical<F> randomHistorical) {
        gpuErrchk( cudaMalloc(&randomHistorical.uniformMatrix, &uniformCudaMatrix) );
        gpuErrchk( memcpyHostToDevice(&randomHistorical.uniformMatrix, &uniformCudaMatrix) );

        gpuErrchk( cudaMalloc(&randomHistorical.posPoissonMatrix, &posPoissonCudaMatrix) );
        gpuErrchk( memcpyHostToDevice(&randomHistorical.posPoissonMatrix, &posPoissonCudaMatrix) );

        gpuErrchk( cudaMalloc(&randomHistorical.negPoissonMatrix, &negPoissonCudaMatrix) );
        gpuErrchk( memcpyHostToDevice(&randomHistorical.negPoissonMatrix, &negPoissonCudaMatrix) );
    }
    
    __device__ void* get(int id) {
        return NULL;
    }
    
    __device__ F sampleUniform(int x, int y, void* empty) {
        const F* rowPtr = uniformCudaMatrix.getRowPtr(y);
        return rowPtr[x];
    }
    
    __device__ unsigned int samplePosPoisson(int x, int y, void* empty) {
        const unsigned int* rowPtr = posPoissonCudaMatrix.getRowPtr(y);
        return rowPtr[x];
    }
    
    __device__ unsigned int sampleNegPoisson(int x, int y, void* empty) {
        const unsigned int* rowPtr = negPoissonCudaMatrix.getRowPtr(y);
        return rowPtr[x];
    }
    
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

    const double posLambda;
    const double negLambda;

    curandDiscreteDistribution_t posPoisson;
    curandDiscreteDistribution_t negPoisson;

    RandomGen(const int numBlocks, const int numThreads, const double posLambda, const double negLambda)
        : posLambda(posLambda), negLambda(negLambda) {
        if (posLambda != 0.0) {
            gpuErrchkCuRand( curandCreatePoissonDistribution(posLambda,&posPoisson) );
        }
        if (negLambda != 0.0) {
            gpuErrchkCuRand( curandCreatePoissonDistribution(negLambda,&negPoisson) );
        }
        gpuErrchk( cudaMalloc((void **)&states, numBlocks * numThreads * sizeof(G)) );
        setup_kernel<G><<<numBlocks,numThreads>>>(states);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    __device__ G get(int id) {
        return states[id];
    }
    __device__ F sampleUniform(int x, int y, G* localState) {
        return curand_uniform_internal<F,G>(localState);
    }
    __device__ unsigned int samplePosPoisson(int x, int y, G* localState) {
        if (posLambda == 0.0) {
            return 0;
        }
        return curand_discrete(localState, posPoisson);
    }
    __device__ unsigned int sampleNegPoisson(int x, int y, G* localState) {
        if (negLambda == 0.0) {
            return 0;
        }
        return curand_discrete(localState, negPoisson);
    }
    __device__ void put(int id, G localState) {
        states[id] = localState;
    }
};

