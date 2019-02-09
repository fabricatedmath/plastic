#pragma once

#include <Eigen/Dense>
#include "state.h"

using namespace Eigen;

template<typename T>
struct CudaVectorX {
    T* data;
};

template<typename T>
cudaError_t cudaMalloc(VectorX<T>* v, CudaVectorX<T>* cv) {
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(T));
}

template<typename T>
cudaError_t memcpyHostToDevice(VectorX<T>* v, CudaVectorX<T>* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t memcpyDeviceToHost(VectorX<T>* v, CudaVectorX<T>* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
struct CudaMatrixX {
    T* data;
    size_t pitch;

    __device__ T* getRowPtr(int row) {
        return (T*)((char*)data + row*pitch);
    }    
};

template<typename T>
cudaError_t cudaMalloc(MatrixRX<T>* m, CudaMatrixX<T>* cm) {
   return cudaMallocPitch((void**)&cm->data, &cm->pitch, m->cols() * sizeof(T), m->rows());
}

template<typename T>
cudaError_t memcpyHostToDevice(MatrixRX<T>* m, CudaMatrixX<T>* cm) {
    return cudaMemcpy2D(cm->data, cm->pitch, m->data(), m->cols() * sizeof(T), m->cols() * sizeof(T), m->rows(), cudaMemcpyHostToDevice);
}

template<typename T>
cudaError_t memcpyDeviceToHost(MatrixRX<T>* m, CudaMatrixX<T>* cm) {
    return cudaMemcpy2D(m->data(), m->cols() * sizeof(T), cm->data, cm->pitch, m->cols() * sizeof(T), m->rows(), cudaMemcpyDeviceToHost);
}

template<typename F, typename I>
struct CudaMutableState {
    CudaMatrixX<F> w;
    CudaMatrixX<F> wff;

    CudaVectorX<F> vthresh;
    CudaVectorX<F> vlongtrace;
    CudaVectorX<F> vpos;
    CudaVectorX<F> vneg;

    CudaVectorX<F> xplastLat;
    CudaVectorX<F> xplastFF;

    CudaVectorX<F> wadap;
    CudaVectorX<F> z;
};

template<typename F, typename I>
cudaError_t cudaMalloc(MutableState<F,I>* s, CudaMutableState<F,I>* cs) {
    errRet( cudaMalloc(&s->w,&cs->w) );
    errRet( cudaMalloc(&s->wff,&cs->wff) );
    
    errRet( cudaMalloc(&s->vthresh,&cs->vthresh) );
    errRet( cudaMalloc(&s->vlongtrace,&cs->vlongtrace) );
    errRet( cudaMalloc(&s->vpos,&cs->vpos) );
    errRet( cudaMalloc(&s->vneg,&cs->vneg) );
    
    errRet( cudaMalloc(&s->xplastLat,&cs->xplastLat) );
    errRet( cudaMalloc(&s->xplastFF,&cs->xplastFF) );
    
    errRet( cudaMalloc(&s->wadap,&cs->wadap) );
    errRet( cudaMalloc(&s->z,&cs->z) );
    
    return cudaSuccess;
}

template<typename F, typename I>
cudaError_t memcpyHostToDevice(MutableState<F,I>* s, CudaMutableState<F,I>* cs) {
    errRet( memcpyHostToDevice(&s->w,&cs->w) );
    errRet( memcpyHostToDevice(&s->wff,&cs->wff) );

    errRet( memcpyHostToDevice(&s->vthresh,&cs->vthresh) );
    errRet( memcpyHostToDevice(&s->vlongtrace,&cs->vlongtrace) );
    errRet( memcpyHostToDevice(&s->vpos,&cs->vpos) );
    errRet( memcpyHostToDevice(&s->vneg,&cs->vneg) );
    
    errRet( memcpyHostToDevice(&s->xplastLat,&cs->xplastLat) );
    errRet( memcpyHostToDevice(&s->xplastFF,&cs->xplastFF) );
    
    errRet( memcpyHostToDevice(&s->wadap,&cs->wadap) );
    errRet( memcpyHostToDevice(&s->z,&cs->z) );
    
    return cudaSuccess;
}

template<typename F, typename I>
cudaError_t memcpyDeviceToHost(MutableState<F,I>* s, CudaMutableState<F,I>* cs) {
    errRet( memcpyDeviceToHost(&s->w,&cs->w) );
    errRet( memcpyDeviceToHost(&s->wff,&cs->wff) );

    errRet( memcpyDeviceToHost(&s->vthresh,&cs->vthresh) );
    errRet( memcpyDeviceToHost(&s->vlongtrace,&cs->vlongtrace) );
    errRet( memcpyDeviceToHost(&s->vpos,&cs->vpos) );
    errRet( memcpyDeviceToHost(&s->vneg,&cs->vneg) );
    
    errRet( memcpyDeviceToHost(&s->xplastLat,&cs->xplastLat) );
    errRet( memcpyDeviceToHost(&s->xplastFF,&cs->xplastFF) );
    
    errRet( memcpyDeviceToHost(&s->wadap,&cs->wadap) );
    errRet( memcpyDeviceToHost(&s->z,&cs->z) );

    return cudaSuccess;
}

template<typename F, typename I>
struct CudaStaticState {
    CudaMatrixX<F> input;
    CudaMatrixX<I> delays;
    CudaVectorX<F> altds;
};

template<typename F, typename I>
cudaError_t cudaMalloc(StaticState<F,I>* s, CudaStaticState<F,I>* cs) {
    errRet( cudaMalloc(&s->input,&cs->input) );
    errRet( cudaMalloc(&s->delays,&cs->delays) );
    errRet( cudaMalloc(&s->altds,&cs->altds) );
    return cudaSuccess;
}

template<typename F, typename I>
cudaError_t memcpyHostToDevice(StaticState<F,I>* s, CudaStaticState<F,I>* cs) {
    errRet( memcpyHostToDevice(&s->input,&cs->input) );
    errRet( memcpyHostToDevice(&s->delays,&cs->delays) );
    errRet( memcpyHostToDevice(&s->altds,&cs->altds) );
    return cudaSuccess;
}

template<typename F, typename I>
cudaError_t memcpyDeviceToHost(StaticState<F,I>* s, CudaStaticState<F,I>* cs) {
    errRet( memcpyDeviceToHost(&s->input,&cs->input) );
    errRet( memcpyDeviceToHost(&s->delays,&cs->delays) );
    errRet( memcpyDeviceToHost(&s->altds,&cs->altds) );
    return cudaSuccess;
}

template<typename F, typename I>
struct CudaBuffers {
    CudaMatrixX<I> incomingSpikes;
    CudaVectorX<I> firings;
    
    CudaMatrixX<I> lgnfirings;
    CudaMatrixX<I> poissonNoise;
    CudaVectorX<F> neuronInputs;
    
    CudaVectorX<F> eachNeurLTD;
    CudaVectorX<F> eachNeurLTP;
};

template<typename F, typename I>
cudaError_t cudaMalloc(Buffers<F,I>* s, CudaBuffers<F,I>* cs) {
    errRet( cudaMalloc(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( cudaMalloc(&s->firings,&cs->firings) );
    errRet( cudaMalloc(&s->lgnfirings,&cs->lgnfirings) );
    errRet( cudaMalloc(&s->poissonNoise,&cs->poissonNoise) );
    errRet( cudaMalloc(&s->neuronInputs,&cs->neuronInputs) );
    errRet( cudaMalloc(&s->eachNeurLTD,&cs->eachNeurLTD) );
    errRet( cudaMalloc(&s->eachNeurLTP,&cs->eachNeurLTP) );
    return cudaSuccess;
}

template<typename F, typename I>
cudaError_t memcpyHostToDevice(Buffers<F,I>* s, CudaBuffers<F,I>* cs) {
    errRet( memcpyHostToDevice(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( memcpyHostToDevice(&s->firings,&cs->firings) );
    errRet( memcpyHostToDevice(&s->lgnfirings,&cs->lgnfirings) );
    errRet( memcpyHostToDevice(&s->poissonNoise,&cs->poissonNoise) );
    errRet( memcpyHostToDevice(&s->neuronInputs,&cs->neuronInputs) );
    errRet( memcpyHostToDevice(&s->eachNeurLTD,&cs->eachNeurLTD) );
    errRet( memcpyHostToDevice(&s->eachNeurLTP,&cs->eachNeurLTP) );
    return cudaSuccess;
}

template<typename F, typename I>
cudaError_t memcpyDeviceToHost(Buffers<F,I>* s, CudaBuffers<F,I>* cs) {
    errRet( memcpyDeviceToHost(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( memcpyDeviceToHost(&s->firings,&cs->firings) );
    errRet( memcpyDeviceToHost(&s->lgnfirings,&cs->lgnfirings) );
    errRet( memcpyDeviceToHost(&s->poissonNoise,&cs->poissonNoise) );
    errRet( memcpyDeviceToHost(&s->neuronInputs,&cs->neuronInputs) );
    errRet( memcpyDeviceToHost(&s->eachNeurLTD,&cs->eachNeurLTD) );
    errRet( memcpyDeviceToHost(&s->eachNeurLTP,&cs->eachNeurLTP) );
    return cudaSuccess;
}
