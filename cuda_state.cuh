#pragma once

#include <Eigen/Dense>
#include "state.h"

using namespace Eigen;

/* CudaVectorXf */
struct CudaVectorXf {
    float* data;
};

cudaError_t cudaMalloc(VectorXf* v, CudaVectorXf* cv) {
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(float));
}

cudaError_t memcpyHostToDevice(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(float), cudaMemcpyDeviceToHost);
}

/* CudaVectorXi */
struct CudaVectorXi {
    int* data;
};

cudaError_t cudaMalloc(VectorXi* v, CudaVectorXi* cv) {
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(int));
}

cudaError_t memcpyHostToDevice(VectorXi* v, CudaVectorXi* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(int), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(VectorXi* v, CudaVectorXi* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(int), cudaMemcpyDeviceToHost);
}

/* CudaMatrixXf */
struct CudaMatrixXf {
    float* data;
    size_t pitch;
};

__device__ float* getRowPtr(CudaMatrixXf cm, int row) {
    return (float*)((char*)cm.data + row*cm.pitch);
}

cudaError_t cudaMalloc(MatrixRXf* m, CudaMatrixXf* cm) {
   return cudaMallocPitch((void**)&cm->data, &cm->pitch, m->cols() * sizeof(float), m->rows());
}

cudaError_t memcpyHostToDevice(MatrixRXf* m, CudaMatrixXf* cm) {
    return cudaMemcpy2D(cm->data, cm->pitch, m->data(), m->cols() * sizeof(float), m->cols() * sizeof(float), m->rows(), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(MatrixRXf* m, CudaMatrixXf* cm) {
    return cudaMemcpy2D(m->data(), m->cols() * sizeof(float), cm->data, cm->pitch, m->cols() * sizeof(float), m->rows(), cudaMemcpyDeviceToHost);
}

/* CudaMatrixXi */
struct CudaMatrixXi {
    int* data;
    size_t pitch;
};

__device__ int* getRowPtr(CudaMatrixXi cm, int row) {
    return (int*)((char*)cm.data + row*cm.pitch);
}

cudaError_t cudaMalloc(MatrixRXi* m, CudaMatrixXi* cm) {
   return cudaMallocPitch((void**)&cm->data, &cm->pitch, m->cols() * sizeof(int), m->rows());
}

cudaError_t memcpyHostToDevice(MatrixRXi* m, CudaMatrixXi* cm) {
    return cudaMemcpy2D(cm->data, cm->pitch, m->data(), m->cols() * sizeof(int), m->cols() * sizeof(int), m->rows(), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(MatrixRXi* m, CudaMatrixXi* cm) {
    return cudaMemcpy2D(m->data(), m->cols() * sizeof(int), cm->data, cm->pitch, m->cols() * sizeof(int), m->rows(), cudaMemcpyDeviceToHost);
}

/* CudaMutableState */
struct CudaMutableState {
    CudaMatrixXf w;
    CudaMatrixXf wff;
    CudaMatrixXi incomingSpikes;
    CudaVectorXi firings;
};

cudaError_t cudaMalloc(MutableState* s, CudaMutableState* cs) {
    errRet( cudaMalloc(&s->w,&cs->w) );
    errRet( cudaMalloc(&s->wff,&cs->wff) );
    errRet( cudaMalloc(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( cudaMalloc(&s->firings,&cs->firings) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(MutableState* s, CudaMutableState* cs) {
    errRet( memcpyHostToDevice(&s->w,&cs->w) );
    errRet( memcpyHostToDevice(&s->wff,&cs->wff) );
    errRet( memcpyHostToDevice(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( memcpyHostToDevice(&s->firings,&cs->firings) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(MutableState* s, CudaMutableState* cs) {
    errRet( memcpyDeviceToHost(&s->w,&cs->w) );
    errRet( memcpyDeviceToHost(&s->wff,&cs->wff) );
    errRet( memcpyDeviceToHost(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( memcpyDeviceToHost(&s->firings,&cs->firings) );
    return cudaSuccess;
}

/* CudaStaticState */
struct CudaStaticState {
    CudaMatrixXf input;
    CudaMatrixXi delays;
};

cudaError_t cudaMalloc(StaticState* s, CudaStaticState* cs) {
    errRet( cudaMalloc(&s->input,&cs->input) );
    errRet( cudaMalloc(&s->delays,&cs->delays) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(StaticState* s, CudaStaticState* cs) {
    errRet( memcpyHostToDevice(&s->input,&cs->input) );
    errRet( memcpyHostToDevice(&s->delays,&cs->delays) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(StaticState* s, CudaStaticState* cs) {
    errRet( memcpyDeviceToHost(&s->input,&cs->input) );
    errRet( memcpyDeviceToHost(&s->delays,&cs->delays) );
    return cudaSuccess;
}

/* CudaBuffers */
struct CudaBuffers {
    CudaMatrixXf lgnfirings;
    CudaVectorXf neuronInputs;
};

cudaError_t cudaMalloc(Buffers* s, CudaBuffers* cs) {
    errRet( cudaMalloc(&s->lgnfirings,&cs->lgnfirings) );
    errRet( cudaMalloc(&s->neuronInputs,&cs->neuronInputs) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(Buffers* s, CudaBuffers* cs) {
    errRet( memcpyHostToDevice(&s->lgnfirings,&cs->lgnfirings) );
    errRet( memcpyHostToDevice(&s->neuronInputs,&cs->neuronInputs) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(Buffers* s, CudaBuffers* cs) {
    errRet( memcpyDeviceToHost(&s->lgnfirings,&cs->lgnfirings) );
    errRet( memcpyDeviceToHost(&s->neuronInputs,&cs->neuronInputs) );
    return cudaSuccess;
}
