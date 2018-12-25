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

/* CudaMutableState */
struct CudaMutableState {
    CudaMatrixXf w;
    CudaMatrixXf wff;
};

cudaError_t cudaMalloc(MutableState* s, CudaMutableState* cs) {
    errRet( cudaMalloc(&s->w,&cs->w) );
    errRet( cudaMalloc(&s->wff,&cs->wff) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(MutableState* s, CudaMutableState* cs) {
    errRet( memcpyHostToDevice(&s->w,&cs->w) );
    errRet( memcpyHostToDevice(&s->wff,&cs->wff) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(MutableState* s, CudaMutableState* cs) {
    errRet( memcpyDeviceToHost(&s->w,&cs->w) );
    errRet( memcpyDeviceToHost(&s->wff,&cs->wff) );
    return cudaSuccess;
}

/* CudaStaticState */
struct CudaStaticState {
    CudaMatrixXf input;
};

cudaError_t cudaMalloc(StaticState* s, CudaStaticState* cs) {
    errRet( cudaMalloc(&s->input,&cs->input) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(StaticState* s, CudaStaticState* cs) {
    errRet( memcpyHostToDevice(&s->input,&cs->input) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(StaticState* s, CudaStaticState* cs) {
    errRet( memcpyDeviceToHost(&s->input,&cs->input) );
    return cudaSuccess;
}

/* CudaBuffers */
struct CudaBuffers {
    CudaMatrixXf lgnfirings;
};

cudaError_t cudaMalloc(Buffers* s, CudaBuffers* cs) {
    errRet( cudaMalloc(&s->lgnfirings,&cs->lgnfirings) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(Buffers* s, CudaBuffers* cs) {
    errRet( memcpyHostToDevice(&s->lgnfirings,&cs->lgnfirings) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(Buffers* s, CudaBuffers* cs) {
    errRet( memcpyDeviceToHost(&s->lgnfirings,&cs->lgnfirings) );
    return cudaSuccess;
}
