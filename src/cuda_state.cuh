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

    CudaVectorXf v;
    CudaVectorXf vprev;
    CudaVectorXf vthresh;
    CudaVectorXf vlongtrace;
    CudaVectorXf vpos;
    CudaVectorXf vneg;

    CudaVectorXf xplastLat;
    CudaVectorXf xplastFF;

    CudaVectorXf wadap;
    CudaVectorXf z;

    CudaVectorXi isSpiking;
};

cudaError_t cudaMalloc(MutableState* s, CudaMutableState* cs) {
    errRet( cudaMalloc(&s->w,&cs->w) );
    errRet( cudaMalloc(&s->wff,&cs->wff) );
    errRet( cudaMalloc(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( cudaMalloc(&s->firings,&cs->firings) );
    
    errRet( cudaMalloc(&s->v,&cs->v) );
    errRet( cudaMalloc(&s->vprev,&cs->vprev) );
    errRet( cudaMalloc(&s->vthresh,&cs->vthresh) );
    errRet( cudaMalloc(&s->vlongtrace,&cs->vlongtrace) );
    errRet( cudaMalloc(&s->vpos,&cs->vpos) );
    errRet( cudaMalloc(&s->vneg,&cs->vneg) );
    
    errRet( cudaMalloc(&s->xplastLat,&cs->xplastLat) );
    errRet( cudaMalloc(&s->xplastFF,&cs->xplastFF) );
    
    errRet( cudaMalloc(&s->wadap,&cs->wadap) );
    errRet( cudaMalloc(&s->z,&cs->z) );
    
    errRet( cudaMalloc(&s->isSpiking,&cs->isSpiking) );
    
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(MutableState* s, CudaMutableState* cs) {
    errRet( memcpyHostToDevice(&s->w,&cs->w) );
    errRet( memcpyHostToDevice(&s->wff,&cs->wff) );
    errRet( memcpyHostToDevice(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( memcpyHostToDevice(&s->firings,&cs->firings) );

    errRet( memcpyHostToDevice(&s->v,&cs->v) );
    errRet( memcpyHostToDevice(&s->vprev,&cs->vprev) );
    errRet( memcpyHostToDevice(&s->vthresh,&cs->vthresh) );
    errRet( memcpyHostToDevice(&s->vlongtrace,&cs->vlongtrace) );
    errRet( memcpyHostToDevice(&s->vpos,&cs->vpos) );
    errRet( memcpyHostToDevice(&s->vneg,&cs->vneg) );
    
    errRet( memcpyHostToDevice(&s->xplastLat,&cs->xplastLat) );
    errRet( memcpyHostToDevice(&s->xplastFF,&cs->xplastFF) );
    
    errRet( memcpyHostToDevice(&s->wadap,&cs->wadap) );
    errRet( memcpyHostToDevice(&s->z,&cs->z) );
    
    errRet( memcpyHostToDevice(&s->isSpiking,&cs->isSpiking) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(MutableState* s, CudaMutableState* cs) {
    errRet( memcpyDeviceToHost(&s->w,&cs->w) );
    errRet( memcpyDeviceToHost(&s->wff,&cs->wff) );
    errRet( memcpyDeviceToHost(&s->incomingSpikes,&cs->incomingSpikes) );
    errRet( memcpyDeviceToHost(&s->firings,&cs->firings) );

    errRet( memcpyDeviceToHost(&s->v,&cs->v) );
    errRet( memcpyDeviceToHost(&s->vprev,&cs->vprev) );
    errRet( memcpyDeviceToHost(&s->vthresh,&cs->vthresh) );
    errRet( memcpyDeviceToHost(&s->vlongtrace,&cs->vlongtrace) );
    errRet( memcpyDeviceToHost(&s->vpos,&cs->vpos) );
    errRet( memcpyDeviceToHost(&s->vneg,&cs->vneg) );
    
    errRet( memcpyDeviceToHost(&s->xplastLat,&cs->xplastLat) );
    errRet( memcpyDeviceToHost(&s->xplastFF,&cs->xplastFF) );
    
    errRet( memcpyDeviceToHost(&s->wadap,&cs->wadap) );
    errRet( memcpyDeviceToHost(&s->z,&cs->z) );
    
    errRet( memcpyDeviceToHost(&s->isSpiking,&cs->isSpiking) );
    return cudaSuccess;
}

/* CudaStaticState */
struct CudaStaticState {
    CudaMatrixXf input;
    CudaMatrixXi delays;
    CudaVectorXf altds;
};

cudaError_t cudaMalloc(StaticState* s, CudaStaticState* cs) {
    errRet( cudaMalloc(&s->input,&cs->input) );
    errRet( cudaMalloc(&s->delays,&cs->delays) );
    errRet( cudaMalloc(&s->altds,&cs->altds) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(StaticState* s, CudaStaticState* cs) {
    errRet( memcpyHostToDevice(&s->input,&cs->input) );
    errRet( memcpyHostToDevice(&s->delays,&cs->delays) );
    errRet( memcpyHostToDevice(&s->altds,&cs->altds) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(StaticState* s, CudaStaticState* cs) {
    errRet( memcpyDeviceToHost(&s->input,&cs->input) );
    errRet( memcpyDeviceToHost(&s->delays,&cs->delays) );
    errRet( memcpyDeviceToHost(&s->altds,&cs->altds) );
    return cudaSuccess;
}

/* CudaBuffers */
struct CudaBuffers {
    //TODO make lgnfirings an int matrix
    CudaMatrixXf lgnfirings;
    CudaMatrixXi poissonNoise;
    CudaVectorXf neuronInputs;
    CudaVectorXf eachNeurLTD;
    CudaVectorXf eachNeurLTP;
};

cudaError_t cudaMalloc(Buffers* s, CudaBuffers* cs) {
    errRet( cudaMalloc(&s->lgnfirings,&cs->lgnfirings) );
    errRet( cudaMalloc(&s->poissonNoise,&cs->poissonNoise) );
    errRet( cudaMalloc(&s->neuronInputs,&cs->neuronInputs) );
    errRet( cudaMalloc(&s->eachNeurLTD,&cs->eachNeurLTD) );
    errRet( cudaMalloc(&s->eachNeurLTP,&cs->eachNeurLTP) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(Buffers* s, CudaBuffers* cs) {
    errRet( memcpyHostToDevice(&s->lgnfirings,&cs->lgnfirings) );
    errRet( memcpyHostToDevice(&s->poissonNoise,&cs->poissonNoise) );
    errRet( memcpyHostToDevice(&s->neuronInputs,&cs->neuronInputs) );
    errRet( memcpyHostToDevice(&s->eachNeurLTD,&cs->eachNeurLTD) );
    errRet( memcpyHostToDevice(&s->eachNeurLTP,&cs->eachNeurLTP) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(Buffers* s, CudaBuffers* cs) {
    errRet( memcpyDeviceToHost(&s->lgnfirings,&cs->lgnfirings) );
    errRet( memcpyDeviceToHost(&s->poissonNoise,&cs->poissonNoise) );
    errRet( memcpyDeviceToHost(&s->neuronInputs,&cs->neuronInputs) );
    errRet( memcpyDeviceToHost(&s->eachNeurLTD,&cs->eachNeurLTD) );
    errRet( memcpyDeviceToHost(&s->eachNeurLTP,&cs->eachNeurLTP) );
    return cudaSuccess;
}
