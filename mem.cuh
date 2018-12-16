#pragma once
#include <cuda.h>
#include <Eigen/Dense>
#include "type.cuh"
#include "test.h"
#include "err.cuh"

using namespace Eigen;

cudaError_t cudaMalloc(VectorXf* v, CudaVectorXf* cv) {
    return cudaMalloc((void**)&cv->data, v->size() * sizeof(float));
}

cudaError_t memcpyHostToDevice(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)cv->data, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(VectorXf* v, CudaVectorXf* cv) {
    return cudaMemcpy((void**)v->data(), cv->data, v->size() * sizeof(float), cudaMemcpyDeviceToHost);
}

cudaError_t cudaMalloc(MatrixXf* m, CudaMatrixXf* cm) {
   return cudaMallocPitch((void**)&cm->data, &cm->pitch, m->cols() * sizeof(float), m->rows());
}

cudaError_t memcpyHostToDevice(MatrixXf* m, CudaMatrixXf* cm) {
    return cudaMemcpy2D(cm->data, cm->pitch, m->data(), m->cols() * sizeof(float), m->cols() * sizeof(float), m->rows(), cudaMemcpyHostToDevice);
}

cudaError_t memcpyDeviceToHost(MatrixXf* m, CudaMatrixXf* cm) {
    return cudaMemcpy2D(m->data(), m->cols() * sizeof(float), cm->data, cm->pitch, m->cols() * sizeof(float), m->rows(), cudaMemcpyDeviceToHost);
}

cudaError_t cudaMalloc(Test* t, CudaTest* ct) {
    errRet( cudaMalloc(&t->m,&ct->m) );
    errRet( cudaMalloc(&t->v,&ct->v) );
    return cudaSuccess;
}

cudaError_t memcpyHostToDevice(Test* t, CudaTest* ct) {
    errRet( memcpyHostToDevice(&t->m,&ct->m) );
    errRet( memcpyHostToDevice(&t->v,&ct->v) );
    return cudaSuccess;
}

cudaError_t memcpyDeviceToHost(Test* t, CudaTest* ct) {
    errRet( memcpyDeviceToHost(&t->m,&ct->m) );
    errRet( memcpyDeviceToHost(&t->v,&ct->v) );
    return cudaSuccess;
}

__device__ float* getRowPtr(CudaMatrixXf cm, int row) {
    return (float*)((char*)cm.data + row*cm.pitch);
}

