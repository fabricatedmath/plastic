#pragma once

__device__ float* getRowPtr(CudaMatrixXf cm, int row) {
    return (float*)((char*)cm.data + row*cm.pitch);
}


