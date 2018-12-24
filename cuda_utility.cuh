#pragma once

#include <vector>
#include "err.cuh"

void something2() {
    char array[] = "dogs";
    char *p = array;
    const char *arg = p;
    const char *argv[] = {"dogs"};
    
    int argc = 0;
    int device = findCudaDevice(argc, argv);
    cudaDeviceProp prop = { 0 };
    cout << gpuGetMaxGflopsDeviceId() << endl;
    gpuErrchk( cudaSetDevice(device) );
    gpuErrchk( cudaGetDeviceProperties(&prop, device) );
    cout << prop.multiProcessorCount << endl;
    cout << prop.maxThreadsPerMultiProcessor << endl;
    cout << prop.maxThreadsPerBlock << endl;
    cout << endl;
    
    
}
