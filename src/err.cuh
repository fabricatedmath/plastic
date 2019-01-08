#pragma once

#include <curand.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define errRet(ans) do { if((ans)!=cudaSuccess) { \
    return ans; \
        }} while(0)

#define gpuErrchkCuRand(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(curandStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CURAND_STATUS_SUCCESS) 
   {
      fprintf(stderr,"GPUassert: %s %d\n", file, line);
      if (abort) exit(code);
   }
}

#define errRetCuRand(ans) do { if((ans)!=CURAND_STATUS_SUCCESS) { \
    return ans; \
        }} while(0)

