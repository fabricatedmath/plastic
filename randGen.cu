#include "randGen.cuh"

template <typename T, typename G>
__global__ void kernel(T stuff, float *arr, unsigned long long* time) {
    unsigned long long startTime = clock();
    float v = arr[threadIdx.x];
    arr[threadIdx.x] = v + 1.0;
    for (int j = 0; j < 10; j++) {
        G g = stuff.get(threadIdx.x);
        for (int i = 0; i < 1; i++) {
            printf("pos %f\n",stuff.samplePosPoisson(0,&g));
            printf("neg %f\n",stuff.sampleNegPoisson(0,&g));
        }
        stuff.put(threadIdx.x, g);
    }
    unsigned long long endTime = clock();
    *time = endTime - startTime;
}

int main(int argc, char *argv[]) {

    int SIZE = 20;

    const int Dg = 1;
    const int Db = 2;

    const double posLambda = 1.8;
    const double negLambda = 0.1;
    
    bool testing = true;
    
    unsigned long long time;
    unsigned long long* d_time;

    gpuErrchk( cudaMalloc(&d_time, sizeof(unsigned long long)) );

    float* h_arr;
    float* d_arr;
    gpuErrchk( cudaMalloc((void**)&d_arr, 10*sizeof(float)) );

    if(testing) {
        RandomGenHistorical h_rgen;
        h_rgen.arrUniform = (float*)malloc(SIZE * sizeof(float));
        RandomGenHistorical d_rgen;
        gpuErrchk( cudaMalloc((void**)&d_rgen.arrUniform, SIZE * sizeof(float)) );
        gpuErrchk( cudaMalloc((void**)&d_rgen.arrPosPoisson, SIZE * sizeof(float)) );
        gpuErrchk( cudaMalloc((void**)&d_rgen.arrNegPoisson, SIZE * sizeof(float)) );
        kernel<RandomGenHistorical, void*><<<Dg, Db>>>(d_rgen,d_arr,d_time);
    } else {
        typedef RandomGen<curandState,Dg, Db> Rgen;
        Rgen d_rgen(posLambda, negLambda);
        kernel<Rgen,curandState><<<Dg, Db>>>(d_rgen,d_arr,d_time);
    }

    gpuErrchk( cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
    cout << time << endl;
    return 0;
}
