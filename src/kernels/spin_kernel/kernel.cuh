#pragma once

__global__ void spin_kernel(unsigned long long seconds) {
    unsigned long long startTime = clock64();
    unsigned long long hz = 2100000000;
    unsigned long long thresh = hz * seconds;
    do {
    } while ((clock64() - startTime) < thresh);
}
