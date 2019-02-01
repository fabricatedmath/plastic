#pragma once

template <class T>
int printSetBlockGridStats(T func, int numThreads) {
    const char *argv[] = {""};
    
    int argc = 0;
    int device = findCudaDevice(argc, argv);
    cudaDeviceProp prop = { 0 };
    gpuErrchk( cudaSetDevice(device) );
    gpuErrchk( cudaGetDeviceProperties(&prop, device) );

//    int numThreads = 128;
    int maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    int numSms = prop.multiProcessorCount;
    int numBlocksPerSm = 0;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, func, numThreads, 0));
    int numBlocks = numBlocksPerSm * numSms;
    
    cout << "--------Grid/Block Statistics-----------------------------------------" << endl;
    cout << endl;
    cout << "\t\tGlobally" << endl;
    printf("\tSMs:\t\t\t%d\n", numSms);
    printf("\tMax Blocks:\t\t%d\n", maxBlocks);
    cout << endl;
    cout << "\t\tProgram" << endl;
    printf("\tBlocks per SM:\t\t%d\n", numBlocksPerSm);
    printf("\tBlocks:\t\t\t%d\n", numBlocks);
    printf("\tThreads per Block:\t%d\n", numThreads);
    printf("\tThreads:\t\t%d\n", numThreads * numBlocks);
    cout << endl;
    cout << "----------------------------------------------------------------------" << endl;

    return numBlocks;
}

void printNvidiaSmi() {
    std::cout << endl;
    std::cout << "-----------------Nvidia-SMI-----------------------" << endl;
    std::cout << endl;
    std::cout << "PID: " << ::getpid() << endl;
    std::system("nvidia-smi --query-compute-apps=pid,used_memory,used_gpu_memory --format=csv,noheader > /tmp/stats.txt");
    std::cout << std::ifstream("/tmp/stats.txt").rdbuf();
    std::cout << endl;
    std::cout << "--------------------------------------------------" << endl;
    std::cout << endl;
}
