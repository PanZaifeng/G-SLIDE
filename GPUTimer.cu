#include "GPUTimer.h"


GPUTimer::GPUTimer() { 
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
}

GPUTimer::~GPUTimer() {
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

void GPUTimer::start() {
    cudaEventRecord(start_time);
}

// return the elapsed tiem from start_time to end_time
float GPUTimer::record(std::string msg) {
    cudaDeviceSynchronize();
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(&elapsed_time, start_time, end_time);

    if (msg != "")
        std::cout << msg << elapsed_time << " ms" << std::endl;
    
    // restart timer
    cudaEventRecord(start_time);

    return elapsed_time;
}
