#include "GPUTimer.h"
#include "utils.h"
#include <cstdio>

GPUTimer::GPUTimer() {
  CUDA_CHECK(cudaEventCreate(&start_time));
  CUDA_CHECK(cudaEventCreate(&end_time));
}

GPUTimer::~GPUTimer() {
  CUDA_CHECK(cudaEventDestroy(start_time));
  CUDA_CHECK(cudaEventDestroy(end_time));
}

void GPUTimer::start() {
  CUDA_CHECK(cudaEventRecord(start_time));
}

// return the elapsed tiem from start_time to end_time
float GPUTimer::record(std::string msg) {
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(end_time));
  CUDA_CHECK(cudaEventSynchronize(end_time));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_time, end_time));

  if (msg != "") {
    printf("%s%f ms\n", msg.c_str(), elapsed_time);
  }

  // restart timer
  CUDA_CHECK(cudaEventRecord(start_time));

  return elapsed_time;
}
