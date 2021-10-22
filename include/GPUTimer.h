#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#include "utils.h"

class GPUTimer {
  cudaEvent_t start_time;
  cudaEvent_t end_time;
  float elapsed_time;

 public:
  GPUTimer() {
    CUDA_CHECK(cudaEventCreate(&start_time));
    CUDA_CHECK(cudaEventCreate(&end_time));
  }

  ~GPUTimer() {
    CUDA_CHECK(cudaEventDestroy(start_time));
    CUDA_CHECK(cudaEventDestroy(end_time));
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_time)); }

  // return the elapsed time from start_time to end_time
  float record(std::string msg = "") {
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
};
