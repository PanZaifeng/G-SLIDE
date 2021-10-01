#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cassert>

// need to check idx
#define FOR_IDX_SYNC(idx, begin, end)                                \
  for (int idx = (begin) + threadIdx.x; (idx - threadIdx.x) < (end); \
       idx += blockDim.x)

#define FOR_IDX_ASYNC(idx, begin, end) \
  for (int idx = (begin) + threadIdx.x; idx < (end); idx += blockDim.x)

#define FOR_OFFSET(offset, begin, end) \
  for (int offset = (begin); offset < (end); offset += blockDim.x)

#define CUDA_CHECK(ans) cuda_assert((ans), __FILE__, __LINE__)
static inline void cuda_assert(cudaError_t code, const char *file, int line,
                               bool abort_flag = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA assert: " << cudaGetErrorString(code) << " at file "
              << file << " line " << line << std::endl;

    if (abort_flag) exit(code);
  }
}

#define CUBLAS_CHECK(ans) cublas_assert((ans), __FILE__, __LINE__)
static inline void cublas_assert(cublasStatus_t code, const char *file,
                                 int line, bool abort_flag = true) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS assert: " << code << " at file " << file << " line "
              << line << std::endl;

    if (abort_flag) exit(code);
  }
}
