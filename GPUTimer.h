#pragma once

#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class GPUTimer {
    cudaEvent_t start_time;
    cudaEvent_t end_time;

    float elapsed_time;

public:
    GPUTimer();

    ~GPUTimer();

    void start();

    // return the elapsed tiem from start_time to end_time
    float record(std::string msg = "");
};