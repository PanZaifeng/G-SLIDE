#include "kernel.h"
#include <math_constants.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cfloat>
#include <cstdio>

#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001


__forceinline__ __device__
float warp_reduce(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__forceinline__ __device__
float block_reduce(float val) {
    __shared__ float s_sum_buff[32];

    const int wid = threadIdx.x / warpSize;
    const int lane = threadIdx.x - wid * warpSize;

    val = warp_reduce(val);
    if (lane == 0) {
        s_sum_buff[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? s_sum_buff[lane] : 0.;
        val = warp_reduce(val);
    }

    return val;
}

__forceinline__ __device__
float warp_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__forceinline__ __device__
float block_max(float val) {
    __shared__ float s_max_buff[32];

    const int wid = threadIdx.x / warpSize;
    const int lane = threadIdx.x - wid * warpSize;

    val = warp_max(val);
    if (lane == 0) {
        s_max_buff[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? s_max_buff[lane] : 0.;
        val = warp_max(val);
    }

    return val;
}

__global__ void get_dense_acts_knl(int *d_act_nodes,
                                   int *d_act_cols,
                                   const int node_num,
                                   const int batch_size)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < node_num * batch_size) {    
        const int node_id = tid % node_num;
        d_act_nodes[tid] = node_id;

        if (tid <= batch_size) {
            d_act_cols[tid] = tid * node_num;
        }
    }
}

// c d_inputs, d_outputs
__global__ void relu_fwd_knl(const float *d_inputs,
                             const float *d_weights,
                             const float *d_biases,
                             const int *d_act_in,
                             const int *d_act_in_cols,
                             const int *d_act_out,
                             const int *d_act_out_cols,
                             const int out_col_size,
                             const int max_out_num,
                             float *d_outputs) 
{
    extern __shared__ char smem[];
    float *s_inputs = (float *) smem; // blockDim.x
    int *s_act_in = (int *) (s_inputs + blockDim.x); // blockDim.x
    float *s_outputs = (float *) (s_act_in + blockDim.x); // max_out_num
    int *s_act_out = (int *) (s_outputs + max_out_num); // max_out_num

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;

    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_act_out[s_out_idx + act_out_col];
            s_act_out[s_out_idx] = act_out_idx;
            s_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    // __syncthreads();

    for (int offset = act_in_col; offset < act_in_n_col; offset += blockDim.x) {
        int c_idx = offset + tx;
        if (c_idx < act_in_n_col) {
            int act_in_idx = d_act_in[c_idx];
            s_act_in[tx] = act_in_idx;
            s_inputs[tx] = d_inputs[c_idx];
        }
        __syncthreads();

        for (int s_out_offset = 0; s_out_offset < act_out_size; 
            s_out_offset += blockDim.x) {
            const int s_out_idx = s_out_offset + tx;
            if (s_out_idx < act_out_size) {
                const int act_out_idx = s_act_out[s_out_idx];
                const float *d_tmp_weights = d_weights + act_out_idx;
                float psum = 0.;
                for (int i = 0; i < blockDim.x && offset + i < act_in_n_col; i++) {
                    const float weight = d_tmp_weights[s_act_in[i] * out_col_size];
                    psum += s_inputs[i] * weight;
                }
                s_outputs[s_out_idx] += psum;
            }
        }
        __syncthreads();
    }

    float *d_row_outputs = d_outputs + act_out_col;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            d_row_outputs[s_out_idx] = max(0., s_outputs[s_out_idx]);
            // d_row_outputs[s_out_idx] = s_outputs[s_out_idx];
        }
    }
}

// c d_inputs, d_outputs, d_bp_deltas
__global__ void softmax_fwd_knl(const float *d_inputs,
                                const float *d_weights,
                                const float *d_biases,
                                const int *d_act_in,
                                const int *d_act_in_cols,
                                const int *d_act_out,
                                const int *d_act_out_cols,
                                const int *d_labels,
                                const int *d_label_cols,
                                const int out_col_size,
                                const int max_out_num,
                                const int max_label_num,
                                float *d_outputs,
                                float *d_bp_deltas)
{
    extern __shared__ char smem[];
    float *s_inputs = (float *) smem; // blockDim.x
    int *s_act_in = (int *) (s_inputs + blockDim.x); // blockDim.x
    float *s_outputs = (float *) (s_act_in + blockDim.x); // max_out_num
    int *s_act_out = (int *) (s_outputs + max_out_num); // max_out_num
    int *s_labels = (int *) (s_act_out + max_out_num); // max_label_num

    const int row_size = gridDim.x;
    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    // const int wid = tx / warpSize;
    // const int lane = tx - wid * warpSize;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;
    const int label_col = d_label_cols[row_idx];
    const int label_n_col = d_label_cols[row_idx + 1];
    const int label_size = label_n_col - label_col;

    if (tx < label_size) {
        s_labels[tx] = d_labels[tx + label_col];
    }

    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_act_out[s_out_idx + act_out_col];
            s_act_out[s_out_idx] = act_out_idx;
            s_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    // __syncthreads();

    for (int offset = act_in_col; offset < act_in_n_col; offset += blockDim.x) {
        int c_idx = offset + tx;
        if (c_idx < act_in_n_col) {
            int act_in_idx = d_act_in[c_idx];
            s_act_in[tx] = act_in_idx;
            s_inputs[tx] = d_inputs[c_idx];
        }
        __syncthreads();

        for (int s_out_offset = 0; s_out_offset < act_out_size;
            s_out_offset += blockDim.x) {
            const int s_out_idx = s_out_offset + tx;
            if (s_out_idx < act_out_size) {
                const int act_out_idx = s_act_out[s_out_idx];
                const float *d_tmp_weights = d_weights + act_out_idx;
                float psum = 0.;
                for (int i = 0; i < blockDim.x && offset + i < act_in_n_col; i++) {
                    const float weight = d_tmp_weights[s_act_in[i] * out_col_size];
                    psum += s_inputs[i] * weight;
                }
                s_outputs[s_out_idx] += psum;
            }
        }
        __syncthreads();
    }

    __shared__ float s_max;
    float thread_max = -FLT_MAX;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            thread_max = max(thread_max, s_outputs[s_out_idx]);
        }
    }

    if (blockDim.x <= warpSize) {
        thread_max = warp_max(thread_max);
    } else {
        thread_max = block_max(thread_max);
    }

    if (tx == 0)
        s_max = thread_max;
    __syncthreads();
    
    __shared__ float s_sum;
    float thread_sum = 0.;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            float val = __expf(s_outputs[s_out_idx] - s_max);
            // float val = exp(s_outputs[s_out_idx] - s_max);
            s_outputs[s_out_idx] = val;
            thread_sum += val;
        }
    }

    if (blockDim.x <= warpSize) {
        thread_sum = warp_reduce(thread_sum);
    } else {
        thread_sum = block_reduce(thread_sum);
    }

    if (tx == 0)
        s_sum = thread_sum;
    __syncthreads();

    float *d_row_outputs = d_outputs + act_out_col;
    float *d_row_bp_deltas = d_bp_deltas + act_out_col;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const float activation = s_outputs[s_out_idx] / s_sum;
            const int act_out_idx = s_act_out[s_out_idx];
            d_row_outputs[s_out_idx] = activation;

            bool is_in_label = false;
            for (int i = 0; i < label_size; i++) {
                is_in_label = 
                    is_in_label || (s_labels[i] == act_out_idx);
            }

            float bp_delta = -activation;
            if (is_in_label)
                bp_delta += 1.0 / label_size;
            bp_delta /= row_size;
            d_row_bp_deltas[s_out_idx] = bp_delta;
        }
    }
}

__global__ void softmax_fwd_and_bp_col_major_knl(const float *d_c_inputs,
                                                 const float *d_weights,
                                                 const float *d_biases,
                                                 const int *d_act_in,
                                                 const int *d_act_in_cols,
                                                 const int *d_act_out,
                                                 const int *d_act_out_cols,
                                                 const int *d_labels,
                                                 const int *d_label_cols,
                                                 const int in_col_size,
                                                 const int max_in_num,
                                                 const int max_out_num,
                                                 const int max_label_num,
                                                 float *d_c_outputs,
                                                 float *d_c_bp_deltas)
{
    extern __shared__ char smem[];
    float *s_c_inputs = (float *) smem; // max_in_num
    int *s_act_in = (int *) (s_c_inputs + max_in_num); // max_in_num
    float *s_c_outputs = (float *) (s_act_in + max_in_num); // max_out_num
    int *s_act_out = (int *) (s_c_outputs + max_out_num); // max_out_num
    int *s_labels = (int *) (s_act_out + max_out_num); // max_label_num

    const int row_size = gridDim.x;
    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    // const int wid = tx / warpSize;
    // const int lane = tx - wid * warpSize;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_in_size = act_in_n_col - act_in_col;
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;
    const int label_col = d_label_cols[row_idx];
    const int label_n_col = d_label_cols[row_idx + 1];
    const int label_size = label_n_col - label_col;

    if (tx < label_size) {
        s_labels[tx] = d_labels[tx + label_col];
    }

    const float *d_r_c_inputs = d_c_inputs + act_in_col;
    const int *d_r_act_in = d_act_in + act_in_col;
    for (int s_in_offset = 0; s_in_offset < act_in_size;
        s_in_offset += blockDim.x) {
        const int s_in_idx = s_in_offset + tx;
        if (s_in_idx < act_in_size) {
            s_c_inputs[s_in_idx] = d_r_c_inputs[s_in_idx];
            s_act_in[s_in_idx] = d_r_act_in[s_in_idx];
        }
    }
    // __syncthreads();

    const int *d_r_act_out = d_act_out + act_out_col;
    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_r_act_out[s_out_idx];
            s_act_out[s_out_idx] = act_out_idx;
            s_c_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    __syncthreads();

    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = s_act_out[s_out_idx];
            const float *d_out_weights = d_weights + act_out_idx * in_col_size;
            float psum = 0.;
            for (int s_in_idx = 0; s_in_idx < act_in_size; s_in_idx++) {
                const float weight = d_out_weights[s_act_in[s_in_idx]];
                psum += s_c_inputs[s_in_idx] * weight;
            }
            s_c_outputs[s_out_idx] += psum;
        }
    }
    
    __shared__ float s_max;
    float thread_max = -FLT_MAX;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            thread_max = max(thread_max, s_c_outputs[s_out_idx]);
        }
    }

    if (blockDim.x <= warpSize) {
        thread_max = warp_max(thread_max);
    } else {
        thread_max = block_max(thread_max);
    }

    if (tx == 0)
        s_max = thread_max;
    __syncthreads();

    __shared__ float s_sum;
    float thread_sum = 0.;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            float val = __expf(s_c_outputs[s_out_idx] - s_max);
            // float val = exp(s_c_outputs[s_out_idx] - s_max);
            s_c_outputs[s_out_idx] = val;
            thread_sum += val;
        }
    }

    if (blockDim.x <= warpSize) {
        thread_sum = warp_reduce(thread_sum);
    } else {
        thread_sum = block_reduce(thread_sum);
    }

    if (tx == 0)
        s_sum = thread_sum;
    __syncthreads();

    float *d_r_c_outputs = d_c_outputs + act_out_col;
    float *d_r_bp_deltas = d_c_bp_deltas + act_out_col;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const float activation =
                s_c_outputs[s_out_idx] / (s_sum + 0.0000001);
            const int act_out_idx = s_act_out[s_out_idx];
            d_r_c_outputs[s_out_idx] = activation;

            bool is_in_label = false;
            for (int i = 0; i < label_size; i++) {
                is_in_label = 
                    is_in_label || (s_labels[i] == act_out_idx);
            }

            float bp_delta = -activation;
            if (is_in_label)
                bp_delta += 1.0 / label_size;
            bp_delta /= row_size;
            d_r_bp_deltas[s_out_idx] = bp_delta;
        }
    }
}

__global__ void softmax_fwd_col_major_knl(const float *d_c_inputs,
                                          const float *d_weights,
                                          const float *d_biases,
                                          const int *d_act_in,
                                          const int *d_act_in_cols,
                                          const int *d_act_out,
                                          const int *d_act_out_cols,
                                          const int in_col_size,
                                          const int max_in_num,
                                          const int max_out_num,
                                          float *d_c_outputs)
{
    extern __shared__ char smem[];
    float *s_c_inputs = (float *) smem; // max_in_num
    int *s_act_in = (int *) (s_c_inputs + max_in_num); // max_in_num
    float *s_c_outputs = (float *) (s_act_in + max_in_num); // max_out_num
    int *s_act_out = (int *) (s_c_outputs + max_out_num); // max_out_num

    // const int row_size = gridDim.x;
    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    // const int wid = tx / warpSize;
    // const int lane = tx - wid * warpSize;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_in_size = act_in_n_col - act_in_col;
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;

    const float *d_r_c_inputs = d_c_inputs + act_in_col;
    const int *d_r_act_in = d_act_in + act_in_col;
    for (int s_in_offset = 0; s_in_offset < act_in_size;
        s_in_offset += blockDim.x) {
        const int s_in_idx = s_in_offset + tx;
        if (s_in_idx < act_in_size) {
            s_c_inputs[s_in_idx] = d_r_c_inputs[s_in_idx];
            s_act_in[s_in_idx] = d_r_act_in[s_in_idx];
        }
    }
    // __syncthreads();

    const int *d_r_act_out = d_act_out + act_out_col;
    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_r_act_out[s_out_idx];
            s_act_out[s_out_idx] = act_out_idx;
            s_c_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    __syncthreads();

    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = s_act_out[s_out_idx];
            const float *d_out_weights = d_weights + act_out_idx * in_col_size;
            float psum = 0.;
            for (int s_in_idx = 0; s_in_idx < act_in_size; s_in_idx++) {
                const float weight = d_out_weights[s_act_in[s_in_idx]];
                psum += s_c_inputs[s_in_idx] * weight;
            }
            s_c_outputs[s_out_idx] += psum;
        }
    }
    
    __shared__ float s_max;
    float thread_max = -FLT_MAX;
    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            thread_max = max(thread_max, s_c_outputs[s_out_idx]);
        }
    }

    if (blockDim.x <= warpSize) {
        thread_max = warp_max(thread_max);
    } else {
        thread_max = block_max(thread_max);
    }

    if (tx == 0)
        s_max = thread_max;
    __syncthreads();

    __shared__ float s_sum;
    float thread_sum = 0.;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            float val = __expf(s_c_outputs[s_out_idx] - s_max);
            // float val = exp(s_c_outputs[s_out_idx] - s_max);
            s_c_outputs[s_out_idx] = val;
            thread_sum += val;
        }
    }

    if (blockDim.x <= warpSize) {
        thread_sum = warp_reduce(thread_sum);
    } else {
        thread_sum = block_reduce(thread_sum);
    }

    if (tx == 0)
        s_sum = thread_sum;
    __syncthreads();

    float *d_r_c_outputs = d_c_outputs + act_out_col;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const float activation = s_c_outputs[s_out_idx] / s_sum;
            // const int act_out_idx = s_act_out[s_out_idx];
            d_r_c_outputs[s_out_idx] = activation;
        }
    }
}

__global__ void softmax_fwd_col_major_no_sm_knl(const float *d_c_inputs,
                                                const float *d_weights,
                                                const float *d_biases,
                                                const int *d_act_in,
                                                const int *d_act_in_cols,
                                                const int *d_act_out,
                                                const int *d_act_out_cols,
                                                const int in_col_size,
                                                float *d_c_outputs)
{
    // const int row_size = gridDim.x;
    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    // const int wid = tx / warpSize;
    // const int lane = tx - wid * warpSize;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_in_size = act_in_n_col - act_in_col;
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;

    const float *d_r_c_inputs = d_c_inputs + act_in_col;
    const int *d_r_act_in = d_act_in + act_in_col;
    const int *d_r_act_out = d_act_out + act_out_col;
    float *d_r_c_outputs = d_c_outputs + act_out_col;

    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_r_act_out[s_out_idx];
            d_r_c_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    __syncthreads();

    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_r_act_out[s_out_idx];
            const float *d_out_weights = d_weights + act_out_idx * in_col_size;
            float psum = 0.;
            for (int s_in_idx = 0; s_in_idx < act_in_size; s_in_idx++) {
                const float weight = d_out_weights[d_r_act_in[s_in_idx]];
                psum += d_r_c_inputs[s_in_idx] * weight;
            }
            d_r_c_outputs[s_out_idx] += psum;
        }
    }
    
    __shared__ float s_max;
    float thread_max = -FLT_MAX;
    for (int s_out_offset = 0; s_out_offset < act_out_size;
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            thread_max = max(thread_max, d_r_c_outputs[s_out_idx]);
        }
    }

    if (blockDim.x <= warpSize) {
        thread_max = warp_max(thread_max);
    } else {
        thread_max = block_max(thread_max);
    }

    if (tx == 0)
        s_max = thread_max;
    __syncthreads();

    __shared__ float s_sum;
    float thread_sum = 0.;
    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            float val = __expf(d_r_c_outputs[s_out_idx] - s_max);
            // float val = exp(s_c_outputs[s_out_idx] - s_max);
            d_r_c_outputs[s_out_idx] = val;
            thread_sum += val;
        }
    }

    if (blockDim.x <= warpSize) {
        thread_sum = warp_reduce(thread_sum);
    } else {
        thread_sum = block_reduce(thread_sum);
    }

    if (tx == 0)
        s_sum = thread_sum;
    __syncthreads();

    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += blockDim.x) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            d_r_c_outputs[s_out_idx] /= s_sum;
        }
    }
}

__global__ void bp_knl(const float *d_weights,
                       const float *d_prev_activations,
                       const float *d_bp_deltas,
                       const int *d_acts,
                       const int *d_act_cols,
                       const int *d_prev_acts,
                       const int *d_prev_act_cols,
                       const int out_col_size,
                       const int max_act_num,
                       float *d_prev_bp_deltas,
                       float *d_adam_ts,
                       float *d_t_biases)
{
    extern __shared__ char smem[];
    float *s_bp_deltas = (float *) smem; // max_act_num
    int *s_acts = (int *) (s_bp_deltas + max_act_num); // max_act_num

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    const int act_col = d_act_cols[row_idx];
    const int act_n_col = d_act_cols[row_idx + 1];
    const int act_size = act_n_col - act_col;
    const int prev_act_col = d_prev_act_cols[row_idx];
    const int prev_act_n_col = d_prev_act_cols[row_idx + 1];
    // const int prev_act_size = prev_act_n_col - prev_act_col;

    for (int s_offset = 0; s_offset < act_size; s_offset += blockDim.x) {
        const int s_idx = s_offset + tx;
        if (s_idx < act_size) {
            const int c_idx = s_idx + act_col;
            const int node_idx = d_acts[c_idx];
            const float bp_delta = d_bp_deltas[c_idx];
            s_bp_deltas[s_idx] = bp_delta;
            s_acts[s_idx] = node_idx;
            // d_t_biases[node_idx] += bp_delta;
            atomicAdd(d_t_biases + node_idx, bp_delta);
        }
    }
    __syncthreads();

    for (int prev_c_offset = prev_act_col; prev_c_offset < prev_act_n_col;
        prev_c_offset += blockDim.x) {
        const int prev_c_idx = prev_c_offset + tx;
        if (prev_c_idx < prev_act_n_col) {
            const int prev_node_idx = d_prev_acts[prev_c_idx];
            const float prev_activation = d_prev_activations[prev_c_idx];
            float prev_bp_delta = 0.;
            for (int s_idx = 0; s_idx < act_size; s_idx++) {
                const int node_idx = s_acts[s_idx];
                const int weight_idx = prev_node_idx * out_col_size + node_idx;
                const float bp_delta = s_bp_deltas[s_idx];
                if (prev_activation > 0) {
                    const float weight = d_weights[weight_idx];
                    prev_bp_delta += bp_delta * weight;
                }
                // d_adam_ts[weight_idx] += prev_activation * bp_delta;
                atomicAdd(d_adam_ts + weight_idx, prev_activation * bp_delta);
            }

            if (prev_activation > 0) {
                prev_bp_delta += d_prev_bp_deltas[prev_c_idx];
            } // else = 0
            d_prev_bp_deltas[prev_c_idx] = prev_bp_delta;
        }
    }
}

__global__ void bp_col_major_knl(const float *d_weights,
                                 const float *d_prev_activations,
                                 const float *d_bp_deltas,
                                 const int *d_acts,
                                 const int *d_act_cols,
                                 const int *d_prev_acts,
                                 const int *d_prev_act_cols,
                                 const int in_col_size,
                                 const int max_act_num,
                                 float *d_prev_bp_deltas,
                                 float *d_adam_ts,
                                 float *d_t_biases)
{
    extern __shared__ char smem[];
    float *s_bp_deltas = (float *) smem; // max_act_num
    int *s_acts = (int *) (s_bp_deltas + max_act_num); // max_act_num

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    const int act_col = d_act_cols[row_idx];
    const int act_n_col = d_act_cols[row_idx + 1];
    const int act_size = act_n_col - act_col;
    const int prev_act_col = d_prev_act_cols[row_idx];
    const int prev_act_n_col = d_prev_act_cols[row_idx + 1];
    // const int prev_act_size = prev_act_n_col - prev_act_col;

    for (int s_offset = 0; s_offset < act_size; s_offset += blockDim.x) {
        const int s_idx = s_offset + tx;
        if (s_idx < act_size) {
            const int c_idx = s_idx + act_col;
            const int node_idx = d_acts[c_idx];
            const float bp_delta = d_bp_deltas[c_idx];
            s_bp_deltas[s_idx] = bp_delta;
            s_acts[s_idx] = node_idx;
            // d_t_biases[node_idx] += bp_delta;
            atomicAdd(d_t_biases + node_idx, bp_delta);
        }
    }
    __syncthreads();

    for (int prev_c_offset = prev_act_col; prev_c_offset < prev_act_n_col;
        prev_c_offset += blockDim.x) {
        const int prev_c_idx = prev_c_offset + tx;
        if (prev_c_idx < prev_act_n_col) {
            const int prev_node_idx = d_prev_acts[prev_c_idx];
            const float prev_activation = d_prev_activations[prev_c_idx];
            float prev_bp_delta = 0.;
            for (int s_idx = 0; s_idx < act_size; s_idx++) {
                const int node_idx = s_acts[s_idx];
                const int weight_idx = node_idx * in_col_size + prev_node_idx;
                const float bp_delta = s_bp_deltas[s_idx];
                if (prev_activation > 0) {
                    const float weight = d_weights[weight_idx];
                    prev_bp_delta += bp_delta * weight;
                }
                // d_adam_ts[weight_idx] += prev_activation * bp_delta;
                atomicAdd(d_adam_ts + weight_idx, prev_activation * bp_delta);
            }

            if (prev_activation > 0) {
                prev_bp_delta += d_prev_bp_deltas[prev_c_idx];
            } // else = 0
            d_prev_bp_deltas[prev_c_idx] = prev_bp_delta;
        }
    }
}


__global__ void bp_first_layer_knl(const float *d_prev_activations,
                                   const float *d_bp_deltas,
                                   const int *d_acts,
                                   const int *d_act_cols,
                                   const int *d_prev_acts,
                                   const int *d_prev_act_cols,
                                   const int out_col_size,
                                   const int max_act_num,
                                   float *d_adam_ts,
                                   float *d_t_biases)
{
    extern __shared__ char smem[];
    float *s_bp_deltas = (float *) smem; // max_act_num
    int *s_acts = (int *) (s_bp_deltas + max_act_num); // max_act_num

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    const int act_col = d_act_cols[row_idx];
    const int act_n_col = d_act_cols[row_idx + 1];
    const int act_size = act_n_col - act_col;
    const int prev_act_col = d_prev_act_cols[row_idx];
    const int prev_act_n_col = d_prev_act_cols[row_idx + 1];
    // const int prev_act_size = prev_act_n_col - prev_act_col;

    for (int s_offset = 0; s_offset < act_size; s_offset += blockDim.x) {
        const int s_idx = s_offset + tx;
        if (s_idx < act_size) {
            const int c_idx = s_idx + act_col;
            const int node_idx = d_acts[c_idx];
            const float bp_delta = d_bp_deltas[c_idx];
            s_bp_deltas[s_idx] = bp_delta;
            s_acts[s_idx] = node_idx;
            // d_t_biases[node_idx] += bp_delta;
            atomicAdd(d_t_biases + node_idx, bp_delta);
        }
    }
    __syncthreads();

    for (int prev_c_offset = prev_act_col; prev_c_offset < prev_act_n_col;
        prev_c_offset += blockDim.x) {
        const int prev_c_idx = prev_c_offset + tx;
        if (prev_c_idx < prev_act_n_col) {
            const int prev_node_idx = d_prev_acts[prev_c_idx];
            const float prev_activation = d_prev_activations[prev_c_idx];
            for (int s_idx = 0; s_idx < act_size; s_idx++) {
                const int node_idx = s_acts[s_idx];
                const int weight_idx = prev_node_idx * out_col_size + node_idx;
                // d_adam_ts[weight_idx] += prev_activation * bp_delta;
                atomicAdd(d_adam_ts + weight_idx, 
                    prev_activation * s_bp_deltas[s_idx]);
            }
        }
    }
}

__global__ void update_weights_knl(float *d_weights,
                                   float *d_adam_ts,
                                   float *d_adam_moms,
                                   float *d_adam_vels,
                                   const float lr,
                                   const int weight_size)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= weight_size)
        return;

    // const float t = d_adam_ts[idx];
    // d_adam_ts[idx] = 0;
    const float t = atomicExch(d_adam_ts + idx, 0);

    float mom = d_adam_moms[idx];
    d_adam_moms[idx] = mom = BETA1 * mom + (1 - BETA1) * t;

    float vel = d_adam_vels[idx];
    d_adam_vels[idx] = vel = BETA2 * vel + (1 - BETA2) * t * t;

    // d_weights[idx] += lr * mom / (sqrtf(vel) + EPS);
    // atomicAdd(d_weights + idx, lr * mom / (sqrtf(vel) + EPS));

    d_weights[idx] += __fdividef(lr * mom, sqrtf(vel) + EPS);
    // atomicAdd(d_weights + idx, __fdividef(lr * mom, sqrtf(vel) + EPS));
}