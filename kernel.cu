#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include <cassert>
#include <cfloat>
#include <cstdio>

#include "kernel.h"
#include "utils.h"

#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001

#define MAX_INIT 0.0

__forceinline__ __device__ float warp_reduce(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__forceinline__ __device__ float block_reduce(float val) {
  __shared__ float s_sum_buff[32];

  const int wid = threadIdx.x / warpSize;
  const int lane = threadIdx.x - wid * warpSize;

  val = warp_reduce(val);
  if (blockDim.x < warpSize) return val;

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

__forceinline__ __device__ float warp_max(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}

__forceinline__ __device__ float block_max(float val) {
  __shared__ float s_max_buff[32];

  const int wid = threadIdx.x / warpSize;
  const int lane = threadIdx.x - wid * warpSize;

  val = warp_max(val);
  if (blockDim.x < warpSize) return val;

  if (lane == 0) {
    s_max_buff[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < blockDim.x / warpSize) ? s_max_buff[lane] : MAX_INIT;
    val = warp_max(val);
  }

  return val;
}

__global__ void get_dense_acts_knl(CscActNodes csc_acts, const int node_num,
                                   const int batch_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < node_num * batch_size) {
    const int node_id = tid % node_num;
    csc_acts.d_nodes[tid] = node_id;

    if (tid <= batch_size) {
      csc_acts.d_offsets[tid] = tid * node_num;
    }
  }
}

__global__ void relu_fwd_slide_in_knl(const CscActNodes csc_inputs,
                                      const float *d_weights_colmajor,
                                      const float *d_biases,
                                      const int weight_row_num,
                                      const int max_out_num,
                                      CscActNodes csc_outputs) {
  extern __shared__ char smem[];
  float *s_in_vals = (float *)smem;                        // blockDim.x
  int *s_in_nodes = (int *)(s_in_vals + blockDim.x);       // blockDim.x
  float *s_out_vals = (float *)(s_in_nodes + blockDim.x);  // max_out_num
  int *s_out_nodes = (int *)(s_out_vals + max_out_num);    // max_out_num

  const int in_begin = csc_inputs.d_offsets[blockIdx.x];
  const int in_end = csc_inputs.d_offsets[blockIdx.x + 1];
  // const int in_size = in_end - in_begin;
  const int out_begin = csc_outputs.d_offsets[blockIdx.x];
  const int out_end = csc_outputs.d_offsets[blockIdx.x + 1];
  const int out_size = out_end - out_begin;

  assert(out_size <= max_out_num);

  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    const int out_node = csc_outputs.d_nodes[out_begin + s_out_idx];
    s_out_nodes[s_out_idx] = out_node;
    s_out_vals[s_out_idx] = d_biases[out_node];
  }
  // __syncthreads();

  FOR_OFFSET(in_offset, in_begin, in_end) {
    const int in_idx = in_offset + threadIdx.x;
    if (in_idx < in_end) {
      s_in_nodes[threadIdx.x] = csc_inputs.d_nodes[in_idx];
      s_in_vals[threadIdx.x] = csc_inputs.d_vals[in_idx];
    }
    __syncthreads();

    FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
      const int out_node = s_out_nodes[s_out_idx];
      float psum = 0.;
      for (int s_in_idx = 0;
           s_in_idx < blockDim.x && in_offset + s_in_idx < in_end; ++s_in_idx) {
        const int in_node = s_in_nodes[s_in_idx];
        const float in_val = s_in_vals[s_in_idx];
        const float weight =
            d_weights_colmajor[in_node * weight_row_num + out_node];
        psum += in_val * weight;
      }
      s_out_vals[s_out_idx] += psum;
    }
    __syncthreads();
  }

  float *d_out_val_col = csc_outputs.d_vals + out_begin;
  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    d_out_val_col[s_out_idx] = max(0., s_out_vals[s_out_idx]);
  }
}

__global__ void softmax_fwd_bp_rowmajor_slide_in_knl(
    const CscActNodes csc_inputs, const float *d_weights_rowmajor,
    const float *d_biases, const CscActNodes cmprs_labels,
    const int weight_col_num, const int max_out_num, const int max_label_num,
    CscActNodes csc_outputs, float *d_cmprs_bp_deltas) {
  extern __shared__ char smem[];
  float *s_in_vals = (float *)smem;                        // blockDim.x
  int *s_in_nodes = (int *)(s_in_vals + blockDim.x);       // blockDim.x
  float *s_out_vals = (float *)(s_in_nodes + blockDim.x);  // max_out_num
  int *s_out_nodes = (int *)(s_out_vals + max_out_num);    // max_out_num
  int *s_labels = (int *)(s_out_nodes + max_out_num);      // max_label_num

  const int in_begin = csc_inputs.d_offsets[blockIdx.x];
  const int in_end = csc_inputs.d_offsets[blockIdx.x + 1];
  const int out_begin = csc_outputs.d_offsets[blockIdx.x];
  const int out_end = csc_outputs.d_offsets[blockIdx.x + 1];
  const int out_size = out_end - out_begin;
  const int label_begin = cmprs_labels.d_offsets[blockIdx.x];
  const int label_end = cmprs_labels.d_offsets[blockIdx.x + 1];
  const int label_size = label_end - label_begin;

  assert(out_size <= max_out_num);
  assert(label_size <= max_label_num);

  FOR_IDX_ASYNC(s_label_idx, 0, label_size) {
    s_labels[s_label_idx] = cmprs_labels.d_nodes[label_begin + s_label_idx];
  }

  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    const int out_node = csc_outputs.d_nodes[out_begin + s_out_idx];
    s_out_nodes[s_out_idx] = out_node;
    s_out_vals[s_out_idx] = d_biases[out_node];
  }

  FOR_OFFSET(in_offset, in_begin, in_end) {
    const int in_idx = in_offset + threadIdx.x;
    if (in_idx < in_end) {
      s_in_nodes[threadIdx.x] = csc_inputs.d_nodes[in_idx];
      s_in_vals[threadIdx.x] = csc_inputs.d_vals[in_idx];
    }
    __syncthreads();

    FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
      const int out_node = s_out_nodes[s_out_idx];
      float psum = 0.;
      for (int s_in_idx = 0;
           s_in_idx < blockDim.x && in_offset + s_in_idx < in_end; ++s_in_idx) {
        const int in_node = s_in_nodes[s_in_idx];
        const float in_val = s_in_vals[s_in_idx];
        const float weight =
            d_weights_rowmajor[out_node * weight_col_num + in_node];
        psum += in_val * weight;
      }
      s_out_vals[s_out_idx] += psum;
    }
    __syncthreads();
  }

  __shared__ float s_max;
  float thread_max = MAX_INIT;
  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    thread_max = max(thread_max, s_out_vals[s_out_idx]);
  }

  thread_max = block_max(thread_max);
  if (threadIdx.x == 0) s_max = thread_max;
  __syncthreads();

  __shared__ float s_sum;
  float thread_sum = 0.;
  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    float val = __expf(s_out_vals[s_out_idx] - s_max);
    // float val = exp(s_out_vals[s_out_idx] - s_max);
    s_out_vals[s_out_idx] = val;
    thread_sum += val;
  }

  thread_sum = block_reduce(thread_sum);
  if (threadIdx.x == 0) s_sum = thread_sum;
  __syncthreads();

  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    const int out_idx = s_out_idx + out_begin;
    const float val = s_out_vals[s_out_idx] / (s_sum + EPS);
    const int out_node = s_out_nodes[s_out_idx];
    csc_outputs.d_vals[out_idx] = val;

    bool is_in_label = false;
    for (int i = 0; i < label_size; ++i) {
      is_in_label = is_in_label || (s_labels[i] == out_node);
    }

    float bp_delta = -val;
    if (is_in_label) bp_delta += 1.0 / label_size;
    bp_delta /= gridDim.x;
    d_cmprs_bp_deltas[out_idx] = bp_delta;
  }
}

__global__ void softmax_fwd_bp_rowmajor_slide_out_knl(
    const CscActNodes csc_inputs, const float *d_weights_rowmajor,
    const float *d_biases, const CscActNodes cmprs_labels,
    const int weight_col_num, const int max_in_num, const int max_label_num,
    CscActNodes csc_outputs, float *d_cmprs_bp_deltas) {
  extern __shared__ char smem[];
  float *s_in_vals = (float *)smem;                   // max_in_num
  int *s_in_nodes = (int *)(s_in_vals + max_in_num);  // max_in_num
  int *s_labels = (int *)(s_in_nodes + max_in_num);   // max_label_num

  const int in_begin = csc_inputs.d_offsets[blockIdx.x];
  const int in_end = csc_inputs.d_offsets[blockIdx.x + 1];
  const int in_size = in_end - in_begin;
  const int out_begin = csc_outputs.d_offsets[blockIdx.x];
  const int out_end = csc_outputs.d_offsets[blockIdx.x + 1];
  const int label_begin = cmprs_labels.d_offsets[blockIdx.x];
  const int label_end = cmprs_labels.d_offsets[blockIdx.x + 1];
  const int label_size = label_end - label_begin;

  assert(in_size <= max_in_num);
  assert(label_size <= max_label_num);

  FOR_IDX_ASYNC(in_idx, in_begin, in_end) {
    const int s_in_idx = in_idx - in_begin;
    s_in_nodes[s_in_idx] = csc_inputs.d_nodes[in_idx];
    s_in_vals[s_in_idx] = csc_inputs.d_vals[in_idx];
  }

  FOR_IDX_ASYNC(s_label_idx, 0, label_size) {
    s_labels[s_label_idx] = cmprs_labels.d_nodes[label_begin + s_label_idx];
  }
  __syncthreads();

  float thread_max = MAX_INIT;
  FOR_IDX_ASYNC(out_idx, out_begin, out_end) {
    const int out_node = csc_outputs.d_nodes[out_idx];
    float psum = d_biases[out_node];
    for (int s_in_idx = 0; s_in_idx < in_size; ++s_in_idx) {
      const int in_node = s_in_nodes[s_in_idx];
      const float in_val = s_in_vals[s_in_idx];
      const float weight =
          d_weights_rowmajor[out_node * weight_col_num + in_node];
      psum += in_val * weight;
    }
    csc_outputs.d_vals[out_idx] = psum;
    thread_max = max(thread_max, psum);
  }

  __shared__ float s_max;
  thread_max = block_max(thread_max);
  if (threadIdx.x == 0) s_max = thread_max;
  __syncthreads();

  __shared__ float s_sum;
  float thread_sum = 0.;
  FOR_IDX_ASYNC(out_idx, out_begin, out_end) {
    float val = __expf(csc_outputs.d_vals[out_idx] - s_max);
    // float val = exp(csc_outputs.d_vals[out_idx] - s_max);
    csc_outputs.d_vals[out_idx] = val;
    thread_sum += val;
  }

  thread_sum = block_reduce(thread_sum);
  if (threadIdx.x == 0) s_sum = thread_sum;
  __syncthreads();

  FOR_IDX_ASYNC(out_idx, out_begin, out_end) {
    const float val = csc_outputs.d_vals[out_idx] / (s_sum + EPS);
    const int out_node = csc_outputs.d_nodes[out_idx];
    csc_outputs.d_vals[out_idx] = val;

    bool is_in_label = false;
    for (int i = 0; i < label_size; ++i) {
      is_in_label = is_in_label || (s_labels[i] == out_node);
    }

    float bp_delta = -val;
    if (is_in_label) bp_delta += 1.0 / label_size;
    bp_delta /= gridDim.x;
    d_cmprs_bp_deltas[out_idx] = bp_delta;
  }
}

__global__ void softmax_fwd_bp_rowmajor_all_sm_knl(
    const CscActNodes csc_inputs, const float *d_weights_rowmajor,
    const float *d_biases, const CscActNodes cmprs_labels,
    const int weight_col_num, const int max_in_num, const int max_out_num,
    const int max_label_num, CscActNodes csc_outputs,
    float *d_cmprs_bp_deltas) {
  extern __shared__ char smem[];
  float *s_in_vals = (float *)smem;                        // max_in_num
  int *s_in_nodes = (int *)(s_in_vals + max_in_num);       // max_in_num
  float *s_out_vals = (float *)(s_in_nodes + max_in_num);  // max_out_num
  int *s_out_nodes = (int *)(s_out_vals + max_out_num);    // max_out_num
  int *s_labels = (int *)(s_out_nodes + max_out_num);      // max_label_num

  const int in_begin = csc_inputs.d_offsets[blockIdx.x];
  const int in_end = csc_inputs.d_offsets[blockIdx.x + 1];
  const int in_size = in_end - in_begin;
  const int out_begin = csc_outputs.d_offsets[blockIdx.x];
  const int out_end = csc_outputs.d_offsets[blockIdx.x + 1];
  const int out_size = out_end - out_begin;
  const int label_begin = cmprs_labels.d_offsets[blockIdx.x];
  const int label_end = cmprs_labels.d_offsets[blockIdx.x + 1];
  const int label_size = label_end - label_begin;

  assert(in_size <= max_in_num);
  assert(out_size <= max_out_num);
  assert(label_size <= max_label_num);

  FOR_IDX_ASYNC(s_label_idx, 0, label_size) {
    s_labels[s_label_idx] = cmprs_labels.d_nodes[label_begin + s_label_idx];
  }

  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    const int out_node = csc_outputs.d_nodes[out_begin + s_out_idx];
    s_out_nodes[s_out_idx] = out_node;
    s_out_vals[s_out_idx] = d_biases[out_node];
  }

  FOR_IDX_ASYNC(in_idx, in_begin, in_end) {
    const int s_in_idx = in_idx - in_begin;
    s_in_nodes[s_in_idx] = csc_inputs.d_nodes[in_idx];
    s_in_vals[s_in_idx] = csc_inputs.d_vals[in_idx];
  }
  __syncthreads();

  float thread_max = MAX_INIT;
  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    const int out_node = csc_outputs.d_nodes[out_begin + s_out_idx];
    s_out_nodes[s_out_idx] = out_node;
    float psum = d_biases[out_node];
    for (int s_in_idx = 0; s_in_idx < in_size; ++s_in_idx) {
      const int in_node = s_in_nodes[s_in_idx];
      const float in_val = s_in_vals[s_in_idx];
      const float weight =
          d_weights_rowmajor[out_node * weight_col_num + in_node];
      psum += in_val * weight;
    }
    s_out_vals[s_out_idx] = psum;
    thread_max = max(thread_max, psum);
  }

  __shared__ float s_max;
  thread_max = block_max(thread_max);
  if (threadIdx.x == 0) s_max = thread_max;
  __syncthreads();

  __shared__ float s_sum;
  float thread_sum = 0.;
  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    float val = __expf(s_out_vals[s_out_idx] - s_max);
    // float val = exp(s_out_vals[s_out_idx] - s_max);
    s_out_vals[s_out_idx] = val;
    thread_sum += val;
  }

  thread_sum = block_reduce(thread_sum);
  if (threadIdx.x == 0) s_sum = thread_sum;
  __syncthreads();

  FOR_IDX_ASYNC(s_out_idx, 0, out_size) {
    const int out_idx = s_out_idx + out_begin;
    const float val = s_out_vals[s_out_idx] / (s_sum + EPS);
    const int out_node = s_out_nodes[s_out_idx];
    csc_outputs.d_vals[out_idx] = val;

    bool is_in_label = false;
    for (int i = 0; i < label_size; ++i) {
      is_in_label = is_in_label || (s_labels[i] == out_node);
    }

    float bp_delta = -val;
    if (is_in_label) bp_delta += 1.0 / label_size;
    bp_delta /= gridDim.x;
    d_cmprs_bp_deltas[out_idx] = bp_delta;
  }
}

__global__ void bp_knl(const CscActNodes csc_acts, const CscActNodes csc_prev,
                       const float *d_weights_colmajor,
                       const float *d_cmprs_bp_deltas, const int weight_row_num,
                       const int max_act_num, float *d_cmprs_prev_bp_deltas,
                       float *d_adam_ts, float *d_bias_adam_ts) {
  extern __shared__ char smem[];
  float *s_bp_deltas = (float *)smem;                     // max_act_num
  int *s_act_nodes = (int *)(s_bp_deltas + max_act_num);  // max_act_num

  const int act_begin = csc_acts.d_offsets[blockIdx.x];
  const int act_end = csc_acts.d_offsets[blockIdx.x + 1];
  const int act_size = act_end - act_begin;
  const int prev_begin = csc_prev.d_offsets[blockIdx.x];
  const int prev_end = csc_prev.d_offsets[blockIdx.x + 1];

  assert(act_size <= max_act_num);

  FOR_IDX_ASYNC(act_idx, act_begin, act_end) {
    const int act_node = csc_acts.d_nodes[act_idx];
    const float bp_delta = d_cmprs_bp_deltas[act_idx];
    const int s_act_idx = act_idx - act_begin;
    s_act_nodes[s_act_idx] = act_node;
    s_bp_deltas[s_act_idx] = bp_delta;
    atomicAdd(d_bias_adam_ts + act_node, bp_delta);
  }
  __syncthreads();

  FOR_IDX_ASYNC(prev_idx, prev_begin, prev_end) {
    const int prev_node = csc_prev.d_nodes[prev_idx];
    const float prev_val = csc_prev.d_vals[prev_idx];
    float prev_bp_delta = 0.;
    for (int s_act_idx = 0; s_act_idx < act_size; ++s_act_idx) {
      const int act_node = s_act_nodes[s_act_idx];
      const int weight_idx = prev_node * weight_row_num + act_node;
      const float bp_delta = s_bp_deltas[s_act_idx];
      if (prev_val > 0) {
        const float weight = d_weights_colmajor[weight_idx];
        prev_bp_delta += bp_delta * weight;
      }
      atomicAdd(d_adam_ts + weight_idx, prev_val * bp_delta);
    }

    if (prev_val > 0) {
      prev_bp_delta += d_cmprs_prev_bp_deltas[prev_idx];
    }
    d_cmprs_prev_bp_deltas[prev_idx] = prev_bp_delta;
  }
}

__global__ void bp_rowmajor_knl(const CscActNodes csc_acts,
                                const CscActNodes csc_prev,
                                const float *d_weights_rowmajor,
                                const float *d_cmprs_bp_deltas,
                                const int weight_col_num, const int max_act_num,
                                float *d_cmprs_prev_bp_deltas, float *d_adam_ts,
                                float *d_bias_adam_ts) {
  extern __shared__ char smem[];
  float *s_bp_deltas = (float *)smem;                     // max_act_num
  int *s_act_nodes = (int *)(s_bp_deltas + max_act_num);  // max_act_num

  const int act_begin = csc_acts.d_offsets[blockIdx.x];
  const int act_end = csc_acts.d_offsets[blockIdx.x + 1];
  const int act_size = act_end - act_begin;
  const int prev_begin = csc_prev.d_offsets[blockIdx.x];
  const int prev_end = csc_prev.d_offsets[blockIdx.x + 1];

  assert(act_size <= max_act_num);

  FOR_IDX_ASYNC(act_idx, act_begin, act_end) {
    const int act_node = csc_acts.d_nodes[act_idx];
    const float bp_delta = d_cmprs_bp_deltas[act_idx];
    const int s_act_idx = act_idx - act_begin;
    s_act_nodes[s_act_idx] = act_node;
    s_bp_deltas[s_act_idx] = bp_delta;
    atomicAdd(d_bias_adam_ts + act_node, bp_delta);
  }
  __syncthreads();

  FOR_IDX_ASYNC(prev_idx, prev_begin, prev_end) {
    const int prev_node = csc_prev.d_nodes[prev_idx];
    const float prev_val = csc_prev.d_vals[prev_idx];
    float prev_bp_delta = 0.;
    for (int s_act_idx = 0; s_act_idx < act_size; ++s_act_idx) {
      const int act_node = s_act_nodes[s_act_idx];
      const int weight_idx = act_node * weight_col_num + prev_node;
      const float bp_delta = s_bp_deltas[s_act_idx];
      if (prev_val > 0) {
        const float weight = d_weights_rowmajor[weight_idx];
        prev_bp_delta += bp_delta * weight;
      }
      atomicAdd(d_adam_ts + weight_idx, prev_val * bp_delta);
    }

    if (prev_val > 0) {
      prev_bp_delta += d_cmprs_prev_bp_deltas[prev_idx];
    }
    d_cmprs_prev_bp_deltas[prev_idx] = prev_bp_delta;
  }
}

__global__ void bp_rowmajor_slide_knl(
    const CscActNodes csc_acts, const CscActNodes csc_prev,
    const float *d_weights_rowmajor, const float *d_cmprs_bp_deltas,
    const int weight_col_num, const int max_prev_num,
    float *d_cmprs_prev_bp_deltas, float *d_adam_ts, float *d_bias_adam_ts) {
  extern __shared__ char smem[];
  float *s_prev_bp_deltas = (float *)smem;                       // max_prev_num
  int *s_prev_nodes = (int *)(s_prev_bp_deltas + max_prev_num);  // max_prev_num
  float *s_prev_vals = (float *)(s_prev_nodes + max_prev_num);   // max_prev_num

  const int act_begin = csc_acts.d_offsets[blockIdx.x];
  const int act_end = csc_acts.d_offsets[blockIdx.x + 1];
  const int prev_begin = csc_prev.d_offsets[blockIdx.x];
  const int prev_end = csc_prev.d_offsets[blockIdx.x + 1];
  const int prev_size = prev_end - prev_begin;

  assert(prev_size <= max_prev_num);

  FOR_IDX_ASYNC(s_prev_idx, 0, prev_size) {
    const int prev_idx = s_prev_idx + prev_begin;
    const float prev_val = csc_prev.d_vals[prev_idx];
    s_prev_nodes[s_prev_idx] = csc_prev.d_nodes[prev_idx];
    s_prev_vals[s_prev_idx] = prev_val;
    s_prev_bp_deltas[s_prev_idx] =
        prev_val > 0 ? d_cmprs_prev_bp_deltas[prev_idx] : 0;
  }
  __syncthreads();

  FOR_IDX_SYNC(act_idx, act_begin, act_end) {
    int act_node;
    float bp_delta;
    if (act_idx < act_end) {
      act_node = csc_acts.d_nodes[act_idx];
      bp_delta = d_cmprs_bp_deltas[act_idx];
      atomicAdd(d_bias_adam_ts + act_node, bp_delta);
    }

    // TODO: better utilize the bandwidth
    for (int s_prev_idx = 0; s_prev_idx < prev_size; ++s_prev_idx) {
      int prev_node, weight_idx;
      float prev_val, thread_inc = 0;
      if (act_idx < act_end) {
        prev_node = s_prev_nodes[s_prev_idx];
        prev_val = s_prev_vals[s_prev_idx];
        weight_idx = act_node * weight_col_num + prev_node;
        if (prev_val > 0) {
          const float weight = d_weights_rowmajor[weight_idx];
          thread_inc = bp_delta * weight;
        }
      }
      thread_inc = block_reduce(thread_inc);
      if (threadIdx.x == 0) s_prev_bp_deltas[s_prev_idx] += thread_inc;
      if (act_idx < act_end) {
        atomicAdd(d_adam_ts + weight_idx, prev_val * bp_delta);
      }
    }
  }
  __syncthreads();

  FOR_IDX_ASYNC(prev_idx, prev_begin, prev_end) {
    d_cmprs_prev_bp_deltas[prev_idx] = s_prev_bp_deltas[prev_idx - prev_begin];
  }
}

__global__ void bp_first_layer_knl(const CscActNodes csc_acts,
                                   const CscActNodes csc_prev,
                                   const float *d_cmprs_bp_deltas,
                                   const int weight_row_num,
                                   const int max_act_num, float *d_adam_ts,
                                   float *d_bias_adam_ts) {
  extern __shared__ char smem[];
  float *s_bp_deltas = (float *)smem;                     // max_act_num
  int *s_act_nodes = (int *)(s_bp_deltas + max_act_num);  // max_act_num

  const int act_begin = csc_acts.d_offsets[blockIdx.x];
  const int act_end = csc_acts.d_offsets[blockIdx.x + 1];
  const int act_size = act_end - act_begin;
  const int prev_begin = csc_prev.d_offsets[blockIdx.x];
  const int prev_end = csc_prev.d_offsets[blockIdx.x + 1];

  assert(act_size <= max_act_num);

  FOR_IDX_ASYNC(act_idx, act_begin, act_end) {
    const int act_node = csc_acts.d_nodes[act_idx];
    const float bp_delta = d_cmprs_bp_deltas[act_idx];
    const int s_act_idx = act_idx - act_begin;
    s_act_nodes[s_act_idx] = act_node;
    s_bp_deltas[s_act_idx] = bp_delta;
    atomicAdd(d_bias_adam_ts + act_node, bp_delta);
  }
  __syncthreads();

  FOR_IDX_ASYNC(prev_idx, prev_begin, prev_end) {
    const int prev_node = csc_prev.d_nodes[prev_idx];
    const float prev_val = csc_prev.d_vals[prev_idx];
    for (int s_act_idx = 0; s_act_idx < act_size; ++s_act_idx) {
      const int act_node = s_act_nodes[s_act_idx];
      const int weight_idx = prev_node * weight_row_num + act_node;
      const float bp_delta = s_bp_deltas[s_act_idx];
      atomicAdd(d_adam_ts + weight_idx, prev_val * bp_delta);
    }
  }
}

__global__ void update_weights_knl(float *d_weights, float *d_adam_ts,
                                   float *d_adam_moms, float *d_adam_vels,
                                   const float lr, const int weight_size) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= weight_size) return;

  // const float t = d_adam_ts[idx];
  // d_adam_ts[idx] = 0;
  const float t = atomicExch(d_adam_ts + idx, 0);

  float mom = d_adam_moms[idx];
  d_adam_moms[idx] = mom = BETA1 * mom + (1 - BETA1) * t;

  float vel = d_adam_vels[idx];
  d_adam_vels[idx] = vel = BETA2 * vel + (1 - BETA2) * t * t;

  // d_weights[idx] += lr * mom / (sqrtf(vel) + EPS);
  // atomicAdd(d_weights + idx, lr * mom / (sqrtf(vel) + EPS));

  // d_weights[idx] += __fdividef(lr * mom, sqrtf(vel) + EPS);
  atomicAdd(d_weights + idx, __fdividef(lr * mom, sqrtf(vel) + EPS));
}
