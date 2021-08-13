#pragma once


__global__ void relu_fwd_knl1(const float *d_inputs,
                             const float *d_weights,
                             const float *d_biases,
                             const int *d_act_in,
                             const int *d_act_in_cols,
                             const int *d_act_out,
                             const int *d_act_out_cols,
                             const int out_col_size,
                             const int thread_num,
                             float *d_outputs);

template<size_t thread_num, size_t max_out_num> 
__global__ void relu_fwd_knl3(const float *d_inputs,
                             const float *d_weights,
                             const float *d_biases,
                             const int *d_act_in,
                             const int *d_act_in_cols,
                             const int *d_act_out,
                             const int *d_act_out_cols,
                             const int out_col_size,
                             float *d_outputs) 
{
    __shared__ float s_inputs[thread_num];
    __shared__ int s_act_in[thread_num];
    __shared__ float s_outputs[max_out_num];
    __shared__ int s_act_out[max_out_num];

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;

    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += thread_num) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_act_out[s_out_idx + act_out_col];
            s_act_out[s_out_idx] = act_out_idx;
            s_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    // __syncthreads();

    for (int offset = act_in_col; offset < act_in_n_col; offset += thread_num) {
        int c_idx = offset + tx;
        if (c_idx < act_in_n_col) {
            int act_in_idx = d_act_in[c_idx];
            s_act_in[tx] = act_in_idx;
            s_inputs[tx] = d_inputs[c_idx];
        }
        __syncthreads();

        for (int s_out_offset = 0; s_out_offset < act_out_size; 
            s_out_offset += thread_num) {
            const int s_out_idx = s_out_offset + tx;
            if (s_out_idx < act_out_size) {
                const int act_out_idx = s_act_out[s_out_idx];
                const float *d_tmp_weights = d_weights + act_out_idx;
                float psum = 0.;
                for (int i = 0; i < thread_num && offset + i < act_in_n_col; i++) {
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
        s_out_offset += thread_num) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            d_row_outputs[s_out_idx] = max(0., s_outputs[s_out_idx]);
            // d_row_outputs[s_out_idx] = s_outputs[s_out_idx];
        }
    }
}



// template<size_t thread_num=128, size_t max_out_num=128> 
template<size_t thread_num, size_t max_out_num> 
// template<int thread_num=128, int max_out_num=128> 
__global__ void relu_fwd_knl2(const float *d_inputs,
                             const float *d_weights,
                             const float *d_biases,
                             const int *d_act_in,
                             const int *d_act_in_cols,
                             const int *d_act_out,
                             const int *d_act_out_cols,
                             const int out_col_size,
                             float *d_outputs) 
{
    __shared__ float s_inputs[thread_num];
    __shared__ int s_act_in[thread_num];
    __shared__ float s_outputs[max_out_num];
    __shared__ int s_act_out[max_out_num];

    // __shared__ float s_inputs[128];
    // __shared__ int s_act_in[128];
    // __shared__ float s_outputs[128];
    // __shared__ int s_act_out[128];

    // const size_t thread_num = 128;

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;

    const int act_in_col = d_act_in_cols[row_idx];
    const int act_in_n_col = d_act_in_cols[row_idx + 1];
    const int act_out_col = d_act_out_cols[row_idx];
    const int act_out_n_col = d_act_out_cols[row_idx + 1];
    const int act_out_size = act_out_n_col - act_out_col;

    for (int s_out_offset = 0; s_out_offset < act_out_size; 
        s_out_offset += thread_num) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            const int act_out_idx = d_act_out[s_out_idx + act_out_col];
            s_act_out[s_out_idx] = act_out_idx;
            s_outputs[s_out_idx] = d_biases[act_out_idx];
        }
    }
    // __syncthreads();

    for (int offset = act_in_col; offset < act_in_n_col; offset += thread_num) {
        int c_idx = offset + tx;
        if (c_idx < act_in_n_col) {
            int act_in_idx = d_act_in[c_idx];
            s_act_in[tx] = act_in_idx;
            s_inputs[tx] = d_inputs[c_idx];
        }
        __syncthreads();

        for (int s_out_offset = 0; s_out_offset < act_out_size; 
            s_out_offset += thread_num) {
            const int s_out_idx = s_out_offset + tx;
            if (s_out_idx < act_out_size) {
                const int act_out_idx = s_act_out[s_out_idx];
                const float *d_tmp_weights = d_weights + act_out_idx;
                float psum = 0.;
                for (int i = 0; i < thread_num && offset + i < act_in_n_col; i++) {
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
        s_out_offset += thread_num) {
        const int s_out_idx = s_out_offset + tx;
        if (s_out_idx < act_out_size) {
            d_row_outputs[s_out_idx] = max(0., s_outputs[s_out_idx]);
            // d_row_outputs[s_out_idx] = s_outputs[s_out_idx];
        }
    }
}



template<size_t thread_num, size_t max_act_num>
__global__ void bp_knl(const float *d_weights,
                       const float *d_prev_activations,
                       const float *d_bp_deltas,
                       const int *d_acts,
                       const int *d_act_cols,
                       const int *d_prev_acts,
                       const int *d_prev_act_cols,
                       const int out_col_size,
                       float *d_prev_bp_deltas,
                       float *d_adam_ts,
                       float *d_t_biases)
{
    __shared__ float s_bp_deltas[max_act_num];
    __shared__ int s_acts[max_act_num];

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    const int act_col = d_act_cols[row_idx];
    const int act_n_col = d_act_cols[row_idx + 1];
    const int act_size = act_n_col - act_col;
    const int prev_act_col = d_prev_act_cols[row_idx];
    const int prev_act_n_col = d_prev_act_cols[row_idx + 1];
    // const int prev_act_size = prev_act_n_col - prev_act_col;

    for (int s_offset = 0; s_offset < act_size; s_offset += thread_num) {
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
        prev_c_offset += thread_num) {
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

template<size_t thread_num, size_t max_act_num>
__global__ void bp_col_major_knl(const float *d_weights,
                                 const float *d_prev_activations,
                                 const float *d_bp_deltas,
                                 const int *d_acts,
                                 const int *d_act_cols,
                                 const int *d_prev_acts,
                                 const int *d_prev_act_cols,
                                 const int in_col_size,
                                 float *d_prev_bp_deltas,
                                 float *d_adam_ts,
                                 float *d_t_biases)
{
    __shared__ float s_bp_deltas[max_act_num];
    __shared__ int s_acts[max_act_num];

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    const int act_col = d_act_cols[row_idx];
    const int act_n_col = d_act_cols[row_idx + 1];
    const int act_size = act_n_col - act_col;
    const int prev_act_col = d_prev_act_cols[row_idx];
    const int prev_act_n_col = d_prev_act_cols[row_idx + 1];
    // const int prev_act_size = prev_act_n_col - prev_act_col;

    for (int s_offset = 0; s_offset < act_size; s_offset += thread_num) {
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
        prev_c_offset += thread_num) {
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

template<size_t thread_num, size_t max_act_num>
__global__ void bp_first_layer_knl(const float *d_prev_activations,
                                   const float *d_bp_deltas,
                                   const int *d_acts,
                                   const int *d_act_cols,
                                   const int *d_prev_acts,
                                   const int *d_prev_act_cols,
                                   const int out_col_size,
                                   float *d_adam_ts,
                                   float *d_t_biases)
{
    __shared__ float s_bp_deltas[max_act_num];
    __shared__ int s_acts[max_act_num];

    const int row_idx = blockIdx.x;
    const int tx = threadIdx.x;
    const int act_col = d_act_cols[row_idx];
    const int act_n_col = d_act_cols[row_idx + 1];
    const int act_size = act_n_col - act_col;
    const int prev_act_col = d_prev_act_cols[row_idx];
    const int prev_act_n_col = d_prev_act_cols[row_idx + 1];
    // const int prev_act_size = prev_act_n_col - prev_act_col;

    for (int s_offset = 0; s_offset < act_size; s_offset += thread_num) {
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
        prev_c_offset += thread_num) {
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



__global__ void get_dense_acts_knl(int *d_act_nodes,
                                   int *d_act_cols,
                                   const int node_num,
                                   const int batch_size);

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
                             float *d_outputs);

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
                                float *d_bp_deltas);

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
                                                 float *d_c_bp_deltas);

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
                                          float *d_c_outputs);

__global__ void softmax_fwd_col_major_no_sm_knl(const float *d_c_inputs,
                                                const float *d_weights,
                                                const float *d_biases,
                                                const int *d_act_in,
                                                const int *d_act_in_cols,
                                                const int *d_act_out,
                                                const int *d_act_out_cols,
                                                const int in_col_size,
                                                float *d_c_outputs);

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
                       float *d_t_biases);

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
                                 float *d_t_biases);

__global__ void bp_first_layer_knl(const float *d_prev_activations,
                                   const float *d_bp_deltas,
                                   const int *d_acts,
                                   const int *d_act_cols,
                                   const int *d_prev_acts,
                                   const int *d_prev_act_cols,
                                   const int out_col_size,
                                   const int max_act_num,
                                   float *d_adam_ts,
                                   float *d_t_biases);

__global__ void update_weights_knl(float *d_weights,
                                   float *d_adam_ts,
                                   float *d_adam_moms,
                                   float *d_adam_vels,
                                   const float lr,
                                   const int weight_size);

