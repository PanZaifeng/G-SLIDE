#include "ReluLayer.h"
#include "utils.h"
#include "kernel.h"


void ReluLayer::forward(const Layer &prev_layer, const int batch_size,
                        const int thread_num, const int max_act_num)
{
    const int block_num =
        (node_num * batch_size + thread_num - 1) / thread_num;
    get_dense_acts_knl<<<block_num, thread_num>>>(
        this->c_acts.d_nodes, this->c_acts.d_cols,
        this->node_num, batch_size);
    
    const int smem_size =
        (sizeof(int) + sizeof(float)) * (thread_num + max_act_num);
    relu_fwd_knl<<<batch_size, thread_num, smem_size>>>(
        prev_layer.c_acts.d_vals, this->d_weights, this->d_biases,
        prev_layer.c_acts.d_nodes, prev_layer.c_acts.d_cols,
        this->c_acts.d_nodes, this->c_acts.d_cols, this->node_num,
        max_act_num, this->c_acts.d_vals);
}

void ReluLayer::forward(const CompactActNodes &c_inputs, const int batch_size,
                        const int thread_num, const int max_act_num)
{
    const int block_num =
        (node_num * batch_size + thread_num - 1) / thread_num;
    get_dense_acts_knl<<<block_num, thread_num>>>(
        this->c_acts.d_nodes, this->c_acts.d_cols,
        this->node_num, batch_size);
    
    const int smem_size =
        (sizeof(int) + sizeof(float)) * (thread_num + max_act_num);
    relu_fwd_knl<<<batch_size, thread_num, smem_size>>>(
        c_inputs.d_vals, this->d_weights, this->d_biases,
        c_inputs.d_nodes, c_inputs.d_cols, this->c_acts.d_nodes,
        this->c_acts.d_cols, this->node_num, max_act_num,
        this->c_acts.d_vals);
}

void ReluLayer::bp(Layer &prev_layer, const int batch_size,
                   const int thread_num, const int max_act_num)
{
    const int smem_size =
        (sizeof(int) + sizeof(float)) * max_act_num;
    bp_knl<<<batch_size, thread_num, smem_size>>>(
        this->d_weights, prev_layer.c_acts.d_vals, this->d_c_bp_deltas,
        this->c_acts.d_nodes, this->c_acts.d_cols,
        prev_layer.c_acts.d_nodes, prev_layer.c_acts.d_cols,
        this->node_num, max_act_num, prev_layer.d_c_bp_deltas,
        this->weight_adam.d_ts, this->bias_adam.d_ts);
}

void ReluLayer::bp(const CompactActNodes &c_inputs, const int batch_size,
                   const int thread_num, const int max_act_num)
{
    const int smem_size =
        (sizeof(int) + sizeof(float)) * max_act_num;
    bp_first_layer_knl<<<batch_size, thread_num, smem_size>>>(
        c_inputs.d_vals, this->d_c_bp_deltas,
        this->c_acts.d_nodes, this->c_acts.d_cols,
        c_inputs.d_nodes, c_inputs.d_cols,
        this->node_num, max_act_num, this->weight_adam.d_ts,
        this->bias_adam.d_ts);
}
