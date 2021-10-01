#include "ReluLayer.h"
#include "kernel.h"
#include "utils.h"

void ReluLayer::forward(const Layer &prev_layer, const int batch_size,
                        const int thread_num, const int max_out_num) {
  assert(prev_layer.node_num == prev_node_num);

  const int block_num = (node_num * batch_size + thread_num - 1) / thread_num;
  get_dense_acts_knl<<<block_num, thread_num>>>(csc_acts, node_num, batch_size);

  const int smem_size =
      (sizeof(int) + sizeof(float)) * (thread_num + max_out_num);
  relu_fwd_knl<<<batch_size, thread_num, smem_size>>>(
      prev_layer.csc_acts, d_weights, d_biases, node_num, max_out_num,
      csc_acts);
}

void ReluLayer::forward(const CscActNodes &csc_inputs, const int batch_size,
                        const int thread_num, const int max_out_num) {
  const int block_num = (node_num * batch_size + thread_num - 1) / thread_num;
  get_dense_acts_knl<<<block_num, thread_num>>>(csc_acts, node_num, batch_size);

  const int smem_size =
      (sizeof(int) + sizeof(float)) * (thread_num + max_out_num);
  relu_fwd_knl<<<batch_size, thread_num, smem_size>>>(
      csc_inputs, d_weights, d_biases, node_num, max_out_num, csc_acts);
}

void ReluLayer::bp(Layer &prev_layer, const int batch_size,
                   const int thread_num, const int max_act_num) {
  assert(prev_layer.node_num == prev_node_num);

  const int smem_size = (sizeof(int) + sizeof(float)) * max_act_num;
  bp_knl<<<batch_size, thread_num, smem_size>>>(
      csc_acts, prev_layer.csc_acts, d_weights, d_cmprs_bp_deltas, node_num,
      max_act_num, prev_layer.d_cmprs_bp_deltas, weight_adam.d_ts,
      bias_adam.d_ts);
}

void ReluLayer::bp_first_layer(const CscActNodes &csc_inputs,
                               const int batch_size, const int thread_num,
                               const int max_act_num) {
  const int smem_size = (sizeof(int) + sizeof(float)) * max_act_num;
  bp_first_layer_knl<<<batch_size, thread_num, smem_size>>>(
      csc_acts, csc_inputs, d_cmprs_bp_deltas, node_num, max_act_num,
      weight_adam.d_ts, bias_adam.d_ts);
}
