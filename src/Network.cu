#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include "GPUTimer.h"
#include "Network.h"

Network::Network(const std::vector<int> &node_num_per_layer,
                 const std::vector<int> &node_capacity_per_layer,
                 const int input_size, const int max_batch_size,
                 const int input_capacity, const int label_capacity,
                 const int K, const int L, const int bin_size,
                 const int bucket_num_per_tbl, const int bucket_capacity,
                 const int threshold, const int min_softmax_act_num,
                 const int tbl_num_per_tile, const int tbl_num_per_thread,
                 const int linked_bucket_num_per_tbl,
                 const int linked_pool_size)
    : csc_inputs(max_batch_size, input_capacity),
      cmprs_labels(max_batch_size, label_capacity, false),
      layer_num(node_num_per_layer.size()) {
  assert(layer_num >= 2);
  assert(layer_num == node_capacity_per_layer.size());

  relu_layers = std::vector<std::shared_ptr<ReluLayer>>(layer_num - 1);
  for (int l = 0; l < layer_num; ++l) {
    if (l == 0) {
      relu_layers[l] = std::make_shared<ReluLayer>(
          input_size, node_num_per_layer[l], max_batch_size,
          node_capacity_per_layer[l]);
    } else if (l + 1 == layer_num) {
      softmax_layer = std::make_shared<SoftmaxLayer>(
          node_num_per_layer[l - 1], node_num_per_layer[l], max_batch_size,
          node_capacity_per_layer[l], K, L, bin_size, bucket_num_per_tbl,
          bucket_capacity, threshold, min_softmax_act_num, tbl_num_per_tile,
          tbl_num_per_thread, linked_bucket_num_per_tbl, linked_pool_size);
    } else {
      relu_layers[l] = std::make_shared<ReluLayer>(
          node_num_per_layer[l - 1], node_num_per_layer[l], max_batch_size,
          node_capacity_per_layer[l]);
    }
  }
}

Network::~Network() {
  csc_inputs.free();
  cmprs_labels.free();
}

int Network::eval(const std::vector<int> &h_cmprs_input_nodes,
                  const std::vector<float> &h_cmprs_input_vals,
                  const std::vector<int> &h_cmprs_input_offsets,
                  const std::vector<int> &h_cmprs_label_nodes,
                  const std::vector<int> &h_cmprs_label_offsets,
                  const int batch_size, const int thread_num) {
  // forward
  csc_inputs.extract_from(h_cmprs_input_nodes, h_cmprs_input_vals,
                          h_cmprs_input_offsets);

  for (int l = 0; l < relu_layers.size(); ++l) {
    const int node_num = relu_layers[l]->node_num;
    if (l == 0) {
      relu_layers[l]->forward(csc_inputs, batch_size, thread_num, node_num);
    } else {
      relu_layers[l]->forward(*relu_layers[l - 1], batch_size, thread_num,
                              node_num);
    }
  }
  softmax_layer->forward_dense(*relu_layers.back(), batch_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  int correct_cnt = 0;
  for (int b = 0; b < batch_size; b++) {
    const float *begin =
        softmax_layer->d_dense_activations + b * softmax_layer->node_num;
    const float *end = begin + softmax_layer->node_num;
    const int max_node =
        thrust::max_element(thrust::device, begin, end) - begin;

    const int label_begin = h_cmprs_label_offsets[b];
    const int label_end = h_cmprs_label_offsets[b + 1];
    if (std::find(h_cmprs_label_nodes.begin() + label_begin,
                  h_cmprs_label_nodes.begin() + label_end,
                  max_node) != h_cmprs_label_nodes.begin() + label_end) {
      correct_cnt++;
    }
  }

  return correct_cnt;
}

void Network::train(const std::vector<int> &h_cmprs_input_nodes,
                    const std::vector<float> &h_cmprs_input_vals,
                    const std::vector<int> &h_cmprs_input_offsets,
                    const std::vector<int> &h_cmprs_label_nodes,
                    const std::vector<int> &h_cmprs_label_offsets,
                    const std::vector<int> &max_act_nums, const int batch_size,
                    const float lr, const int max_label_num,
                    const int thread_num, const bool rebuild,
                    const bool reshuffle) {
  GPUTimer timer;

  csc_inputs.extract_from(h_cmprs_input_nodes, h_cmprs_input_vals,
                          h_cmprs_input_offsets);
  cmprs_labels.extract_from(h_cmprs_label_nodes, h_cmprs_label_offsets);

  // forward
  timer.start();
  for (int l = 0; l < relu_layers.size(); ++l) {
    if (l == 0) {
      relu_layers[l]->forward(csc_inputs, batch_size, thread_num,
                              max_act_nums[l]);
    } else {
      relu_layers[l]->forward(*relu_layers[l - 1], batch_size, thread_num,
                              max_act_nums[l]);
    }
    timer.record("[FW" + std::to_string(l) + "] ");
  }
  softmax_layer->forward(*relu_layers.back(), cmprs_labels, batch_size,
                         thread_num, *(max_act_nums.end() - 2),
                         max_act_nums.back(), max_label_num);
  // timer.record("[FW" + std::to_string(layer_num - 1) + "] ");
  timer.record();

  // backpropagate
  softmax_layer->bp(*relu_layers.back(), batch_size, thread_num,
                    *(max_act_nums.end() - 2), max_act_nums.back());
  timer.record("[BP" + std::to_string(layer_num - 1) + "] ");
  for (int l = relu_layers.size() - 1; l >= 0; --l) {
    if (l == 0) {
      relu_layers[l]->bp_first_layer(csc_inputs, batch_size, thread_num,
                                     max_act_nums[l]);
    } else {
      relu_layers[l]->bp(*relu_layers[l - 1], batch_size, thread_num,
                         max_act_nums[l]);
    }
    timer.record("[BP" + std::to_string(l) + "] ");
  }

  // update
  for (int l = 0; l < relu_layers.size(); ++l) {
    relu_layers[l]->update_weights(thread_num, lr);
    relu_layers[l]->update_biases(thread_num, lr);
    timer.record("[UD" + std::to_string(l) + "] ");
  }
  softmax_layer->update_weights(thread_num, lr);
  softmax_layer->update_biases(thread_num, lr);
  timer.record("[UD" + std::to_string(layer_num - 1) + "] ");

  if (rebuild || reshuffle) {
    softmax_layer->rebuild(reshuffle);
    timer.record("[LSH_RC] ");
  }

  CUDA_CHECK(cudaDeviceSynchronize());
}
