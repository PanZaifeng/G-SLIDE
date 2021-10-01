#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <vector>

#include "CscActNodes.h"
#include "Layer.h"
#include "ReluLayer.h"
#include "SoftmaxLayer.h"
#include "utils.h"

class Network {
  std::vector<std::shared_ptr<ReluLayer>> relu_layers;
  std::shared_ptr<SoftmaxLayer> softmax_layer;

  CscActNodes csc_inputs;
  CscActNodes cmprs_labels;

  const int layer_num;

 public:
  Network(const std::vector<int> &node_num_per_layer,
          const std::vector<int> &node_capacity_per_layer, const int input_size,
          const int max_batch_size, const int input_capacity,
          const int label_capacity, const int K, const int L,
          const int bin_size, const int bucket_num_per_tbl,
          const int bucket_capacity, const int threshold,
          const int tbl_num_per_tile, const int tbl_num_per_thread,
          const int linked_bucket_num_per_tbl, const int linked_pool_size);

  Network(const Network &) = delete;
  Network(Network &&) = delete;
  Network &operator=(const Network &) = delete;

  virtual ~Network();

  int eval(const std::vector<int> &h_cmprs_input_nodes,
           const std::vector<float> &h_cmprs_input_vals,
           const std::vector<int> &h_cmprs_input_offsets,
           const std::vector<int> &h_cmprs_label_nodes,
           const std::vector<int> &h_cmprs_label_offsets, const int batch_size,
           const int thread_num);

  void test_get_act_nodes(const int layer_idx,
                          const std::vector<int> &h_cmprs_act_nodes,
                          const std::vector<int> &h_cmprs_act_offsets) {
    if (layer_idx < relu_layers.size()) {
      relu_layers[layer_idx]->test_get_acts(h_cmprs_act_nodes,
                                            h_cmprs_act_offsets);
    } else {
      softmax_layer->test_get_acts(h_cmprs_act_nodes, h_cmprs_act_offsets);
    }
  }

  void train(const std::vector<int> &h_cmprs_input_nodes,
             const std::vector<float> &h_cmprs_input_vals,
             const std::vector<int> &h_cmprs_input_offsets,
             const std::vector<int> &h_cmprs_label_nodes,
             const std::vector<int> &h_cmprs_label_offsets,
             const std::vector<int> &max_act_nums, const int batch_size,
             const float lr, const int max_label_num, const int thread_num,
             const bool rebuild);

  void rebuild() { softmax_layer->rebuild(); }
};