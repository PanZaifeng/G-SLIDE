#pragma once

#include "CscActNodes.h"
#include "Layer.h"

class ReluLayer : public Layer {  // weight: col major
 public:
  ReluLayer(const int prev_node_num, const int node_num,
            const int max_batch_size, const int node_capacity)
      : Layer(prev_node_num, node_num, max_batch_size, node_capacity) {}

  ReluLayer(const Layer &) = delete;
  ReluLayer(Layer &&) = delete;
  ReluLayer &operator=(const ReluLayer &) = delete;

  void forward(const Layer &prev_layer, const int batch_size,
               const int thread_num, const int max_out_num);

  void forward(const CscActNodes &csc_inputs, const int batch_size,
               const int thread_num, const int max_out_num);

  void bp(Layer &prev_layer, const int batch_size, const int thread_num,
          const int max_act_num);

  void bp_first_layer(const CscActNodes &csc_inputs, const int batch_size,
                      const int thread_num, const int max_act_num);
};
