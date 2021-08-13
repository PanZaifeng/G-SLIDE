#pragma once

#include "Layer.h"
#include "CompactActNodes.h"


class ReluLayer : public Layer {
public:
    ReluLayer(const int prev_node_num, const int node_num,
              const int batch_capacity, const int node_capacity)
        : Layer(prev_node_num, node_num, batch_capacity, node_capacity) {}

    ReluLayer(const Layer &) = delete;
    ReluLayer(Layer &&) = delete;
    ReluLayer &operator=(const ReluLayer &) = delete;

    void forward(const Layer &prev_layer, const int batch_size,
                 const int thread_num, const int max_act_num);

    void forward(const CompactActNodes &c_inputs, const int batch_size,
                 const int thread_num, const int max_act_num);

    void bp(Layer &prev_layer, const int batch_size,
            const int thread_num, const int max_act_num);

    void bp(const CompactActNodes &c_inputs, const int batch_size,
            const int thread_num, const int max_act_num);
};
