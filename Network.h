#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include "utils.h"
#include "Layer.h"
#include "ReluLayer.h"
#include "SoftmaxLayer.h"


class Network {
    std::vector<std::shared_ptr<ReluLayer>> relu_layers;
    std::shared_ptr<SoftmaxLayer> softmax_layer;

    CompactActNodes c_inputs;
    CompactLabels c_labels;

    const int layer_num;

public:
    Network(const std::vector<int> &node_num_per_layer,
            const std::vector<int> &node_capacity_per_layer,
            const int input_size, const int batch_capacity,
            const int input_capacity, const int label_capacity,
            const int K, const int L, const int bin_size,
            const int pack_num, const int tbl_bucket_num,
            const int bucket_unit_size, const int tbl_capacity);

    Network(const Network &) = delete;
    Network(Network &&) = delete;
    Network &operator=(const Network &) = delete;

    virtual ~Network() {}

    int eval(const std::vector<int> &h_c_input_nodes,
             const std::vector<float> &h_c_input_vals,
             const std::vector<int> &h_c_input_cols,
             const std::vector<int> &h_c_label_nodes,
             const std::vector<int> &h_c_label_cols,
             const int batch_size, const int thread_num);

    void test_get_act_nodes(const int layer_idx,
                            const std::vector<int> &h_c_act_nodes,
                            const std::vector<int> &h_c_act_cols)
    {
        if (layer_idx < relu_layers.size())
            relu_layers[layer_idx]->test_get_acts(h_c_act_nodes, h_c_act_cols);
        else
            softmax_layer->test_get_acts(h_c_act_nodes, h_c_act_cols);
    }

    void train(const std::vector<int> &h_c_input_nodes,
               const std::vector<float> &h_c_input_vals,
               const std::vector<int> &h_c_input_cols,
               const std::vector<int> &h_c_label_nodes,
               const std::vector<int> &h_c_label_cols,
               const std::vector<int> &max_act_nums,
               const int batch_size, const float lr,
               const int max_label_num, const int thread_num,
               const bool rebuild);

    void rebuild() {
        softmax_layer->rebuild();
    }
};