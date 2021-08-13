#pragma once

#include "CompactActNodes.h"


class Layer {
public:
    const int prev_node_num;
    const int node_num;

    float *d_weights;
    float *d_biases;

    CompactActNodes c_acts;
    float *d_c_bp_deltas;

    struct Adam {
        float *d_ts;
        float *d_moms;
        float *d_vels;

        Adam(int size);

        ~Adam();
    };

    Adam weight_adam;
    Adam bias_adam;

public:
    Layer(const int prev_node_num, const int node_num,
          const int batch_capacity, const int node_capacity);

    Layer(const Layer &) = delete;
    Layer(Layer &&) = delete;
    Layer &operator=(const Layer &) = delete;

    virtual ~Layer();

    void test_get_acts(const std::vector<int> &h_c_act_nodes,
                       const std::vector<int> &h_c_act_cols) {
        c_acts.extract_from(h_c_act_nodes, h_c_act_cols);
    }

    virtual void update_weights(const int thread_num, const float lr);
    virtual void update_biases(const int thread_num, const float lr);
};
