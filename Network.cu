#include "Network.h"


Network::Network(const std::vector<int> &node_num_per_layer,
                 const std::vector<int> &node_capacity_per_layer,
                 const int input_size, const int batch_capacity,
                 const int input_capacity, const int label_capacity,
                 const int K, const int L, const int bin_size,
                 const int pack_num, const int tbl_bucket_num,
                 const int bucket_unit_size, const int tbl_capacity)
: c_inputs(batch_capacity, input_capacity),
  c_labels(batch_capacity, label_capacity),
  layer_num(node_num_per_layer.size())
{
    assert(layer_num >= 2);
    assert(layer_num == node_capacity_per_layer.size());

    relu_layers = 
        std::vector<std::shared_ptr<ReluLayer>>(layer_num - 1);
    for (int l = 0; l < layer_num; l++) {
        if (l == 0) {
            relu_layers[l] = std::make_shared<ReluLayer>(
                input_size, node_num_per_layer[l], batch_capacity,
                node_capacity_per_layer[l]);
        } else if (l + 1 == layer_num) {
            softmax_layer = std::make_shared<SoftmaxLayer>(
                node_num_per_layer[l - 1], node_num_per_layer[l],
                batch_capacity, node_capacity_per_layer[l],
                K, L, bin_size, pack_num, tbl_bucket_num, 
                bucket_unit_size, tbl_capacity);
        } else {
            relu_layers[l] = std::make_shared<ReluLayer>(
                node_num_per_layer[l - 1], node_num_per_layer[l],
                batch_capacity, node_capacity_per_layer[l]);
        }
    }
}

int Network::eval(const std::vector<int> &h_c_input_nodes,
                  const std::vector<float> &h_c_input_vals,
                  const std::vector<int> &h_c_input_cols,
                  const std::vector<int> &h_c_label_nodes,
                  const std::vector<int> &h_c_label_cols,
                  const int batch_size, const int thread_num)
{
    // forward
    c_inputs.extract_from(h_c_input_nodes, h_c_input_vals, h_c_input_cols);

    for (int l = 0; l < relu_layers.size(); l++) {
        const int max_act_num = relu_layers[l]->node_num;
        if (l == 0) {
            relu_layers[l]->forward(c_inputs, 
                batch_size, thread_num, max_act_num);
        } else {
            relu_layers[l]->forward(*relu_layers[l - 1],
                batch_size, thread_num, max_act_num);
        }
    }
    softmax_layer->forward_dense(*relu_layers.back(), batch_size);

    CUDA_CHECK( cudaDeviceSynchronize() );

    int correct_cnt = 0;
    for (int b = 0; b < batch_size; b++) {
        const float *begin = 
            softmax_layer->d_dense_activations + b * softmax_layer->node_num;
        const float *end = begin + softmax_layer->node_num;
        const int max_node = 
            thrust::max_element(thrust::device, begin, end) - begin;
        
        const int label_col = h_c_label_cols[b];
        const int label_n_col = h_c_label_cols[b + 1];
        if (std::find(h_c_label_nodes.begin() + label_col,
                h_c_label_nodes.begin() + label_n_col, max_node)
            != h_c_label_nodes.begin() + label_n_col) 
        {
            correct_cnt++;
        }
    }

    return correct_cnt;
}

void Network::train(const std::vector<int> &h_c_input_nodes,
                    const std::vector<float> &h_c_input_vals,
                    const std::vector<int> &h_c_input_cols,
                    const std::vector<int> &h_c_label_nodes,
                    const std::vector<int> &h_c_label_cols,
                    const std::vector<int> &max_act_nums,
                    const int batch_size, const float lr,
                    const int max_label_num, const int thread_num,
                    const bool rebuild)
{
    GPUTimer timer;

    c_inputs.extract_from(h_c_input_nodes, h_c_input_vals, h_c_input_cols);
    c_labels.extract_from(h_c_label_nodes, h_c_label_cols);

    // forward
    timer.start();
    for (int l = 0; l < relu_layers.size(); l++) {
        if (l == 0) {
            relu_layers[l]->forward(c_inputs,
                batch_size, thread_num, max_act_nums[l]);
        } else {
            relu_layers[l]->forward(*relu_layers[l - 1],
                batch_size, thread_num, max_act_nums[l]);
        }
        timer.record("[FW " + std::to_string(l) + "] ");
    }
    softmax_layer->forward(*relu_layers.back(), c_labels, batch_size,
        thread_num, max_act_nums.back(), max_act_nums[layer_num - 2],
        max_label_num);
    timer.record("[FW " + std::to_string(layer_num - 1) + "] ");

    // backpropagate
    softmax_layer->bp(*relu_layers.back(), batch_size,
        thread_num, max_act_nums.back());
    timer.record("[BP " + std::to_string(layer_num - 1) + "] ");
    for (int l = relu_layers.size() - 1; l >= 0; l--) {
        if (l == 0) {
            relu_layers[l]->bp(c_inputs,
                batch_size, thread_num, max_act_nums[l]);
        } else {
            relu_layers[l]->bp(*relu_layers[l - 1],
                batch_size, thread_num, max_act_nums[l]);
        }
        timer.record("[BP " + std::to_string(l) + "] ");
    }

    // update
    for (int l = 0; l < relu_layers.size(); l++) {
        relu_layers[l]->update_weights(thread_num, lr);
        relu_layers[l]->update_biases(thread_num, lr);
        timer.record("[UPDATE " + std::to_string(l) + "] ");
    }
    softmax_layer->update_weights(thread_num, lr);
    softmax_layer->update_biases(thread_num, lr);
    timer.record("[UPDATE " + std::to_string(layer_num - 1) + "] ");

    if (rebuild) {
        softmax_layer->rebuild();
        timer.record("[REBUILD] ");
    }

    CUDA_CHECK( cudaDeviceSynchronize() );
}



