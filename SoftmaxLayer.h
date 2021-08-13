#pragma once

#include <memory>

#include "Layer.h"
#include "CompactLabels.h"
#include "LSH.h"


class SoftmaxLayer : public Layer {
public:
    std::shared_ptr<LSH> hash_tbls_sp;

    float *d_dense_activations;
    cublasHandle_t handle;

public:
    SoftmaxLayer(const int prev_node_num, const int node_num,
                 const int batch_capacity, const int node_capacity,
                 const int K, const int L, const int bin_size,
                 const int pack_num, const int tbl_bucket_num, 
                 const int bucket_unit_size, const int tbl_capacity);

    SoftmaxLayer(const Layer &) = delete;
    SoftmaxLayer(Layer &&) = delete;
    SoftmaxLayer &operator=(const SoftmaxLayer &) = delete;

    ~SoftmaxLayer();

    void forward(const Layer &prev_layer, const CompactLabels &c_labels,
                 const int batch_size, const int thread_num,
                 const int max_act_num, const int max_prev_act_num,
                 const int max_label_num);

    void forward(const Layer &prev_layer, const int batch_size,
                 const int thread_num, const int max_act_num,
                 const int max_prev_act_num);

    void forward_dense(const Layer &prev_layer, const int batch_size);

    void bp(Layer &prev_layer, const int batch_size,
            const int thread_num, const int max_act_num);

    void rebuild() {
        hash_tbls_sp->build(d_weights);
    }
};
