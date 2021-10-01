#pragma once

#include <cublas_v2.h>

#include <memory>

#include "CscActNodes.h"
#include "LSH.h"
#include "Layer.h"

class SoftmaxLayer : public Layer {
 public:
  std::shared_ptr<LSH> lsh_tbls_ptr;

  float *d_dense_activations;
  cublasHandle_t handle;

 public:
  SoftmaxLayer(const int prev_node_num, const int node_num,
               const int max_batch_size, const int node_capacity, const int K,
               const int L, const int bin_size, const int tbl_num_per_tile,
               const int tbl_num_per_thread, const int bucket_num_per_tbl,
               const int bucket_capacity, const int linked_bucket_num_per_tbl,
               const int linked_pool_size, const int threshold);

  SoftmaxLayer(const Layer &) = delete;
  SoftmaxLayer(Layer &&) = delete;
  SoftmaxLayer &operator=(const SoftmaxLayer &) = delete;

  ~SoftmaxLayer();

  void forward(const Layer &prev_layer, const CscActNodes &cmprs_labels,
               const int batch_size, const int thread_num,
               const int max_act_num, const int max_label_num);

  void forward(const Layer &prev_layer, const int batch_size,
               const int thread_num, const int max_act_num);

  void forward_dense(const Layer &prev_layer, const int batch_size);

  void bp(Layer &prev_layer, const int batch_size, const int thread_num,
          const int max_act_num);

  void rebuild() { lsh_tbls_ptr->build(d_weights); }
};
