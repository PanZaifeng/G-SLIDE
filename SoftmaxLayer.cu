#include "GPUTimer.h"
#include "SoftmaxLayer.h"
#include "kernel.h"
#include "utils.h"

SoftmaxLayer::SoftmaxLayer(const int prev_node_num, const int node_num,
                           const int max_batch_size, const int node_capacity,
                           const int K, const int L, const int bin_size,
                           const int bucket_num_per_tbl,
                           const int bucket_capacity, const int threshold,
                           const int min_act_num, const int tbl_num_per_tile,
                           const int tbl_num_per_thread,
                           const int linked_bucket_num_per_tbl,
                           const int linked_pool_size)
    : Layer(prev_node_num, node_num, max_batch_size, node_capacity) {
  lsh_tbls_ptr = std::make_shared<LSH>(
      node_num, prev_node_num, max_batch_size, K, L, bin_size,
      bucket_num_per_tbl, bucket_capacity, threshold, min_act_num,
      tbl_num_per_tile, tbl_num_per_thread, linked_bucket_num_per_tbl,
      linked_pool_size);

  GPUTimer timer;
  timer.start();
  lsh_tbls_ptr->build(d_weights, true);
  timer.record("[Build LSH Table] ");

  CUBLAS_CHECK(cublasCreate(&handle));
  CUDA_CHECK(cudaMallocManaged(&d_dense_activations,
                               sizeof(float) * max_batch_size * node_num));
}

SoftmaxLayer::~SoftmaxLayer() {
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_dense_activations));
}

void SoftmaxLayer::forward(const Layer &prev_layer,
                           const CscActNodes &cmprs_labels,
                           const int batch_size, const int thread_num,
                           const int max_in_num, const int max_out_num,
                           const int max_label_num) {
  assert(prev_layer.node_num == prev_node_num);

  lsh_tbls_ptr->query_act_nodes(prev_layer.csc_acts, cmprs_labels, batch_size,
                                csc_acts);

  int smem_size = (sizeof(int) + sizeof(float)) * (max_in_num + max_out_num) +
                  sizeof(int) * max_label_num;
  if (is_smem_enough((void *)softmax_fwd_bp_rowmajor_all_sm_knl, thread_num,
                     smem_size)) {
    softmax_fwd_bp_rowmajor_all_sm_knl<<<batch_size, thread_num, smem_size>>>(
        prev_layer.csc_acts, d_weights, d_biases, cmprs_labels, prev_node_num,
        max_in_num, max_out_num, max_label_num, csc_acts, d_cmprs_bp_deltas);
  } else {
    smem_size = (sizeof(int) + sizeof(float)) * max_in_num +
                sizeof(int) * max_label_num;
    softmax_fwd_bp_rowmajor_slide_out_knl<<<batch_size, thread_num,
                                            smem_size>>>(
        prev_layer.csc_acts, d_weights, d_biases, cmprs_labels, prev_node_num,
        max_in_num, max_label_num, csc_acts, d_cmprs_bp_deltas);
  }
}

void SoftmaxLayer::forward_dense(const Layer &prev_layer,
                                 const int batch_size) {
  const float alpha = 1.;
  const float beta = 0.;

  CUBLAS_CHECK(cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, node_num, batch_size, prev_node_num,
      &alpha, d_weights, prev_node_num, prev_layer.csc_acts.d_vals,
      prev_node_num, &beta, d_dense_activations, node_num));
}

void SoftmaxLayer::bp(Layer &prev_layer, const int batch_size,
                      const int thread_num, const int max_prev_num,
                      const int max_act_num) {
  int smem_size = (sizeof(int) + sizeof(float)) * max_act_num;
  if (is_smem_enough((void *)bp_rowmajor_knl, thread_num, smem_size)) {
    bp_rowmajor_knl<<<batch_size, thread_num, smem_size>>>(
        csc_acts, prev_layer.csc_acts, d_weights, d_cmprs_bp_deltas,
        prev_node_num, max_act_num, prev_layer.d_cmprs_bp_deltas,
        weight_adam.d_ts, bias_adam.d_ts);
  } else {
    int smem_size = (sizeof(int) + sizeof(float) * 2) * max_prev_num;
    bp_rowmajor_slide_knl<<<batch_size, thread_num, smem_size>>>(
        csc_acts, prev_layer.csc_acts, d_weights, d_cmprs_bp_deltas,
        prev_node_num, max_prev_num, prev_layer.d_cmprs_bp_deltas,
        weight_adam.d_ts, bias_adam.d_ts);
  }
}
