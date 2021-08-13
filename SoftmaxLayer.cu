#include "SoftmaxLayer.h"
#include "kernel.h"
#include "utils.h"


SoftmaxLayer::SoftmaxLayer(const int prev_node_num, const int node_num,
                           const int batch_capacity, const int node_capacity,
                           const int K, const int L, const int bin_size,
                           const int pack_num, const int tbl_bucket_num, 
                           const int bucket_unit_size, const int tbl_capacity)
    : Layer(prev_node_num, node_num, batch_capacity, node_capacity)
{
    hash_tbls_sp = std::make_shared<LSH>(
        node_num, prev_node_num, K, L, bin_size, pack_num,
        tbl_bucket_num, bucket_unit_size, tbl_capacity,
        batch_capacity);
    
    GPUTimer timer;
    timer.start();
    hash_tbls_sp->build(d_weights);
    timer.record("[Build Hash Table] ");

    CUBLAS_CHECK( cublasCreate(&handle) );
    CUDA_CHECK( cudaMallocManaged(&d_dense_activations,
                    sizeof(float) * batch_capacity * node_num) );
}

SoftmaxLayer::~SoftmaxLayer() {
    CUBLAS_CHECK( cublasDestroy(handle) );
    CUDA_CHECK( cudaFree(d_dense_activations) );
}

void SoftmaxLayer::forward(const Layer &prev_layer, const CompactLabels &c_labels,
                           const int batch_size, const int thread_num,
                           const int max_act_num, const int max_prev_act_num,
                           const int max_label_num)
{
    hash_tbls_sp->get_act_nodes(prev_layer.c_acts.d_vals, c_labels,
                    batch_size, this->c_acts);
    
    const int smem_size = 
        (sizeof(int) + sizeof(float)) * (max_prev_act_num + max_act_num)
        + sizeof(int) * max_label_num;
    softmax_fwd_and_bp_col_major_knl<<<batch_size, thread_num, smem_size>>>(
        prev_layer.c_acts.d_vals, this->d_weights, this->d_biases,
        prev_layer.c_acts.d_nodes, prev_layer.c_acts.d_cols,
        this->c_acts.d_nodes, this->c_acts.d_cols,
        c_labels.d_nodes, c_labels.d_cols, prev_layer.node_num,
        max_prev_act_num, max_act_num, max_label_num,
        this->c_acts.d_vals, this->d_c_bp_deltas);
}

void SoftmaxLayer::forward(const Layer &prev_layer, const int batch_size,
                           const int thread_num, const int max_act_num,
                           const int max_prev_act_num)
{
    hash_tbls_sp->get_act_nodes(prev_layer.c_acts.d_vals,
                    batch_size, this->c_acts);

    const int smem_size = 
        (sizeof(int) + sizeof(float)) * (max_prev_act_num + max_act_num);
    softmax_fwd_col_major_knl<<<batch_size, thread_num, smem_size>>>(
        prev_layer.c_acts.d_vals, this->d_weights, this->d_biases,
        prev_layer.c_acts.d_nodes, prev_layer.c_acts.d_cols,
        this->c_acts.d_nodes, this->c_acts.d_cols,
        max_prev_act_num, max_act_num,
        prev_layer.node_num, this->c_acts.d_vals);
}

void SoftmaxLayer::forward_dense(const Layer &prev_layer, 
                                 const int batch_size)
{
    const float alpha = 1.;
    const float beta = 0.;

    CUBLAS_CHECK( cublasSgemm(this->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        this->node_num, batch_size, prev_layer.node_num,
        &alpha, this->d_weights, prev_layer.node_num,
        prev_layer.c_acts.d_vals, prev_layer.node_num, &beta,
        this->d_dense_activations, this->node_num) );
}

void SoftmaxLayer::bp(Layer &prev_layer, const int batch_size,
                      const int thread_num, const int max_act_num)
{
    const int smem_size = 
        (sizeof(int) + sizeof(float)) * max_act_num;
    bp_col_major_knl<<<batch_size, thread_num, smem_size>>>(
        this->d_weights, prev_layer.c_acts.d_vals, this->d_c_bp_deltas,
        this->c_acts.d_nodes, this->c_acts.d_cols,
        prev_layer.c_acts.d_nodes, prev_layer.c_acts.d_cols,
        prev_layer.node_num, max_act_num, prev_layer.d_c_bp_deltas,
        this->weight_adam.d_ts, this->bias_adam.d_ts);
}
