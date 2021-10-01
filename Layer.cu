#include <algorithm>
#include <random>
#include <vector>

#include "Layer.h"
#include "kernel.h"
#include "utils.h"

Layer::Adam::Adam(int size) {
  CUDA_CHECK(cudaMalloc(&d_ts, sizeof(float) * size));
  CUDA_CHECK(cudaMalloc(&d_moms, sizeof(float) * size));
  CUDA_CHECK(cudaMalloc(&d_vels, sizeof(float) * size));

  CUDA_CHECK(cudaMemset(d_ts, 0, sizeof(float) * size));
  CUDA_CHECK(cudaMemset(d_moms, 0, sizeof(float) * size));
  CUDA_CHECK(cudaMemset(d_vels, 0, sizeof(float) * size));
}

Layer::Adam::~Adam() {
  CUDA_CHECK(cudaFree(d_ts));
  CUDA_CHECK(cudaFree(d_moms));
  CUDA_CHECK(cudaFree(d_vels));
}

Layer::Layer(const int prev_node_num, const int node_num,
             const int max_batch_size, const int node_capacity)
    : prev_node_num(prev_node_num),
      node_num(node_num),
      weight_adam(prev_node_num * node_num),
      bias_adam(node_num),
      csc_acts(max_batch_size, node_capacity) {
  const int weight_size = prev_node_num * node_num;
  CUDA_CHECK(cudaMalloc(&d_weights, sizeof(float) * weight_size));
  CUDA_CHECK(cudaMalloc(&d_biases, sizeof(float) * node_num));

  std::vector<float> tmp_weights(weight_size);
  std::vector<float> tmp_biases(node_num);

  std::random_device rd;
  std::default_random_engine dre(rd());
  std::normal_distribution<float> distribution(0.0, 0.01);

  std::generate(tmp_weights.begin(), tmp_weights.end(),
                [&]() { return distribution(dre); });
  std::generate(tmp_biases.begin(), tmp_biases.end(),
                [&]() { return distribution(dre); });

  CUDA_CHECK(cudaMemcpy(d_weights, &tmp_weights[0], sizeof(float) * weight_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biases, &tmp_biases[0], sizeof(float) * node_num,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_cmprs_bp_deltas, sizeof(float) * node_capacity));
  CUDA_CHECK(cudaMemset(d_cmprs_bp_deltas, 0, sizeof(float) * node_capacity));
}

Layer::~Layer() {
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_biases));
  CUDA_CHECK(cudaFree(d_cmprs_bp_deltas));
  csc_acts.free();
}

void Layer::update_weights(const int thread_num, const float lr) {
  const int weight_size = prev_node_num * node_num;
  const int weight_block_num = (weight_size + thread_num - 1) / thread_num;
  update_weights_knl<<<weight_block_num, thread_num>>>(
      d_weights, weight_adam.d_ts, weight_adam.d_moms, weight_adam.d_vels, lr,
      weight_size);
}

void Layer::update_biases(const int thread_num, const float lr) {
  const int bias_block_num = (node_num + thread_num - 1) / thread_num;
  update_weights_knl<<<bias_block_num, thread_num>>>(
      d_biases, bias_adam.d_ts, bias_adam.d_moms, bias_adam.d_vels, lr,
      node_num);
}
