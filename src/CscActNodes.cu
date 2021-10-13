#include <cassert>

#include "CscActNodes.h"
#include "utils.h"

CscActNodes::CscActNodes(int max_batch_size, int node_capacity,
                         bool val_enabled, bool is_managed)
    : max_batch_size(max_batch_size),
      node_capacity(node_capacity),
      val_enabled(val_enabled) {
  if (!is_managed) {
    CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(int) * node_capacity));
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * (max_batch_size + 1)));
    if (val_enabled)
      CUDA_CHECK(cudaMalloc(&d_vals, sizeof(float) * node_capacity));
  } else {
    CUDA_CHECK(cudaMallocManaged(&d_nodes, sizeof(int) * node_capacity));
    CUDA_CHECK(
        cudaMallocManaged(&d_offsets, sizeof(int) * (max_batch_size + 1)));
    if (val_enabled)
      CUDA_CHECK(cudaMallocManaged(&d_vals, sizeof(float) * node_capacity));
  }

  CUDA_CHECK(cudaMemset(d_nodes, 0, sizeof(int) * node_capacity));
  CUDA_CHECK(cudaMemset(d_offsets, 0, sizeof(int) * (max_batch_size + 1)));
  if (val_enabled)
    CUDA_CHECK(cudaMemset(d_vals, 0, sizeof(float) * node_capacity));
}

void CscActNodes::free() {
  CUDA_CHECK(cudaFree(d_nodes));
  CUDA_CHECK(cudaFree(d_offsets));
  if (val_enabled) CUDA_CHECK(cudaFree(d_vals));
}

void CscActNodes::extract_from(const std::vector<int> &h_cmprs_nodes,
                               const std::vector<float> &h_cmprs_vals,
                               const std::vector<int> &h_cmprs_offsets) {
  assert(h_cmprs_nodes.size() <= node_capacity);

  assert(val_enabled);

  CUDA_CHECK(cudaMemcpy(d_nodes, &h_cmprs_nodes[0],
                        sizeof(int) * h_cmprs_nodes.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals, &h_cmprs_vals[0],
                        sizeof(float) * h_cmprs_vals.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, &h_cmprs_offsets[0],
                        sizeof(int) * h_cmprs_offsets.size(),
                        cudaMemcpyHostToDevice));
}

void CscActNodes::extract_from(const std::vector<int> &h_cmprs_nodes,
                               const std::vector<int> &h_cmprs_offsets) {
  assert(h_cmprs_nodes.size() <= node_capacity);

  CUDA_CHECK(cudaMemcpy(d_nodes, &h_cmprs_nodes[0],
                        sizeof(int) * h_cmprs_nodes.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, &h_cmprs_offsets[0],
                        sizeof(int) * h_cmprs_offsets.size(),
                        cudaMemcpyHostToDevice));
}

void CscActNodes::extract_to(std::vector<int> &h_cmprs_nodes,
                             std::vector<float> &h_cmprs_vals,
                             std::vector<int> &h_cmprs_offsets,
                             const int batch_size) const {
  assert(val_enabled);

  h_cmprs_offsets = std::vector<int>(batch_size + 1);
  CUDA_CHECK(cudaMemcpy(&h_cmprs_offsets[0], d_offsets,
                        sizeof(int) * h_cmprs_offsets.size(),
                        cudaMemcpyDeviceToHost));

  int csc_size = h_cmprs_offsets.back();
  h_cmprs_nodes = std::vector<int>(csc_size);
  h_cmprs_vals = std::vector<float>(csc_size);
  CUDA_CHECK(cudaMemcpy(&h_cmprs_nodes[0], d_nodes, sizeof(int) * csc_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_cmprs_vals[0], d_vals, sizeof(float) * csc_size,
                        cudaMemcpyDeviceToHost));
}

void CscActNodes::extract_to(std::vector<int> &h_cmprs_nodes,
                             std::vector<int> &h_cmprs_offsets,
                             const int batch_size) const {
  h_cmprs_offsets = std::vector<int>(batch_size + 1);
  CUDA_CHECK(cudaMemcpy(&h_cmprs_offsets[0], d_offsets,
                        sizeof(int) * h_cmprs_offsets.size(),
                        cudaMemcpyDeviceToHost));

  int csc_size = h_cmprs_offsets.back();
  h_cmprs_nodes = std::vector<int>(csc_size);
  CUDA_CHECK(cudaMemcpy(&h_cmprs_nodes[0], d_nodes, sizeof(int) * csc_size,
                        cudaMemcpyDeviceToHost));
}
