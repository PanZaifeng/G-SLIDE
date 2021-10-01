#pragma once
#include <vector>

struct CscActNodes {
  int max_batch_size;
  int node_capacity;
  int *d_nodes;
  float *d_vals;
  int *d_offsets;

  const bool val_enabled;

  CscActNodes(int max_batch_size, int node_capacity, bool val_enabled = true,
              bool is_managed = false);

  // virtual ~CscActNodes();

  void free();

  void extract_from(const std::vector<int> &h_cmprs_nodes,
                    const std::vector<float> &h_cmprs_vals,
                    const std::vector<int> &h_cmprs_offsets);

  void extract_from(const std::vector<int> &h_cmprs_nodes,
                    const std::vector<int> &h_cmprs_offsets);

  void extract_to(std::vector<int> &h_cmprs_nodes,
                  std::vector<float> &h_cmprs_vals,
                  std::vector<int> &h_cmprs_offsets,
                  const int batch_size) const;

  void extract_to(std::vector<int> &h_cmprs_nodes,
                  std::vector<int> &h_cmprs_offsets,
                  const int batch_size) const;
};
