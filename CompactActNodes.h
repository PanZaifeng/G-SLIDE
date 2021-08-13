#pragma once

#include <vector>


struct CompactActNodes {
    int batch_capacity;
    int node_capacity;
    int *d_nodes;
    float *d_vals;
    int *d_cols;

    CompactActNodes(int batch_capacity, int node_capacity,
                    bool is_managed = false);

    CompactActNodes(const CompactActNodes &) = delete;
    CompactActNodes(CompactActNodes &&) = delete;
    CompactActNodes &operator=(const CompactActNodes &) = delete;

    ~CompactActNodes();

    void extract_from(const std::vector<int> &h_c_nodes,
                      const std::vector<float> &h_c_vals,
                      const std::vector<int> &h_c_cols);

    void extract_from(const std::vector<int> &h_c_nodes,
                      const std::vector<int> &h_c_cols);

    void extract_to(std::vector<int> &h_c_nodes,
                    std::vector<float> &h_c_vals,
                    std::vector<int> &h_c_cols,
                    const int batch_size);
};
