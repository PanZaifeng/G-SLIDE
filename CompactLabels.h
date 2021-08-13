#pragma once

#include <vector>


struct CompactLabels {
    int batch_capacity;
    int label_capacity;
    int *d_nodes;
    int *d_cols;

    CompactLabels(int batch_capacity, int label_capacity);

    CompactLabels(const CompactLabels &) = delete;
    CompactLabels(CompactLabels &&) = delete;
    CompactLabels &operator=(const CompactLabels &) = delete;

    ~CompactLabels();

    void extract_from(const std::vector<int> &h_c_nodes,
                      const std::vector<int> &h_c_cols);
};
