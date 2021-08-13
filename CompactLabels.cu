#include "CompactLabels.h"
#include "utils.h"


CompactLabels::CompactLabels(int batch_capacity, int label_capacity)
: batch_capacity(batch_capacity), label_capacity(label_capacity) {
    CUDA_CHECK( cudaMalloc(&d_nodes, sizeof(int) * label_capacity) );
    CUDA_CHECK( cudaMalloc(&d_cols, sizeof(int) * (batch_capacity + 1)) );

    CUDA_CHECK( cudaMemset(d_nodes, 0, sizeof(int) * label_capacity) );
    CUDA_CHECK( cudaMemset(d_cols, 0, sizeof(int) * (batch_capacity + 1)) );
}

CompactLabels::~CompactLabels() {
    CUDA_CHECK( cudaFree(d_nodes) );
    CUDA_CHECK( cudaFree(d_cols) );
}

void CompactLabels::extract_from(const std::vector<int> &h_c_nodes,
                                 const std::vector<int> &h_c_cols)
{
    // printf("[%d, %d]\n", (int) h_c_nodes.size(), label_capacity);

    CUDA_CHECK( cudaMemcpy(d_nodes, &h_c_nodes[0],
                    sizeof(int) * h_c_nodes.size(), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_cols, &h_c_cols[0],
                    sizeof(int) * h_c_cols.size(), cudaMemcpyHostToDevice) );
}
