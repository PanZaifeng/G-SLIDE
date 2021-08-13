#include "CompactActNodes.h"
#include "utils.h"


CompactActNodes::CompactActNodes(int batch_capacity, int node_capacity,
                                 bool is_managed)
: batch_capacity(batch_capacity), node_capacity(node_capacity) {
    if (!is_managed) {
        CUDA_CHECK( cudaMalloc(&d_nodes, sizeof(int) * node_capacity) );
        CUDA_CHECK( cudaMalloc(&d_vals, sizeof(float) * node_capacity) );
        CUDA_CHECK( cudaMalloc(&d_cols, sizeof(int) * (batch_capacity + 1)) );
    } else {
        CUDA_CHECK( cudaMallocManaged(&d_nodes,
                        sizeof(int) * node_capacity) );
        CUDA_CHECK( cudaMallocManaged(&d_vals,
                        sizeof(float) * node_capacity) );
        CUDA_CHECK( cudaMallocManaged(&d_cols,
                        sizeof(int) * (batch_capacity + 1)) );
    }
    
    CUDA_CHECK( cudaMemset(d_nodes, 0, sizeof(int) * node_capacity) );
    CUDA_CHECK( cudaMemset(d_vals, 0, sizeof(float) * node_capacity) );
    CUDA_CHECK( cudaMemset(d_cols, 0, sizeof(int) * (batch_capacity + 1)) );
}

CompactActNodes::~CompactActNodes() {
    CUDA_CHECK( cudaFree(d_nodes) );
    CUDA_CHECK( cudaFree(d_vals) );
    CUDA_CHECK( cudaFree(d_cols) );
}

void CompactActNodes::extract_from(const std::vector<int> &h_c_nodes,
                                   const std::vector<float> &h_c_vals,
                                   const std::vector<int> &h_c_cols)
{
    // printf("[%d, %d]\n", (int) h_c_nodes.size(), node_capacity);
    // assert(h_c_nodes.size() <= node_capacity);

    CUDA_CHECK( cudaMemcpy(d_nodes, &h_c_nodes[0],
                    sizeof(int) * h_c_nodes.size(), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_vals, &h_c_vals[0],
                    sizeof(float) * h_c_vals.size(), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_cols, &h_c_cols[0],
                    sizeof(int) * h_c_cols.size(), cudaMemcpyHostToDevice) );
}

void CompactActNodes::extract_from(const std::vector<int> &h_c_nodes,
                                   const std::vector<int> &h_c_cols)
{
    // printf("[%d, %d]\n", (int) h_c_nodes.size(), node_capacity);
    // assert(h_c_nodes.size() <= node_capacity);

    CUDA_CHECK( cudaMemcpy(d_nodes, &h_c_nodes[0],
                    sizeof(int) * h_c_nodes.size(), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_cols, &h_c_cols[0],
                    sizeof(int) * h_c_cols.size(), cudaMemcpyHostToDevice) );
}

void CompactActNodes::extract_to(std::vector<int> &h_c_nodes,
                                 std::vector<float> &h_c_vals,
                                 std::vector<int> &h_c_cols,
                                 const int batch_size)
{
    h_c_cols = std::vector<int>(batch_size + 1);
    CUDA_CHECK( cudaMemcpy(&h_c_cols[0], d_cols,
                    sizeof(int) * h_c_cols.size(), cudaMemcpyDeviceToHost) );
    
    int c_size = h_c_cols.back();
    h_c_nodes = std::vector<int>(c_size);
    h_c_vals = std::vector<float>(c_size);
    CUDA_CHECK( cudaMemcpy(&h_c_nodes[0], d_nodes, sizeof(int) * c_size,
                    cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(&h_c_vals[0], d_vals, sizeof(float) * c_size,
                    cudaMemcpyDeviceToHost) );
}
