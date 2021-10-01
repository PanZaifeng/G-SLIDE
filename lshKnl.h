#pragma once
#include "CscActNodes.h"

__global__ void init_bins_knl(int *d_bins, const int prev_node_num,
                              const int tot_elem_num);

__global__ void gen_rand_keys_knl(unsigned int *d_rand_keys, const int seed,
                                  const int prev_node_num,
                                  const int tot_elem_num);

// Assumption: prev_node_num is small while node_num is large
__global__ void init_hash_knl(const int *d_bins,
                              const float *d_weights_rowmajor,
                              const int prev_node_num, const int node_num,
                              const int tot_elem_num, const int L, const int K,
                              const int bin_size, const int tbl_num_per_tile,
                              const int bucket_num_per_tbl,
                              const int bucket_capacity, int *d_buckets,
                              int *d_bucket_sizes);

// Assumption: prev_node_num is small while node_num is large
__global__ void init_hash_knl(
    const int *d_bins, const float *d_weights_rowmajor, const int prev_node_num,
    const int node_num, const int tot_elem_num, const int L, const int K,
    const int bin_size, const int tbl_num_per_tile,
    const int tbl_num_per_thread, const int bucket_num_per_tbl,
    const int bucket_capacity, int *d_buckets, int *d_bucket_sizes);

// Assumption: previous layer is dense
__global__ void get_hash_knl(const int *d_bins,
                             const float *d_dense_inputs_colmajor,
                             const int *d_bucket_sizes, const int in_node_num,
                             const int tot_elem_num, const int L, const int K,
                             const int bin_size, const int tbl_num_per_tile,
                             const int batch_size, const int bucket_num_per_tbl,
                             const int bucket_capacity,
                             int *d_hashed_bucket_ids_colmajor,
                             int *d_hashed_bucket_sizes_colmajor);

__global__ void gather_buckets_knl(const int *d_hashed_bucket_ids_colmajor,
                                   const int *d_buckets, const int L,
                                   const int batch_size,
                                   const int bucket_num_per_tbl,
                                   const int bucket_capacity,
                                   CscActNodes cmprs_gathered);
