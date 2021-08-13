#pragma once


__global__ void init_bins_knl(int *d_bins,
                              const int prev_node_num,
                              const int tot_bin_size);

__global__ void gen_rand_keys_knl(unsigned int *d_rand_keys,
                                  const int seed,
                                  const int prev_node_num,
                                  const int tot_bin_size);

// Assumption: prev_node_num is small while node_num is large
__global__ void init_hash_vert_sm_knl(const int *d_bins,
                                      const float *d_weights, // col major
                                      const int prev_node_num,
                                      const int node_num,
                                      const int tot_bin_size,
                                      const int L,
                                      const int K,
                                      const int bin_size,
                                      const int pack_num,
                                      const int tbl_bucket_num,
                                      const int bucket_unit_size,
                                      int *d_buckets,
                                      int *d_bucket_sizes);

__global__ void init_hash_knl(const int *d_bins,
                              const float *d_weights, // col major
                              const int prev_node_num,
                              const int node_num,
                              const int tot_bin_size,
                              const int L,
                              const int K,
                              const int bin_size,
                              const int pack_num,
                              const int tbl_bucket_num,
                              const int bucket_unit_size,
                              int *d_buckets,
                              int *d_bucket_sizes);

// Assumption: previous layer is dense
__global__ void get_hash_knl(const int *d_bins,
                             const float *d_inputs,
                             const int *d_bucket_sizes,
                             const int input_node_num,
                             const int tot_bin_size,
                             const int L,
                             const int K,
                             const int bin_size,
                             const int pack_num,
                             const int batch_size,
                             const int tbl_bucket_num,
                             const int bucket_unit_size,
                             int *d_hashed_bucket_ids,
                             int *d_hashed_bucket_sizes);

__global__ void gather_buckets_knl(const int *d_hashed_bucket_ids,
                                   const int *d_buckets,
                                   const int *d_gathered_cols,
                                   const int L,
                                   const int tbl_bucket_num,
                                   const int bucket_unit_size,
                                   const int hashed_bucket_num,
                                   int *d_gathered_nodes);

__global__ void unique_knl(const int *d_gathered_nodes,
                           const int *d_gathered_cols,
                           const int L,
                           const int tbl_capacity,
                           int *d_tbl_entries,
                           int *d_tbl_keys,
                           int *d_tbl_nexts,
                           int *d_tbl_locks,
                           int *d_tbl_sizes);

__global__ void make_labels_act_knl(const int *d_labels,
                                    const int *d_label_cols,
                                    const int batch_size,
                                    const int tbl_capacity,
                                    int *d_tbl_entries,
                                    int *d_tbl_keys,
                                    int *d_tbl_nexts,
                                    int *d_tbl_locks,
                                    int *d_tbl_sizes);

__global__ void node_count_knl(const int *d_gathered_nodes,
                               const int *d_gathered_cols,
                               const int L,
                               const int tbl_capacity,
                               int *d_tbl_entries,
                               int *d_tbl_keys,
                               int *d_tbl_nexts,
                               int *d_tbl_locks,
                               int *d_tbl_vals,
                               int *d_tbl_sizes);

__global__ void make_labels_act_knl(const int *d_labels,
                                    const int *d_label_cols,
                                    const int batch_size,
                                    const int thresh,
                                    const int tbl_capacity,
                                    int *d_tbl_entries,
                                    int *d_tbl_keys,
                                    int *d_tbl_nexts,
                                    int *d_tbl_locks,
                                    int *d_tbl_vals,
                                    int *d_tbl_sizes);
