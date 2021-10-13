#pragma once
#include "CscActNodes.h"
#include "GPUMultiLinkedHashTable.h"

class LSH {
  unsigned int *d_rand_keys;
  int *d_bins;

  int *d_rand_node_keys;
  int *d_rand_nodes;

  const int node_num;
  const int prev_node_num;

  const int K, L;
  const int bin_size;
  const int tot_elem_num;
  const int ceil_elem_num;
  const int tbl_num_per_tile;
  const int tbl_num_per_thread;

  const int bucket_num_per_tbl;
  const int bucket_capacity;
  int *d_buckets;
  int *d_bucket_sizes;

  const int min_act_num;

  int *d_hashed_bucket_ids_colmajor;
  // int *d_hashed_bucket_sizes;

  CscActNodes cmprs_gathered;

  GPUMultiLinkedHashTable multi_linked_htables;

 public:
  LSH(const int node_num, const int prev_node_num, const int max_batch_size,
      const int K, const int L, const int bin_size,
      const int bucket_num_per_tbl, const int bucket_capacity,
      const int threshold, const int min_act_num, const int tbl_num_per_tile,
      const int tbl_num_per_thread, const int linked_bucket_num_per_tbl,
      const int linked_pool_size);

  ~LSH();

  void shuffle_bins();
  void shuffle_rand();

  void build(const float *d_weights_rowmajor, const bool reshuffle);

  void query_act_nodes(const CscActNodes &csc_inputs,
                       const CscActNodes &cmprs_labels, const int batch_size,
                       CscActNodes &csc_acts);

  void query_act_nodes(const CscActNodes &csc_inputs, const int batch_size,
                       CscActNodes &csc_acts);
};