#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <unordered_map>
#include <vector>

#include "LSH.h"
#include "lshKnl.h"
#include "utils.h"

LSH::LSH(const int node_num, const int prev_node_num, const int max_batch_size,
         const int K, const int L, const int bin_size,
         const int bucket_num_per_tbl, const int bucket_capacity,
         const int threshold, const int tbl_num_per_tile,
         const int tbl_num_per_thread, const int linked_bucket_num_per_tbl,
         const int linked_pool_size)
    : node_num(node_num),
      prev_node_num(prev_node_num),
      K(K),
      L(L),
      bin_size(bin_size),
      bucket_num_per_tbl(bucket_num_per_tbl),
      bucket_capacity(bucket_capacity),
      tot_elem_num(K * L * bin_size),
      tbl_num_per_tile(tbl_num_per_tile),
      tbl_num_per_thread(tbl_num_per_thread),
      cmprs_gathered(L * max_batch_size, bucket_capacity * L * max_batch_size,
                     false, true),
      multi_linked_htables(max_batch_size, linked_bucket_num_per_tbl,
                           linked_pool_size, threshold) {
  CUDA_CHECK(
      cudaMallocManaged(&d_rand_keys, sizeof(unsigned int) * tot_elem_num));
  CUDA_CHECK(cudaMalloc(&d_bins, sizeof(int) * tot_elem_num));

  const int thread_num = 128;
  const int block_num = (tot_elem_num + thread_num - 1) / thread_num;
  init_bins_knl<<<block_num, thread_num>>>(d_bins, prev_node_num, tot_elem_num);

  const size_t tot_bucket_num = L * bucket_num_per_tbl;
  const size_t tot_bucket_capacity = tot_bucket_num * bucket_capacity;
  CUDA_CHECK(cudaMallocManaged(&d_buckets, sizeof(int) * tot_bucket_capacity));
  CUDA_CHECK(cudaMallocManaged(&d_bucket_sizes, sizeof(int) * tot_bucket_num));

  CUDA_CHECK(cudaMallocManaged(&d_hashed_bucket_ids_colmajor,
                               sizeof(int) * L * max_batch_size));

  // CUDA_CHECK(cudaMallocManaged(
  //     &d_gathered_nodes, sizeof(int) * bucket_capacity * L *
  //     max_batch_size));
  // CUDA_CHECK(cudaMallocManaged(&d_gathered_offsets,
  //                              sizeof(int) * (1 + L * max_batch_size)));
  // CUDA_CHECK(cudaMemset(d_gathered_offsets, 0, sizeof(int)));
}

LSH::~LSH() {
  CUDA_CHECK(cudaFree(d_rand_keys));
  CUDA_CHECK(cudaFree(d_bins));

  CUDA_CHECK(cudaFree(d_buckets));
  CUDA_CHECK(cudaFree(d_bucket_sizes));

  CUDA_CHECK(cudaFree(d_hashed_bucket_ids_colmajor));

  // CUDA_CHECK(cudaFree(d_gathered_nodes));
  // CUDA_CHECK(cudaFree(d_gathered_offsets));
  cmprs_gathered.free();
  multi_linked_htables.free();
}

void LSH::shuffle_bins() {
  const int thread_num = 128;
  const int block_num = (tot_elem_num + thread_num - 1) / thread_num;
  gen_rand_keys_knl<<<block_num, thread_num>>>(d_rand_keys, rand(),
                                               prev_node_num, tot_elem_num);

  thrust::sort_by_key(thrust::device, d_rand_keys, d_rand_keys + tot_elem_num,
                      d_bins);
}

void LSH::build(const float *d_weights_rowmajor) {
  shuffle_bins();
  CUDA_CHECK(
      cudaMemset(d_bucket_sizes, 0, sizeof(int) * L * bucket_num_per_tbl));

  const int thread_num = 64;
  const int block_num = (node_num + thread_num - 1) / thread_num;
  const int smem_size =
      (K * bin_size * tbl_num_per_tile + thread_num * prev_node_num) *
      sizeof(int);
  if (tbl_num_per_thread == tbl_num_per_tile) {
    init_hash_knl<<<block_num, thread_num, smem_size>>>(
        d_bins, d_weights_rowmajor, prev_node_num, node_num, tot_elem_num, L, K,
        bin_size, tbl_num_per_tile, bucket_num_per_tbl, bucket_capacity,
        d_buckets, d_bucket_sizes);
  } else {
    init_hash_knl<<<block_num, thread_num, smem_size>>>(
        d_bins, d_weights_rowmajor, prev_node_num, node_num, tot_elem_num, L, K,
        bin_size, tbl_num_per_tile, tbl_num_per_thread, bucket_num_per_tbl,
        bucket_capacity, d_buckets, d_bucket_sizes);
  }
}

void LSH::query_act_nodes(const CscActNodes &csc_inputs,
                          const CscActNodes &cmprs_labels, const int batch_size,
                          CscActNodes &csc_acts) {
  // Assume inputs is dense
  // TODO: dense -> sparse transform
  const float *d_dense_inputs_colmajor = csc_inputs.d_vals;

  const int thread_num = 128;
  const int hash_block_num = (L + tbl_num_per_tile - 1) / tbl_num_per_tile;
  const int smem_size = sizeof(int) * K * bin_size * tbl_num_per_tile;
  get_hash_knl<<<hash_block_num, thread_num, smem_size>>>(
      d_bins, d_dense_inputs_colmajor, d_bucket_sizes, prev_node_num,
      tot_elem_num, L, K, bin_size, tbl_num_per_tile, batch_size,
      bucket_num_per_tbl, bucket_capacity, d_hashed_bucket_ids_colmajor,
      cmprs_gathered.d_offsets + 1);

  thrust::inclusive_scan(thrust::device, cmprs_gathered.d_offsets + 1,
                         cmprs_gathered.d_offsets + 1 + L * batch_size,
                         cmprs_gathered.d_offsets + 1);

  const int gather_block_num = (L * batch_size + thread_num - 1) / thread_num;
  gather_buckets_knl<<<gather_block_num, thread_num>>>(
      d_hashed_bucket_ids_colmajor, d_buckets, L, batch_size,
      bucket_num_per_tbl, bucket_capacity, cmprs_gathered);

  multi_linked_htables.init_tbls();
  multi_linked_htables.block_reduce_cnt(cmprs_gathered, L, batch_size,
                                        thread_num);
  multi_linked_htables.activate_labels_seq(cmprs_labels, batch_size,
                                           thread_num);
  multi_linked_htables.get_act_nodes(csc_acts, batch_size);

  /*
  std::vector<int> h_gathered_nodes;
  std::vector<int> h_gathered_offsets;
  cmprs_gathered.extract_to(h_gathered_nodes, h_gathered_offsets, L *
  batch_size);

  std::vector<std::unordered_map<int, int>> golden_maps(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    int begin = h_gathered_offsets[i * L];
    int end = h_gathered_offsets[(i + 1) * L];
    for (int j = begin; j < end; ++j) {
      int node = h_gathered_nodes[j];
      ++golden_maps[i][node];
    }
  }

  std::vector<int> h_labels;
  std::vector<int> h_label_offsets;
  cmprs_labels.extract_to(h_labels, h_label_offsets, batch_size);

  for (int i = 0; i < batch_size; ++i) {
    int begin = h_label_offsets[i];
    int end = h_label_offsets[i + 1];
    for (int j = begin; j < end; ++j) {
      int node = h_labels[j];
      ++golden_maps[i][node];
    }
  }

  std::vector<int> h_cmprs_nodes;
  std::vector<int> h_cmprs_offsets;
  csc_acts.extract_to(h_cmprs_nodes, h_cmprs_offsets, batch_size);

  bool pass = true;
  for (int i = 0; i < batch_size; ++i) {
    int begin = h_cmprs_offsets[i];
    int end = h_cmprs_offsets[i + 1];
    if (end - begin != golden_maps[i].size()) {
      printf("Size err at %d, device %d, golden %ld\n", i, end - begin,
             golden_maps[i].size());
      pass = false;
    } else {
      for (int j = begin; j < end; ++j) {
        int node = h_cmprs_nodes[j];
        if (!golden_maps[i].count(node)) {
          printf("Node err %d at %d\n", node, i);
          pass = false;
        }
      }
    }
  }

  if (pass) {
    printf("Query Pass!\n");
  } else {
    printf("Query Fail!\n");
    exit(-1);
  }
  */
}

void LSH::query_act_nodes(const CscActNodes &csc_inputs, const int batch_size,
                          CscActNodes &csc_acts) {
  // Assume inputs is dense
  // TODO: dense -> sparse transform
  const float *d_dense_inputs_colmajor = csc_inputs.d_vals;

  const int thread_num = 128;
  const int hash_block_num = (L + tbl_num_per_tile - 1) / tbl_num_per_tile;
  const int smem_size = sizeof(int) * K * bin_size * tbl_num_per_tile;
  get_hash_knl<<<hash_block_num, thread_num, smem_size>>>(
      d_bins, d_dense_inputs_colmajor, d_bucket_sizes, prev_node_num,
      tot_elem_num, L, K, bin_size, tbl_num_per_tile, batch_size,
      bucket_num_per_tbl, bucket_capacity, d_hashed_bucket_ids_colmajor,
      cmprs_gathered.d_offsets + 1);

  thrust::inclusive_scan(thrust::device, cmprs_gathered.d_offsets + 1,
                         cmprs_gathered.d_offsets + 1 + L * batch_size,
                         cmprs_gathered.d_offsets + 1);

  const int gather_block_num = (L * batch_size + thread_num - 1) / thread_num;
  gather_buckets_knl<<<gather_block_num, thread_num>>>(
      d_hashed_bucket_ids_colmajor, d_buckets, L, batch_size,
      bucket_num_per_tbl, bucket_capacity, cmprs_gathered);

  multi_linked_htables.init_tbls();
  multi_linked_htables.block_reduce_cnt(cmprs_gathered, L, batch_size,
                                        thread_num);
  multi_linked_htables.get_act_nodes(csc_acts, batch_size);
}