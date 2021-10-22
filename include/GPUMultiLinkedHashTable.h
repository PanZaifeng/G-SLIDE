#include "CscActNodes.h"

struct GPUMultiLinkedHashTable {
  const int max_tbl_num;
  const size_t bucket_num_per_tbl;
  const size_t pool_size;

  int *d_multi_tbl_keys;
  int *d_multi_tbl_vals;
  int *d_multi_tbl_nexts;
  int *d_multi_tbl_locks;
  int *d_multi_tbl_sizes;
  int *d_multi_tbl_pool_used_sizes;

  int threshold;

  struct filter {
    int threshold;

    filter(int threshold) : threshold(threshold) {}

    __device__ bool operator()(const int cnt) { return cnt >= threshold; }
  };

  GPUMultiLinkedHashTable(const int max_tbl_num,
                          const size_t bucket_num_per_tbl,
                          const size_t pool_size, const int threshold);

  // virtual ~GPUMultiLinkedHashTable();

  virtual void free();

  void init_tbls();

  __forceinline__ __device__ int d_hashier(const int key) {
    return key * 2654435761 & (~(1 << 31));
  }

  __device__ void d_block_reduce_cnt(const int *d_raw_keys,
                                     const int raw_key_begin,
                                     const int raw_key_end, const int tbl_id);

  __forceinline__ __device__ void d_insert_label_seq(
      const int label, int *d_tbl_keys, int *d_tbl_vals, int *d_tbl_nexts,
      int &tbl_size, int &tbl_pool_used_size) {
    const int bucket_idx = d_hashier(label) % bucket_num_per_tbl;

    int pre_entry_idx = -1, entry_idx = bucket_idx;
    int searched_key = d_tbl_keys[bucket_idx];
    if (searched_key != -1 && searched_key != label) {
      do {
        pre_entry_idx = entry_idx;
        entry_idx = d_tbl_nexts[pre_entry_idx];
      } while (entry_idx != -1 &&
               (searched_key = d_tbl_keys[entry_idx]) != label);
    }

    if (searched_key == label) {
      const int old_val = d_tbl_vals[entry_idx];
      if (old_val < threshold) ++tbl_size;
      d_tbl_vals[entry_idx] = old_val + threshold;
    } else {
      if (entry_idx == -1)
        entry_idx = tbl_pool_used_size++ + bucket_num_per_tbl;
      d_tbl_keys[entry_idx] = label;
      d_tbl_vals[entry_idx] = threshold;
      if (pre_entry_idx != -1) d_tbl_nexts[pre_entry_idx] = entry_idx;
      ++tbl_size;
    }
  }

  __device__ void d_activate_labels_seq(const int *d_labels,
                                        const int label_begin,
                                        const int label_end, const int tbl_id);

  __device__ void d_activate_labels_seq(const int *d_labels,
                                        const int *d_rand_nodes,
                                        const int label_begin,
                                        const int label_end, const int tbl_id,
                                        const int min_act_num,
                                        const int node_num, const int seed);

  void block_reduce_cnt(const CscActNodes &cmprs_gathered, const int L,
                        const int batch_size, const int thread_num);

  void activate_labels_seq(const CscActNodes &cmprs_labels,
                           const int batch_size, const int thread_num);

  void activate_labels_seq(const CscActNodes &cmprs_labels,
                           const int *d_rand_nodes, const int batch_size,
                           const int min_act_num, const int node_num,
                           const int thread_num);

  void get_act_nodes(CscActNodes &csc_acts, const int batch_size);
};

__global__ void block_reduce_cnt_knl(
    const CscActNodes cmprs_gathered, const int L,
    GPUMultiLinkedHashTable multi_linked_htables);

__global__ void activate_labels_seq_knl(
    const CscActNodes cmprs_labels, const int batch_size,
    GPUMultiLinkedHashTable multi_linked_htables);

__global__ void activate_labels_seq_knl(
    const CscActNodes cmprs_labels, const int *d_rand_nodes,
    const int batch_size, const int min_act_num, const int node_num,
    const int seed, GPUMultiLinkedHashTable multi_linked_htables);
