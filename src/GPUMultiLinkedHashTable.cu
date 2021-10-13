#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/scan.h>

#include <cassert>
#include <cstdio>

#include "GPUMultiLinkedHashTable.h"
#include "utils.h"

GPUMultiLinkedHashTable::GPUMultiLinkedHashTable(
    const int max_tbl_num, const size_t bucket_num_per_tbl,
    const size_t pool_size, const int threshold)
    : max_tbl_num(max_tbl_num),
      bucket_num_per_tbl(bucket_num_per_tbl),
      pool_size(pool_size),
      threshold(threshold) {
  CUDA_CHECK(cudaMallocManaged(
      &d_multi_tbl_keys,
      sizeof(int) * (bucket_num_per_tbl + pool_size) * max_tbl_num));
  CUDA_CHECK(cudaMallocManaged(
      &d_multi_tbl_vals,
      sizeof(int) * (bucket_num_per_tbl + pool_size) * max_tbl_num));
  CUDA_CHECK(cudaMallocManaged(
      &d_multi_tbl_nexts,
      sizeof(int) * (bucket_num_per_tbl + pool_size) * max_tbl_num));
  CUDA_CHECK(cudaMallocManaged(&d_multi_tbl_locks,
                               sizeof(int) * bucket_num_per_tbl * max_tbl_num));
  CUDA_CHECK(cudaMallocManaged(&d_multi_tbl_sizes, sizeof(int) * max_tbl_num));
  CUDA_CHECK(cudaMallocManaged(&d_multi_tbl_pool_used_sizes,
                               sizeof(int) * max_tbl_num));

  init_tbls();
}

void GPUMultiLinkedHashTable::free() {
  CUDA_CHECK(cudaFree(d_multi_tbl_keys));
  CUDA_CHECK(cudaFree(d_multi_tbl_vals));
  CUDA_CHECK(cudaFree(d_multi_tbl_nexts));
  CUDA_CHECK(cudaFree(d_multi_tbl_locks));
  CUDA_CHECK(cudaFree(d_multi_tbl_sizes));
  CUDA_CHECK(cudaFree(d_multi_tbl_pool_used_sizes));
}

void GPUMultiLinkedHashTable::init_tbls() {
  CUDA_CHECK(
      cudaMemset(d_multi_tbl_keys, -1,
                 sizeof(int) * (bucket_num_per_tbl + pool_size) * max_tbl_num));
  CUDA_CHECK(
      cudaMemset(d_multi_tbl_vals, 0,
                 sizeof(int) * (bucket_num_per_tbl + pool_size) * max_tbl_num));
  CUDA_CHECK(
      cudaMemset(d_multi_tbl_nexts, -1,
                 sizeof(int) * (bucket_num_per_tbl + pool_size) * max_tbl_num));
  CUDA_CHECK(cudaMemset(d_multi_tbl_locks, 0,
                        sizeof(int) * bucket_num_per_tbl * max_tbl_num));
  CUDA_CHECK(cudaMemset(d_multi_tbl_sizes, 0, sizeof(int) * max_tbl_num));
  CUDA_CHECK(
      cudaMemset(d_multi_tbl_pool_used_sizes, 0, sizeof(int) * max_tbl_num));
}

__device__ void GPUMultiLinkedHashTable::d_block_add(const int tbl_id,
                                                     const int key,
                                                     bool exec_flag) {
  assert(tbl_id < max_tbl_num);

  int *d_tbl_keys =
      d_multi_tbl_keys + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_vals =
      d_multi_tbl_vals + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_nexts =
      d_multi_tbl_nexts + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_locks = d_multi_tbl_locks + tbl_id * bucket_num_per_tbl;

  __shared__ int s_tbl_size;
  __shared__ int s_tbl_pool_used_size;
  if (threadIdx.x == 0) {
    s_tbl_size = d_multi_tbl_sizes[tbl_id];
    s_tbl_pool_used_size = d_multi_tbl_pool_used_sizes[tbl_id];
  }
  __syncthreads();

  int bucket_idx;
  if (exec_flag) {
    bucket_idx = d_hashier(key) % bucket_num_per_tbl;
  }

  int pre_entry_idx = -1, entry_idx;
  int stop_flag;
  do {
    stop_flag = 1;

    if (exec_flag) {
      int searched_key = key + 1;
      if (pre_entry_idx == -1) {
        entry_idx = bucket_idx;
        searched_key = d_tbl_keys[entry_idx];
      }

      if (searched_key != -1 && searched_key != key) {
        do {
          pre_entry_idx = entry_idx;
          entry_idx = d_tbl_nexts[pre_entry_idx];
        } while (entry_idx != -1 &&
                 (searched_key = d_tbl_keys[entry_idx]) != key);
      }

      if (searched_key == key) {
        if (atomicAdd(d_tbl_vals + entry_idx, 1) == threshold - 1)
          atomicAdd(&s_tbl_size, 1);
        exec_flag = false;
      } else {
        if (atomicCAS(d_tbl_locks + bucket_idx, 0, 1) ==
            0) {  // acquire lock successfully
          // check again
          __threadfence_block();  // ensure read after lock
          if (pre_entry_idx == -1 ? d_tbl_keys[entry_idx] == -1
                                  : d_tbl_nexts[pre_entry_idx] == -1) {
            if (entry_idx == -1)
              entry_idx =
                  atomicAdd(&s_tbl_pool_used_size, 1) + bucket_num_per_tbl;
            d_tbl_keys[entry_idx] = key;
            __threadfence_block();  // ensure key writen before next

            if (pre_entry_idx != -1) {
              d_tbl_nexts[pre_entry_idx] = entry_idx;
              __threadfence_block();
            }

            if (atomicAdd(d_tbl_vals + entry_idx, 1) == threshold - 1)
              atomicAdd(&s_tbl_size, 1);

            exec_flag = false;
          }

          atomicExch(d_tbl_locks + bucket_idx, 0);  // release lock
        }

        if (exec_flag) {  // fail to acquire lock or list has been updated
          if (pre_entry_idx != -1) entry_idx = pre_entry_idx;  // roll back
          stop_flag = 0;
        }
      }
    }

    stop_flag = __syncthreads_and(stop_flag);
  } while (stop_flag == 0);

  assert(s_tbl_pool_used_size <= pool_size);

  if (threadIdx.x == 0) {
    d_multi_tbl_sizes[tbl_id] = s_tbl_size;
    d_multi_tbl_pool_used_sizes[tbl_id] = s_tbl_pool_used_size;
  }
}

__device__ void GPUMultiLinkedHashTable::d_block_reduce_cnt(
    const int *d_raw_keys, const int raw_key_begin, const int raw_key_end,
    const int tbl_id) {
  assert(tbl_id < max_tbl_num);

  int *d_tbl_keys =
      d_multi_tbl_keys + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_vals =
      d_multi_tbl_vals + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_nexts =
      d_multi_tbl_nexts + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_locks = d_multi_tbl_locks + tbl_id * bucket_num_per_tbl;

  __shared__ int s_tbl_size;
  __shared__ int s_tbl_pool_used_size;
  if (threadIdx.x == 0) {
    s_tbl_size = d_multi_tbl_sizes[tbl_id];
    s_tbl_pool_used_size = d_multi_tbl_pool_used_sizes[tbl_id];
  }
  __syncthreads();

  FOR_IDX_SYNC(raw_key_idx, raw_key_begin, raw_key_end) {
    bool exec_flag = raw_key_idx < raw_key_end;

    int key, bucket_idx;
    if (exec_flag) {
      key = d_raw_keys[raw_key_idx];
      bucket_idx = d_hashier(key) % bucket_num_per_tbl;
    }

    int pre_entry_idx = -1, entry_idx;
    int stop_flag;
    do {
      stop_flag = 1;

      if (exec_flag) {
        int searched_key = key + 1;
        if (pre_entry_idx == -1) {
          entry_idx = bucket_idx;
          searched_key = d_tbl_keys[entry_idx];
        }

        if (searched_key != -1 && searched_key != key) {
          do {
            pre_entry_idx = entry_idx;
            entry_idx = d_tbl_nexts[pre_entry_idx];
          } while (entry_idx != -1 &&
                   (searched_key = d_tbl_keys[entry_idx]) != key);
        }

        if (searched_key == key) {
          if (atomicAdd(d_tbl_vals + entry_idx, 1) == threshold - 1)
            atomicAdd(&s_tbl_size, 1);
          exec_flag = false;
        } else {
          if (atomicCAS(d_tbl_locks + bucket_idx, 0, 1) ==
              0) {  // acquire lock successfully
            // check again
            __threadfence_block();  // ensure read after lock
            if (pre_entry_idx == -1 ? d_tbl_keys[entry_idx] == -1
                                    : d_tbl_nexts[pre_entry_idx] == -1) {
              if (entry_idx == -1)
                entry_idx =
                    atomicAdd(&s_tbl_pool_used_size, 1) + bucket_num_per_tbl;
              d_tbl_keys[entry_idx] = key;
              __threadfence_block();  // ensure key writen before next

              if (pre_entry_idx != -1) {
                d_tbl_nexts[pre_entry_idx] = entry_idx;
                __threadfence_block();
              }

              if (atomicAdd(d_tbl_vals + entry_idx, 1) == threshold - 1)
                atomicAdd(&s_tbl_size, 1);

              exec_flag = false;
            }

            atomicExch(d_tbl_locks + bucket_idx, 0);  // release lock
          }

          if (exec_flag) {  // fail to acquire lock or list has been updated
            if (pre_entry_idx != -1) entry_idx = pre_entry_idx;  // roll back
            stop_flag = 0;
          }
        }
      }

      stop_flag = __syncthreads_and(stop_flag);
    } while (stop_flag == 0);
  }

  assert(s_tbl_pool_used_size <= pool_size);

  if (threadIdx.x == 0) {
    d_multi_tbl_sizes[tbl_id] = s_tbl_size;
    d_multi_tbl_pool_used_sizes[tbl_id] = s_tbl_pool_used_size;
  }
}

// number of labels of one sample tend to be small, so activate them
// sequentially
__device__ void GPUMultiLinkedHashTable::d_activate_labels_seq(
    const int *d_labels, const int *d_rand_nodes, const int label_begin,
    const int label_end, const int tbl_id, const int min_act_num,
    const int node_num, const int seed) {
  int *d_tbl_keys =
      d_multi_tbl_keys + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_vals =
      d_multi_tbl_vals + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_nexts =
      d_multi_tbl_nexts + tbl_id * (bucket_num_per_tbl + pool_size);
  // int *d_tbl_locks = d_multi_tbl_locks + tbl_id * bucket_num_per_tbl;

  int tbl_size = d_multi_tbl_sizes[tbl_id];
  int tbl_pool_used_size = d_multi_tbl_pool_used_sizes[tbl_id];

  for (int i = label_begin; i < label_end; ++i) {
    const int label = d_labels[i];
    d_insert_label_seq(label, d_tbl_keys, d_tbl_vals, d_tbl_nexts, tbl_size,
                       tbl_pool_used_size);
  }

  if (tbl_size < min_act_num) {
    // printf("Start random nodes inserting\n");

    thrust::minstd_rand rand_eng(seed);
    rand_eng.discard(tbl_id);
    int i = rand_eng() % node_num;
    while (tbl_size < min_act_num) {
      const int node = d_rand_nodes[i];
      d_insert_label_seq(node, d_tbl_keys, d_tbl_vals, d_tbl_nexts, tbl_size,
                         tbl_pool_used_size);
      i = (i + 1) % node_num;
    }
  }

  assert(tbl_pool_used_size <= pool_size);

  d_multi_tbl_sizes[tbl_id] = tbl_size;
  d_multi_tbl_pool_used_sizes[tbl_id] = tbl_pool_used_size;
}

__device__ void GPUMultiLinkedHashTable::d_activate_labels_seq(
    const int *d_labels, const int label_begin, const int label_end,
    const int tbl_id) {
  int *d_tbl_keys =
      d_multi_tbl_keys + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_vals =
      d_multi_tbl_vals + tbl_id * (bucket_num_per_tbl + pool_size);
  int *d_tbl_nexts =
      d_multi_tbl_nexts + tbl_id * (bucket_num_per_tbl + pool_size);
  // int *d_tbl_locks = d_multi_tbl_locks + tbl_id * bucket_num_per_tbl;

  int tbl_size = d_multi_tbl_sizes[tbl_id];
  int tbl_pool_used_size = d_multi_tbl_pool_used_sizes[tbl_id];

  for (int i = label_begin; i < label_end; ++i) {
    const int label = d_labels[i];
    d_insert_label_seq(label, d_tbl_keys, d_tbl_vals, d_tbl_nexts, tbl_size,
                       tbl_pool_used_size);
  }

  assert(tbl_pool_used_size <= pool_size);

  d_multi_tbl_sizes[tbl_id] = tbl_size;
  d_multi_tbl_pool_used_sizes[tbl_id] = tbl_pool_used_size;
}

void GPUMultiLinkedHashTable::block_reduce_cnt(
    const CscActNodes &cmprs_gathered, const int L, const int batch_size,
    const int thread_num) {
  block_reduce_cnt_knl<<<batch_size, thread_num>>>(cmprs_gathered, L, *this);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUMultiLinkedHashTable::activate_labels_seq(
    const CscActNodes &cmprs_labels, const int batch_size,
    const int thread_num) {
  const int block_num = (batch_size + thread_num - 1) / thread_num;
  activate_labels_seq_knl<<<block_num, thread_num>>>(cmprs_labels, batch_size,
                                                     *this);
}

void GPUMultiLinkedHashTable::activate_labels_seq(
    const CscActNodes &cmprs_labels, const int *d_rand_nodes,
    const int batch_size, const int min_act_num, const int node_num,
    const int thread_num) {
  const int block_num = (batch_size + thread_num - 1) / thread_num;
  activate_labels_seq_knl<<<block_num, thread_num>>>(cmprs_labels, d_rand_nodes,
                                                     batch_size, min_act_num,
                                                     node_num, rand(), *this);
}

void GPUMultiLinkedHashTable::get_act_nodes(CscActNodes &csc_acts,
                                            const int batch_size) {
  assert(batch_size <= max_tbl_num);

  CUDA_CHECK(cudaMemset(csc_acts.d_offsets, 0, sizeof(int)));
  thrust::inclusive_scan(thrust::device, d_multi_tbl_sizes,
                         d_multi_tbl_sizes + batch_size,
                         csc_acts.d_offsets + 1);

  int tot_node_num;
  CUDA_CHECK(cudaMemcpy(&tot_node_num, csc_acts.d_offsets + batch_size,
                        sizeof(int), cudaMemcpyDeviceToHost));
  assert(tot_node_num <= csc_acts.node_capacity);

  thrust::copy_if(
      thrust::device, d_multi_tbl_keys,
      d_multi_tbl_keys + (bucket_num_per_tbl + pool_size) * batch_size,
      d_multi_tbl_vals, csc_acts.d_nodes, filter(threshold));
}

__global__ void block_reduce_cnt_knl(
    const CscActNodes cmprs_gathered, const int L,
    GPUMultiLinkedHashTable multi_linked_htables) {
  assert(blockIdx.x < multi_linked_htables.max_tbl_num);

  const int gathered_begin = cmprs_gathered.d_offsets[blockIdx.x * L];
  const int gathered_end = cmprs_gathered.d_offsets[(blockIdx.x + 1) * L];
  multi_linked_htables.d_block_reduce_cnt(
      cmprs_gathered.d_nodes, gathered_begin, gathered_end, blockIdx.x);
}

__global__ void activate_labels_seq_knl(
    const CscActNodes cmprs_labels, const int batch_size,
    GPUMultiLinkedHashTable multi_linked_htables) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= batch_size) return;

  const int label_begin = cmprs_labels.d_offsets[tid];
  const int label_end = cmprs_labels.d_offsets[tid + 1];
  multi_linked_htables.d_activate_labels_seq(cmprs_labels.d_nodes, label_begin,
                                             label_end, tid);
}

__global__ void activate_labels_seq_knl(
    const CscActNodes cmprs_labels, const int *d_rand_nodes,
    const int batch_size, const int min_act_num, const int node_num,
    const int seed, GPUMultiLinkedHashTable multi_linked_htables) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= batch_size) return;

  const int label_begin = cmprs_labels.d_offsets[tid];
  const int label_end = cmprs_labels.d_offsets[tid + 1];
  multi_linked_htables.d_activate_labels_seq(cmprs_labels.d_nodes, d_rand_nodes,
                                             label_begin, label_end, tid,
                                             min_act_num, node_num, seed);
}
