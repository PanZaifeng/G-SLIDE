#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/random.h>

#include <cfloat>

#include "lshKnl.h"
#include "utils.h"

__global__ void init_bins_knl(int *d_bins, const int prev_node_num,
                              const int tot_elem_num) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= tot_elem_num) return;

  d_bins[tid] = tid % prev_node_num;
}

__global__ void gen_rand_keys_knl(unsigned int *d_rand_keys, const int seed,
                                  const int prev_node_num,
                                  const int tot_elem_num) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= tot_elem_num) return;

  const int permute_id = tid / prev_node_num;
  unsigned int key = permute_id << 16;

  thrust::minstd_rand rand_eng(seed);
  rand_eng.discard(tid);
  key |= rand_eng() & (1 << 16 - 1);

  d_rand_keys[tid] = key;
}

__global__ void gen_rand_keys_knl(int *d_rand_keys, const int seed,
                                  const int node_num) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= node_num) return;

  thrust::minstd_rand rand_eng(seed);
  rand_eng.discard(tid);

  d_rand_keys[tid] = rand_eng();
}

// Assumption: prev_node_num is small while node_num is large
// tt: one thread for each tile
__global__ void init_hash_tt_knl(
    const int *d_bins, const float *d_weights_rowmajor, const int prev_node_num,
    const int node_num, const int tot_elem_num, const int L, const int K,
    const int bin_size, const int tbl_num_per_tile,
    const int bucket_num_per_tbl, const int bucket_capacity, int *d_buckets,
    int *d_bucket_sizes) {
  const int elem_num_per_tile = K * bin_size * tbl_num_per_tile;
  extern __shared__ int smem[];
  int *s_tile_bins = smem;  // elem_num_per_tile
  float *s_weights_rowmajor =
      (float *)s_tile_bins + elem_num_per_tile;  // blockDim.x * prev_node_num

  const int log_bin_size = (int)logf(bin_size);

  const int tot_weight_size = node_num * prev_node_num;
  const int s_weight_size = blockDim.x * prev_node_num;
  const int weight_begin = blockIdx.x * s_weight_size;
  FOR_IDX_ASYNC(s_weight_idx, 0, s_weight_size) {
    const int weight_idx = s_weight_idx + weight_begin;
    if (weight_idx < tot_weight_size) {
      s_weights_rowmajor[s_weight_idx] = d_weights_rowmajor[weight_idx];
    }
  }
  // __syncthreads();

  const int node_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const bool exec_flag = node_idx < node_num;
  const float *s_node_weights =
      s_weights_rowmajor + threadIdx.x * prev_node_num;
  for (int tbl_begin = 0; tbl_begin < L; tbl_begin += tbl_num_per_tile) {
    const int tile_bin_begin = tbl_begin * K * bin_size;
    FOR_IDX_ASYNC(bin_elem_idx, tile_bin_begin,
                  min(tot_elem_num, tile_bin_begin + elem_num_per_tile)) {
      s_tile_bins[bin_elem_idx - tile_bin_begin] = d_bins[bin_elem_idx];
    }
    __syncthreads();

    if (exec_flag) {
      for (int i = 0; i < tbl_num_per_tile && i + tbl_begin < L; ++i) {
        int bucket_idx = 0;
        for (int j = 0; j < K; ++j) {
          float maxv = -FLT_MAX;
          int hash = 0;
          for (int k = 0; k < bin_size; ++k) {
            int idx = s_tile_bins[(i * K + j) * bin_size + k];
            float weight = s_node_weights[idx];
            if (weight > maxv) {
              maxv = weight;
              hash = k;
            }
          }
          bucket_idx += hash << ((K - 1 - j) * log_bin_size);
        }

        const int glb_bucket_idx =
            bucket_idx + (i + tbl_begin) * bucket_num_per_tbl;
        const int pos =
            atomicAdd(d_bucket_sizes + glb_bucket_idx, 1) % bucket_capacity;
        d_buckets[pos + glb_bucket_idx * bucket_capacity] = node_idx;
      }
    }

    __syncthreads();
  }
}

// Assumption: prev_node_num is small while node_num is large
__global__ void init_hash_knl(
    const int *d_bins, const float *d_weights_rowmajor, const int prev_node_num,
    const int node_num, const int tot_elem_num, const int L, const int K,
    const int bin_size, const int tbl_num_per_tile,
    const int tbl_num_per_thread, const int bucket_num_per_tbl,
    const int bucket_capacity, int *d_buckets, int *d_bucket_sizes) {
  const int elem_num_per_tile = K * bin_size * tbl_num_per_tile;
  extern __shared__ int smem[];
  int *s_tile_bins = smem;  // elem_num_per_tile
  float *s_weights_rowmajor =
      (float *)s_tile_bins + elem_num_per_tile;  // blockDim.x * prev_node_num

  const int log_bin_size = (int)logf(bin_size);

  const int thread_num_per_tile = tbl_num_per_tile / tbl_num_per_thread;
  assert(thread_num_per_tile * tbl_num_per_thread == tbl_num_per_tile);

  const int tot_weight_size = node_num * prev_node_num;
  const int s_weight_size = blockDim.x * prev_node_num;
  const int weight_begin = blockIdx.x * s_weight_size;
  FOR_IDX_ASYNC(s_weight_idx, 0, s_weight_size) {
    const int weight_idx = s_weight_idx + weight_begin;
    if (weight_idx < tot_weight_size) {
      s_weights_rowmajor[s_weight_idx] = d_weights_rowmajor[weight_idx];
    }
  }
  // __syncthreads();

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int node_idx = tid / thread_num_per_tile;
  const int tile_lane_idx = tid - node_idx * thread_num_per_tile;
  const bool exec_flag = node_idx < node_num;
  const float *s_node_weights =
      s_weights_rowmajor + threadIdx.x * prev_node_num;
  for (int tile_tbl_begin = 0; tile_tbl_begin < L;
       tile_tbl_begin += tbl_num_per_tile) {
    const int tile_bin_begin = tile_tbl_begin * K * bin_size;
    FOR_IDX_ASYNC(bin_elem_idx, tile_bin_begin,
                  min(tot_elem_num, tile_bin_begin + elem_num_per_tile)) {
      s_tile_bins[bin_elem_idx - tile_bin_begin] = d_bins[bin_elem_idx];
    }
    __syncthreads();

    if (exec_flag) {
      const int lane_begin = tile_lane_idx * tbl_num_per_thread;
      for (int i = lane_begin;
           (i - lane_begin) < tbl_num_per_thread && i + tile_tbl_begin < L;
           ++i) {
        int bucket_idx = 0;
        for (int j = 0; j < K; ++j) {
          float maxv = -FLT_MAX;
          int hash = 0;
          for (int k = 0; k < bin_size; ++k) {
            int idx = s_tile_bins[(i * K + j) * bin_size + k];
            float weight = s_node_weights[idx];
            if (weight > maxv) {
              maxv = weight;
              hash = k;
            }
          }
          bucket_idx += hash << ((K - 1 - j) * log_bin_size);
        }

        const int glb_bucket_idx =
            bucket_idx + (i + tile_tbl_begin) * bucket_num_per_tbl;
        const int pos =
            atomicAdd(d_bucket_sizes + glb_bucket_idx, 1) % bucket_capacity;
        d_buckets[pos + glb_bucket_idx * bucket_capacity] = node_idx;
      }
    }

    __syncthreads();
  }
}

// No shared memory for weights
__global__ void init_hash_no_sw_knl(
    const int *d_bins, const float *d_weights_rowmajor, const int prev_node_num,
    const int node_num, const int tot_elem_num, const int L, const int K,
    const int bin_size, const int tbl_num_per_tile,
    const int bucket_num_per_tbl, const int bucket_capacity, int *d_buckets,
    int *d_bucket_sizes) {
  extern __shared__ int s_tile_bins[];  // K * bin_size * tbl_num_per_tile

  const int log_bin_size = (int)logf(bin_size);

  const int elem_num_per_tile = K * bin_size * tbl_num_per_tile;
  const int tile_bin_begin = blockIdx.x * elem_num_per_tile;
  FOR_IDX_ASYNC(bin_elem_idx, tile_bin_begin,
                min(tot_elem_num, tile_bin_begin + elem_num_per_tile)) {
    s_tile_bins[bin_elem_idx - tile_bin_begin] = d_bins[bin_elem_idx];
  }
  __syncthreads();

  FOR_IDX_ASYNC(node_idx, 0, node_num) {
    const float *d_node_weights = d_weights_rowmajor + node_idx * prev_node_num;
    for (int i = 0;
         i < tbl_num_per_tile && i + blockIdx.x * tbl_num_per_tile < L; ++i) {
      int bucket_idx = 0;
      for (int j = 0; j < K; ++j) {
        float maxv = -FLT_MAX;
        int hash = 0;
        for (int k = 0; k < bin_size; ++k) {
          int idx = s_tile_bins[(i * K + j) * bin_size + k];
          float weight = d_node_weights[idx];
          if (weight > maxv) {
            maxv = weight;
            hash = k;
          }
        }
        bucket_idx += hash << ((K - 1 - j) * log_bin_size);
      }

      const int glb_bucket_idx =
          bucket_idx + (i + blockIdx.x * tbl_num_per_tile) * bucket_num_per_tbl;
      const int pos =
          atomicAdd(d_bucket_sizes + glb_bucket_idx, 1) % bucket_capacity;
      d_buckets[pos + glb_bucket_idx * bucket_capacity] = node_idx;
    }
  }
}

// Assumption: previous layer is dense
__global__ void get_hash_knl(const int *d_bins,
                             const float *d_dense_inputs_colmajor,
                             const int *d_bucket_sizes, const int in_node_num,
                             const int tot_elem_num, const int L, const int K,
                             const int bin_size, const int tbl_num_per_tile,
                             const int batch_size, const int bucket_num_per_tbl,
                             const int bucket_capacity,
                             int *d_hashed_bucket_ids_colmajor,
                             int *d_hashed_bucket_sizes_colmajor) {
  extern __shared__ int s_tile_bins[];  // K * bin_size * tbl_num_per_tile

  const int log_bin_size = (int)logf(bin_size);

  const int elem_num_per_tile = K * bin_size * tbl_num_per_tile;
  const int tile_bin_begin = blockIdx.x * elem_num_per_tile;
  FOR_IDX_ASYNC(bin_elem_idx, tile_bin_begin,
                min(tot_elem_num, tile_bin_begin + elem_num_per_tile)) {
    s_tile_bins[bin_elem_idx - tile_bin_begin] = d_bins[bin_elem_idx];
  }
  __syncthreads();

  FOR_IDX_ASYNC(col_idx, 0, batch_size) {
    const float *d_input_col = d_dense_inputs_colmajor + col_idx * in_node_num;
    const int offset = col_idx * L + blockIdx.x * tbl_num_per_tile;
    int *d_hashed_bucket_id_col_tile = d_hashed_bucket_ids_colmajor + offset;
    int *d_hashed_bucket_size_col_tile =
        d_hashed_bucket_sizes_colmajor + offset;
    for (int i = 0;
         i < tbl_num_per_tile && i + blockIdx.x * tbl_num_per_tile < L; ++i) {
      int bucket_idx = 0;
      for (int j = 0; j < K; ++j) {
        float maxv = -FLT_MAX;
        int hash = 0;
        for (int k = 0; k < bin_size; ++k) {
          int idx = s_tile_bins[(i * K + j) * bin_size + k];
          float input_val = d_input_col[idx];
          if (input_val > maxv) {
            maxv = input_val;
            hash = k;
          }
        }
        bucket_idx += hash << ((K - 1 - j) * log_bin_size);
      }
      // TODO: first write to shared memory
      d_hashed_bucket_id_col_tile[i] = bucket_idx;
      d_hashed_bucket_size_col_tile[i] =
          min(d_bucket_sizes[bucket_idx + (i + blockIdx.x * tbl_num_per_tile) *
                                              bucket_num_per_tbl],
              bucket_capacity);
    }
  }
}

__global__ void gather_buckets_knl(const int *d_hashed_bucket_ids_colmajor,
                                   const int *d_buckets, const int L,
                                   const int batch_size,
                                   const int bucket_num_per_tbl,
                                   const int bucket_capacity,
                                   CscActNodes cmprs_gathered) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= L * batch_size) return;

  const int bucket_id = d_hashed_bucket_ids_colmajor[tid];
  const int tbl_id = tid % L;
  const int bucket_begin =
      (bucket_id + tbl_id * bucket_num_per_tbl) * bucket_capacity;
  const int gathered_begin = cmprs_gathered.d_offsets[tid];
  const int gathered_end = cmprs_gathered.d_offsets[tid + 1];
  const int gathered_size = gathered_end - gathered_begin;

  // TODO: widely write
  for (int i = 0; i < gathered_size; ++i) {
    cmprs_gathered.d_nodes[gathered_begin + i] = d_buckets[bucket_begin + i];
  }
}
