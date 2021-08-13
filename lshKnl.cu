#include "lshKnl.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/random.h>
#include <cfloat>


__global__ void init_bins_knl(int *d_bins,
                              const int prev_node_num,
                              const int tot_bin_size)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= tot_bin_size)
        return;

    d_bins[tid] = tid % prev_node_num;
}

__global__ void gen_rand_keys_knl(unsigned int *d_rand_keys,
                                  const int seed,
                                  const int prev_node_num,
                                  const int tot_bin_size)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= tot_bin_size)
        return;

    const unsigned int permute_id = tid / prev_node_num;
    unsigned int key = permute_id << 16;

    thrust::minstd_rand rand_eng(seed);
    rand_eng.discard(tid);
    key |= rand_eng() & (1 << 16 - 1);

    d_rand_keys[tid] = key;
}

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
                                      int *d_bucket_sizes)
{
    extern __shared__ int sm[];
    int *s_pack_k_bins = sm; // K * bin_size * pack_num
    const int pack_size = K * bin_size * pack_num;
    float *s_weights = (float *) sm + pack_size; // blockDim.x * prev_node_num

    const int log_bin_size = (int) logf(bin_size);

    const int tot_weight_size = node_num * prev_node_num;
    const int s_weight_size = blockDim.x * prev_node_num;
    const int weight_offset = blockIdx.x * s_weight_size;
    for (int offset = 0; offset < s_weight_size; offset += blockDim.x) {
        const int s_idx = threadIdx.x + offset;
        const int weight_idx = s_idx + weight_offset;
        if (s_idx < s_weight_size && weight_idx < tot_weight_size) {
            s_weights[s_idx] = d_weights[weight_idx];
        }
    }
    // __syncthreads();

    const int node_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const bool exec_flag = node_idx < node_num;
    const float *s_node_weights = s_weights + threadIdx.x * prev_node_num;
    for (int tbl_offset = 0; tbl_offset < L; tbl_offset += pack_num) {
        const int pack_offset = tbl_offset * K * bin_size;
        const int col = min(tot_bin_size, pack_offset + pack_size);
        for (int inner_offset = pack_offset; inner_offset < col;
            inner_offset += blockDim.x) {
            const int bin_content_idx = inner_offset + threadIdx.x;
            if (bin_content_idx < col) {
                s_pack_k_bins[bin_content_idx - pack_offset]
                    = d_bins[bin_content_idx];
            }
        }
        __syncthreads();

        if (exec_flag) {
            for (int i = 0; i < pack_num && i + tbl_offset < L; ++i) {
                int bucket_idx = 0;
                for (int j = 0; j < K; ++j) {
                    float maxv = -FLT_MAX;
                    int hash = 0;
                    for (int k = 0; k < bin_size; ++k) {
                        int idx = s_pack_k_bins[(i * K + j) * bin_size + k];
                        float weight = s_node_weights[idx];
                        if (weight > maxv) {
                            maxv = weight;
                            hash = k;
                        }
                    }
                    bucket_idx += hash << ((K - 1 - j) * log_bin_size);
                }

                const int glb_bucket_idx =
                    bucket_idx + (i + tbl_offset) * tbl_bucket_num;
                int pos =
                    atomicAdd(d_bucket_sizes + glb_bucket_idx, 1)
                    % bucket_unit_size;
                d_buckets[pos + glb_bucket_idx * bucket_unit_size] = node_idx;
            }
        }

        __syncthreads();
    }
}

// Assumption: previous layer is dense
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
                              int *d_bucket_sizes)
{
    extern __shared__ int s_pack_k_bins[]; // K * bin_size * pack_num

    const int log_bin_size = (int) logf(bin_size);

    const int pack_size = K * bin_size * pack_num;
    const int init_offset = blockIdx.x * pack_size;
    const int col = min(tot_bin_size, init_offset + pack_size);
    for (int offset = init_offset; offset < col; offset += blockDim.x) {
        const int bin_content_idx = offset + threadIdx.x;
        if (bin_content_idx < col) {
            s_pack_k_bins[bin_content_idx - init_offset]
                = d_bins[bin_content_idx];
        }
    }
    __syncthreads();

    for (int node_offset = 0; node_offset < node_num;
        node_offset += blockDim.x) {
        int node_idx = threadIdx.x + node_offset;
        if (node_idx < node_num) {
            const float *d_node_weights = d_weights + node_idx * prev_node_num;
            for (int i = 0; i < pack_num && i + blockIdx.x * pack_num < L;
                ++i) {
                int bucket_idx = 0;
                for (int j = 0; j < K; ++j) {
                    float maxv = -FLT_MAX;
                    int hash = 0;
                    for (int k = 0; k < bin_size; ++k) {
                        int idx = s_pack_k_bins[(i * K + j) * bin_size + k];
                        float weight = d_node_weights[idx];
                        if (weight > maxv) {
                            maxv = weight;
                            hash = k;
                        }
                    }
                    bucket_idx += hash << ((K - 1 - j) * log_bin_size);
                }
                int glb_bucket_idx = 
                    bucket_idx + (i + blockIdx.x * pack_num) * tbl_bucket_num;
                int pos = atomicAdd(d_bucket_sizes + glb_bucket_idx, 1) % bucket_unit_size;
                d_buckets[pos + glb_bucket_idx * bucket_unit_size] = node_idx;
            }
        }
    }
}

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
                             int *d_hashed_bucket_sizes)
{
    extern __shared__ int s_pack_k_bins[]; // K * bin_size * pack_num

    const int log_bin_size = (int) logf(bin_size);

    const int pack_size = K * bin_size * pack_num;
    const int init_offset = blockIdx.x * pack_size;
    const int col = min(tot_bin_size, init_offset + pack_size);
    for (int offset = init_offset; offset < col; offset += blockDim.x) {
        const int bin_content_idx = offset + threadIdx.x;
        if (bin_content_idx < col) {
            s_pack_k_bins[bin_content_idx - init_offset]
                = d_bins[bin_content_idx];
        }
    }
    __syncthreads();

    for (int row_offset = 0; row_offset < batch_size;
        row_offset += blockDim.x) {
        int row_idx = threadIdx.x + row_offset;
        if (row_idx < batch_size) {
            const float *d_row_inputs = d_inputs + row_idx * input_node_num;
            int *d_shift_hashed_bucket_ids = 
                d_hashed_bucket_ids + row_idx * L + blockIdx.x * pack_num;
            int *d_shift_hashed_bucket_sizes = 
                d_hashed_bucket_sizes + row_idx * L + blockIdx.x * pack_num;
            for (int i = 0; i < pack_num && i + blockIdx.x * pack_num < L;
                ++i) {
                int bucket_idx = 0;
                for (int j = 0; j < K; ++j) {
                    float maxv = -FLT_MAX;
                    int hash = 0;
                    for (int k = 0; k < bin_size; ++k) {
                        int idx = s_pack_k_bins[(i * K + j) * bin_size + k];
                        float input_val = d_row_inputs[idx];
                        if (input_val > maxv) {
                            maxv = input_val;
                            hash = k;
                        }
                    }
                    bucket_idx += hash << ((K - 1 - j) * log_bin_size);
                }
                // TODO: first write to shared memory
                d_shift_hashed_bucket_ids[i] = bucket_idx;
                d_shift_hashed_bucket_sizes[i] = 
                    min(d_bucket_sizes[bucket_idx + (i + blockIdx.x * pack_num)
                                       * tbl_bucket_num], bucket_unit_size);
            }
        }
    }
}

__global__ void gather_buckets_knl(const int *d_hashed_bucket_ids,
                                   const int *d_buckets,
                                   const int *d_gathered_cols,
                                   const int L,
                                   const int tbl_bucket_num,
                                   const int bucket_unit_size,
                                   const int hashed_bucket_num,
                                   int *d_gathered_nodes)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= hashed_bucket_num)
        return;

    const int bucket_id = d_hashed_bucket_ids[tid];
    const int tbl_id = tid % L;
    const int bucket_col = (bucket_id + tbl_id * tbl_bucket_num) * bucket_unit_size;
    const int gather_col = d_gathered_cols[tid];
    const int gather_n_col = d_gathered_cols[tid + 1];
    const int gather_size = gather_n_col - gather_col;

    // TODO: wide write
    for (int i = 0; i < gather_size; ++i) {
        d_gathered_nodes[gather_col + i] = d_buckets[bucket_col + i];
    }
}

__forceinline__ __device__
unsigned int hashier(const int key) {
    return key * 2654435761;
}

__global__ void unique_knl(const int *d_gathered_nodes,
                           const int *d_gathered_cols,
                           const int L,
                           const int tbl_capacity,
                           int *d_tbl_entries,
                           int *d_tbl_keys,
                           int *d_tbl_nexts,
                           int *d_tbl_locks,
                           int *d_tbl_sizes)
{
    const int gathered_col = d_gathered_cols[blockIdx.x * L];
    const int gathered_n_col = d_gathered_cols[(blockIdx.x + 1) * L];
    const int blk_tbl_offset = blockIdx.x * tbl_capacity;

    int *d_blk_tbl_entries = d_tbl_entries + blk_tbl_offset;
    int *d_blk_tbl_keys = d_tbl_keys + blk_tbl_offset;
    int *d_blk_tbl_nexts = d_tbl_nexts + blk_tbl_offset;
    int *d_blk_tbl_locks = d_tbl_locks + blk_tbl_offset;
  
    __shared__ int s_tbl_size;
    if (threadIdx.x == 0)
        s_tbl_size = 0;

    __shared__ bool s_stop_flag;

    for (int offset = gathered_col; offset < gathered_n_col;
        offset += blockDim.x) {
        const int gathered_idx = offset + threadIdx.x;
        bool exec_flag = gathered_idx < gathered_n_col;
        
        int key, hashed;
        if (exec_flag) {
            key = d_gathered_nodes[gathered_idx]; // TODO: first load to shared memory
            hashed = hashier(key) % tbl_capacity;
        }
        // __syncthreads();

        int pre_tbl_node = -1, tbl_node;
        do {
            if (threadIdx.x == 0)
                s_stop_flag = true;
            __syncthreads();

            if (exec_flag) {
                if (pre_tbl_node == -1) {
                    tbl_node = d_blk_tbl_entries[hashed];
                }
                
                while (tbl_node != -1 && d_blk_tbl_keys[tbl_node] != key) {
                    pre_tbl_node = tbl_node;
                    tbl_node = d_blk_tbl_nexts[pre_tbl_node];
                }

                if (tbl_node != -1) {
                    exec_flag = false;
                } else {
                    if (atomicCAS(d_blk_tbl_locks + hashed, 0, 1) == 0) { // acquire lock successfully
                        // check again
                        bool check_flag = pre_tbl_node == -1 
                            ? (tbl_node = d_blk_tbl_entries[hashed]) == -1
                            : (tbl_node = d_blk_tbl_nexts[pre_tbl_node]) == -1;
                        if (check_flag) {
                            const int pos = atomicAdd(&s_tbl_size, 1);
                            d_blk_tbl_keys[pos] = key;
                            __threadfence();
                            
                            // write pos at last
                            if (pre_tbl_node == -1)
                                d_blk_tbl_entries[hashed] = pos;
                            else
                                d_blk_tbl_nexts[pre_tbl_node] = pos;
                            __threadfence();

                            exec_flag = false;
                        }

                        atomicExch(d_blk_tbl_locks + hashed, 0); // release lock
                    } else { // other thread hold lock
                        s_stop_flag = false;
                    }
                }
            }
            __syncthreads(); // avoid threads holding locks waiting infinitely
        } while (!s_stop_flag);
    }

    if (threadIdx.x == 0) {
        d_tbl_sizes[blockIdx.x] = s_tbl_size;
    }
}

__global__ void make_labels_act_knl(const int *d_labels,
                                    const int *d_label_cols,
                                    const int batch_size,
                                    const int tbl_capacity,
                                    int *d_tbl_entries,
                                    int *d_tbl_keys,
                                    int *d_tbl_nexts,
                                    int *d_tbl_locks,
                                    int *d_tbl_sizes)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size)
        return;
    
    const int label_col = d_label_cols[tid];
    const int label_n_col = d_label_cols[tid + 1];
    const int t_tbl_offset = tid * tbl_capacity;

    int *d_t_tbl_entries = d_tbl_entries + t_tbl_offset;
    int *d_t_tbl_keys = d_tbl_keys + t_tbl_offset;
    int *d_t_tbl_nexts = d_tbl_nexts + t_tbl_offset;
    int t_tbl_size = d_tbl_sizes[tid];

    for (int i = label_col; i < label_n_col; ++i) {
        const int label = d_labels[i];
        const int hashed = hashier(label) % tbl_capacity;
        
        // int pre_tbl_node;
        int tbl_node = d_t_tbl_entries[hashed];
        while (tbl_node != -1 && d_t_tbl_keys[tbl_node] != label) {
            // pre_tbl_node = tbl_node;
            tbl_node = d_t_tbl_nexts[tbl_node];
        }

        if (tbl_node == -1) {
            const int pos = t_tbl_size++;
            d_t_tbl_keys[pos] = label;

            // if (pre_tbl_node == -1)
            //     d_t_tbl_entries[hashed] = pos;
            // else
            //     d_t_tbl_nexts[pre_tbl_node] = pos;
        }
    }

    d_tbl_sizes[tid] = t_tbl_size;
}

__global__ void node_count_knl(const int *d_gathered_nodes,
                               const int *d_gathered_cols,
                               const int L,
                               const int tbl_capacity,
                               int *d_tbl_entries,
                               int *d_tbl_keys,
                               int *d_tbl_nexts,
                               int *d_tbl_locks,
                               int *d_tbl_vals,
                               int *d_tbl_sizes)
{
    const int gathered_col = d_gathered_cols[blockIdx.x * L];
    const int gathered_n_col = d_gathered_cols[(blockIdx.x + 1) * L];
    const int blk_tbl_offset = blockIdx.x * tbl_capacity;

    int *d_blk_tbl_entries = d_tbl_entries + blk_tbl_offset;
    int *d_blk_tbl_keys = d_tbl_keys + blk_tbl_offset;
    int *d_blk_tbl_nexts = d_tbl_nexts + blk_tbl_offset;
    int *d_blk_tbl_locks = d_tbl_locks + blk_tbl_offset;
    int *d_blk_tbl_vals = d_tbl_vals + blk_tbl_offset;
  
    __shared__ int s_tbl_size;
    if (threadIdx.x == 0)
        s_tbl_size = 0;

    __shared__ bool s_stop_flag;

    for (int offset = gathered_col; offset < gathered_n_col;
        offset += blockDim.x) {
        const int gathered_idx = offset + threadIdx.x;
        bool exec_flag = gathered_idx < gathered_n_col;
        
        int key, hashed;
        if (exec_flag) {
            key = d_gathered_nodes[gathered_idx]; // TODO: first load to shared memory
            hashed = hashier(key) % tbl_capacity;
        }
        // __syncthreads();

        int pre_tbl_node = -1, tbl_node;
        do {
            if (threadIdx.x == 0)
                s_stop_flag = true;
            __syncthreads();

            if (exec_flag) {
                if (pre_tbl_node == -1) {
                    tbl_node = d_blk_tbl_entries[hashed];
                }
                
                while (tbl_node != -1 && d_blk_tbl_keys[tbl_node] != key) {
                    pre_tbl_node = tbl_node;
                    tbl_node = d_blk_tbl_nexts[pre_tbl_node];
                }

                if (tbl_node != -1) {
                    atomicAdd(d_blk_tbl_vals + tbl_node, 1);
                    exec_flag = false;
                } else {
                    if (atomicCAS(d_blk_tbl_locks + hashed, 0, 1) == 0) { // acquire lock successfully
                        // check again
                        bool check_flag = pre_tbl_node == -1 
                            ? (tbl_node = d_blk_tbl_entries[hashed]) == -1
                            : (tbl_node = d_blk_tbl_nexts[pre_tbl_node]) == -1;
                        if (check_flag) {
                            const int pos = atomicAdd(&s_tbl_size, 1);
                            // d_blk_tbl_vals[pos] = 1;
                            atomicAdd(d_blk_tbl_vals + pos, 1);
                            d_blk_tbl_keys[pos] = key;
                            __threadfence();
                            
                            // write pos at last
                            if (pre_tbl_node == -1)
                                d_blk_tbl_entries[hashed] = pos;
                            else
                                d_blk_tbl_nexts[pre_tbl_node] = pos;
                            __threadfence();

                            exec_flag = false;
                        }

                        atomicExch(d_blk_tbl_locks + hashed, 0); // release lock
                    } else { // other thread hold lock
                        s_stop_flag = false;
                    }
                }
            }
            __syncthreads(); // avoid threads holding locks waiting infinitely
        } while (!s_stop_flag);
    }

    if (threadIdx.x == 0) {
        d_tbl_sizes[blockIdx.x] = s_tbl_size;
    }
}

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
                                    int *d_tbl_sizes)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= batch_size)
        return;
    
    const int label_col = d_label_cols[tid];
    const int label_n_col = d_label_cols[tid + 1];
    const int t_tbl_offset = tid * tbl_capacity;

    int *d_t_tbl_entries = d_tbl_entries + t_tbl_offset;
    int *d_t_tbl_keys = d_tbl_keys + t_tbl_offset;
    int *d_t_tbl_nexts = d_tbl_nexts + t_tbl_offset;
    int *d_t_tbl_vals = d_tbl_vals + t_tbl_offset;
    int t_tbl_size = d_tbl_sizes[tid];

    for (int i = label_col; i < label_n_col; ++i) {
        const int label = d_labels[i];
        const int hashed = hashier(label) % tbl_capacity;
        
        // int pre_tbl_node;
        int tbl_node = d_t_tbl_entries[hashed];
        while (tbl_node != -1 && d_t_tbl_keys[tbl_node] != label) {
            // pre_tbl_node = tbl_node;
            tbl_node = d_t_tbl_nexts[tbl_node];
        }

        if (tbl_node != -1) {
            d_t_tbl_vals[tbl_node] = thresh;
        } else {
            const int pos = t_tbl_size++;
            d_t_tbl_keys[pos] = label;
            d_t_tbl_vals[pos] = thresh;

            // if (pre_tbl_node == -1)
            //     d_t_tbl_entries[hashed] = pos;
            // else
            //     d_t_tbl_nexts[pre_tbl_node] = pos;
        }
    }

    d_tbl_sizes[tid] = t_tbl_size;
}

