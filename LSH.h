#pragma once

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <cstdlib>
#include <cstdio>

#include "utils.h"
#include "lshKnl.h"
#include "CompactLabels.h"

#include <unordered_map>


struct filter {
    int thresh;

    filter(int thresh) : thresh(thresh) {}

    __device__ bool operator()(const int cnt) {
        return cnt >= thresh;
    }
};

class LSH {
    unsigned int *d_rand_keys;
    int *d_bins;
    
    const int node_num;
    const int prev_node_num;
    
    const int K, L;
    const int bin_size;
    const int tot_bin_size;
    const int pack_num;

    const int tbl_bucket_num;
    const int bucket_unit_size;
    int *d_buckets;
    int *d_bucket_sizes;

    int *d_hashed_bucket_ids;
    // int *d_hashed_bucket_sizes;
    
    int *d_gathered_nodes;
    int *d_gathered_cols;

    const int tbl_capacity;
    int *d_tbl_entries;
    int *d_tbl_keys;
    int *d_tbl_nexts;
    int *d_tbl_locks;
    int *d_tbl_sizes;

public:
    LSH(const int node_num, const int prev_node_num,
        const int K, const int L, const int bin_size, const int pack_num,
        const int tbl_bucket_num, const int bucket_unit_size,
        const int tbl_capacity, const int max_batch_size)
        : node_num(node_num), prev_node_num(prev_node_num),
        K(K), L(L), bin_size(bin_size), pack_num(pack_num), 
        tot_bin_size(K * L * bin_size), tbl_bucket_num(tbl_bucket_num), 
        bucket_unit_size(bucket_unit_size),
        tbl_capacity(tbl_capacity)
    {
        CUDA_CHECK( cudaMallocManaged(&d_rand_keys,
                        sizeof(unsigned int) * tot_bin_size) );
        CUDA_CHECK( cudaMalloc(&d_bins, sizeof(int) * tot_bin_size) );

        const int thread_num = 128;
        const int block_num = 
            (tot_bin_size + thread_num - 1) / thread_num;
        init_bins_knl<<<block_num, thread_num>>>(
            d_bins, prev_node_num, tot_bin_size);
        
        const size_t tot_bucket_num = L * tbl_bucket_num;
        const size_t tot_bucket_size = tot_bucket_num * bucket_unit_size;
        CUDA_CHECK( cudaMallocManaged(&d_buckets, sizeof(int) * tot_bucket_size) );
        CUDA_CHECK( cudaMallocManaged(&d_bucket_sizes,
                        sizeof(int) * tot_bucket_num) );
        
        CUDA_CHECK( cudaMallocManaged(&d_hashed_bucket_ids,
                        sizeof(int) * L * max_batch_size) );

        CUDA_CHECK( cudaMallocManaged(&d_gathered_nodes,
                        sizeof(int) * bucket_unit_size * L * max_batch_size) );
        CUDA_CHECK( cudaMallocManaged(&d_gathered_cols,
                        sizeof(int) * (1 + L * max_batch_size)) );
        CUDA_CHECK( cudaMemset(d_gathered_cols, 0, sizeof(int)) );

        CUDA_CHECK( cudaMallocManaged(&d_tbl_entries,
                        sizeof(int) * tbl_capacity * max_batch_size) );
        CUDA_CHECK( cudaMallocManaged(&d_tbl_keys,
                        sizeof(int) * tbl_capacity * max_batch_size) );
        CUDA_CHECK( cudaMallocManaged(&d_tbl_nexts,
                        sizeof(int) * tbl_capacity * max_batch_size) );
        CUDA_CHECK( cudaMallocManaged(&d_tbl_locks,
                        sizeof(int) * tbl_capacity * max_batch_size) );
        CUDA_CHECK( cudaMallocManaged(&d_tbl_sizes,
                        sizeof(int) * max_batch_size) );
    }

    ~LSH() {
        CUDA_CHECK( cudaFree(d_rand_keys) );
        CUDA_CHECK( cudaFree(d_bins) );

        CUDA_CHECK( cudaFree(d_buckets) );
        CUDA_CHECK( cudaFree(d_bucket_sizes) );

        CUDA_CHECK( cudaFree(d_hashed_bucket_ids) );

        CUDA_CHECK( cudaFree(d_gathered_nodes) );
        CUDA_CHECK( cudaFree(d_gathered_cols) );

        CUDA_CHECK( cudaFree(d_tbl_entries) );
        CUDA_CHECK( cudaFree(d_tbl_keys) );
        CUDA_CHECK( cudaFree(d_tbl_nexts) );
        CUDA_CHECK( cudaFree(d_tbl_locks) );
        CUDA_CHECK( cudaFree(d_tbl_sizes) );
    }

    void shuffle_bins() {
        const int thread_num = 128;
        const int block_num = 
            (tot_bin_size + thread_num - 1) / thread_num;
        
        gen_rand_keys_knl<<<block_num, thread_num>>>(
            d_rand_keys, rand(), prev_node_num, tot_bin_size);
        
        thrust::sort_by_key(thrust::device, 
            d_rand_keys, d_rand_keys + tot_bin_size, d_bins);
    }

    void build(const float *d_weights, const bool is_col_major = true) {
        shuffle_bins();

        CUDA_CHECK( cudaMemset(d_bucket_sizes, 0,
                        sizeof(int) * L * tbl_bucket_num) );

        if (is_col_major) {
            // const int thread_num = 128;
            // const int block_num = (L + pack_num - 1) / pack_num;
            // const int smem_size = sizeof(int) * K * bin_size * pack_num;
            // init_hash_knl<<<block_num, thread_num, smem_size>>>(
            //     d_bins, d_weights, prev_node_num, node_num,
            //     tot_bin_size, L, K, bin_size, pack_num,
            //     tbl_bucket_num, bucket_unit_size,
            //     d_buckets, d_bucket_sizes);

            const int thread_num = 64;
            const int block_num = (node_num + thread_num - 1) / thread_num;
            const int smem_size = 
                (K * bin_size * pack_num + thread_num * prev_node_num)
                * sizeof(int);
            init_hash_vert_sm_knl<<<block_num, thread_num, smem_size>>>(
                d_bins, d_weights, prev_node_num, node_num,
                tot_bin_size, L, K, bin_size, pack_num,
                tbl_bucket_num, bucket_unit_size,
                d_buckets, d_bucket_sizes);            
        } else {
            // TDOO
        }
    }

    void get_act_nodes(const float *d_inputs,
                       const int batch_size,
                       CompactActNodes &c_acts)
    {
        const int thread_num = 128;
        const int hash_block_num = (L + pack_num - 1) / pack_num;
        const int smem_size = sizeof(int) * K * bin_size * pack_num;
        get_hash_knl<<<hash_block_num, thread_num, smem_size>>>(
            d_bins, d_inputs, d_bucket_sizes, prev_node_num,
            tot_bin_size, L, K, bin_size, pack_num, batch_size,
            tbl_bucket_num, bucket_unit_size,
            d_hashed_bucket_ids, d_gathered_cols + 1);
        
        thrust::inclusive_scan(thrust::device,
            d_gathered_cols + 1, d_gathered_cols + 1 + L * batch_size,
            d_gathered_cols + 1);

        const int gather_block_num
            = (L * batch_size + thread_num - 1) / thread_num;
        gather_buckets_knl<<<gather_block_num, thread_num>>>(
            d_hashed_bucket_ids, d_buckets, d_gathered_cols,
            L, tbl_bucket_num, bucket_unit_size, L * batch_size,
            d_gathered_nodes);
        
        CUDA_CHECK( cudaMemset(d_tbl_entries, -1,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_keys, -1,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_nexts, -1,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_locks, 0,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_sizes, 0, sizeof(int) * batch_size) );

        unique_knl<<<batch_size, thread_num>>>(
            d_gathered_nodes, d_gathered_cols, L, tbl_capacity,
            d_tbl_entries, d_tbl_keys, d_tbl_nexts,
            d_tbl_locks, d_tbl_sizes);
        
        thrust::copy_if(thrust::device,
            d_tbl_keys, d_tbl_keys + tbl_capacity * batch_size,
            c_acts.d_nodes, filter(0));
        
        thrust::inclusive_scan(thrust::device,
            d_tbl_sizes, d_tbl_sizes + batch_size,
            c_acts.d_cols + 1);
    }

    void get_act_nodes(const float *d_inputs,
                       const CompactLabels &c_labels,
                       const int batch_size,
                       CompactActNodes &c_acts)
    {
        GPUTimer timer;
        timer.start();

        const int thread_num = 128;
        const int hash_block_num = (L + pack_num - 1) / pack_num;
        const int smem_size = sizeof(int) * K * bin_size * pack_num;
        get_hash_knl<<<hash_block_num, thread_num, smem_size>>>(
            d_bins, d_inputs, d_bucket_sizes, prev_node_num,
            tot_bin_size, L, K, bin_size, pack_num, batch_size,
            tbl_bucket_num, bucket_unit_size,
            d_hashed_bucket_ids, d_gathered_cols + 1);
        
        timer.record("\t[get_hash_knl] ");

        thrust::inclusive_scan(thrust::device,
            d_gathered_cols + 1, d_gathered_cols + 1 + L * batch_size,
            d_gathered_cols + 1);

        timer.record("\t[inclusive_scan] ");

        const int gather_block_num
            = (L * batch_size + thread_num - 1) / thread_num;
        gather_buckets_knl<<<gather_block_num, thread_num>>>(
            d_hashed_bucket_ids, d_buckets, d_gathered_cols,
            L, tbl_bucket_num, bucket_unit_size, L * batch_size,
            d_gathered_nodes);
        
        timer.record("\t[gather_buckets_knl] ");

        CUDA_CHECK( cudaMemset(d_tbl_entries, -1,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_keys, -1,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_nexts, -1,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_locks, 0,
                        sizeof(int) * tbl_capacity * batch_size) );
        CUDA_CHECK( cudaMemset(d_tbl_sizes, 0, sizeof(int) * batch_size) );

        unique_knl<<<batch_size, thread_num>>>(
            d_gathered_nodes, d_gathered_cols, L, tbl_capacity,
            d_tbl_entries, d_tbl_keys, d_tbl_nexts,
            d_tbl_locks, d_tbl_sizes);

        timer.record("\t[unique_knl] ");
        
        make_labels_act_knl<<<1, batch_size>>>(
            c_labels.d_nodes, c_labels.d_cols,
            batch_size, tbl_capacity,
            d_tbl_entries, d_tbl_keys, d_tbl_nexts,
            d_tbl_locks, d_tbl_sizes);
        
        timer.record("\t[make_labels_act_knl] ");

        thrust::copy_if(thrust::device,
            d_tbl_keys, d_tbl_keys + tbl_capacity * batch_size,
            c_acts.d_nodes, filter(0));
        
        timer.record("\t[copy_if] ");
        
        thrust::inclusive_scan(thrust::device,
            d_tbl_sizes, d_tbl_sizes + batch_size,
            c_acts.d_cols + 1);
        
        timer.record("\t[inclusive_scan] ");
    }
};