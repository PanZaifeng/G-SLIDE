#pragma once
#include "CscActNodes.h"

__global__ void get_dense_acts_knl(CscActNodes csc_acts, const int node_num,
                                   const int batch_size);

__global__ void relu_fwd_slide_in_knl(const CscActNodes csc_inputs,
                                      const float *d_weights_colmajor,
                                      const float *d_biases,
                                      const int weight_row_num,
                                      const int max_out_num,
                                      CscActNodes csc_outputs);

__global__ void softmax_fwd_bp_rowmajor_slide_in_knl(
    const CscActNodes csc_inputs, const float *d_weights_rowmajor,
    const float *d_biases, const CscActNodes cmprs_labels,
    const int weight_col_num, const int max_out_num, const int max_label_num,
    CscActNodes csc_outputs, float *d_cmpr_bp_deltas);

__global__ void softmax_fwd_bp_rowmajor_slide_out_knl(
    const CscActNodes csc_inputs, const float *d_weights_rowmajor,
    const float *d_biases, const CscActNodes cmprs_labels,
    const int weight_col_num, const int max_in_num, const int max_label_num,
    CscActNodes csc_outputs, float *d_cmprs_bp_deltas);

__global__ void softmax_fwd_bp_rowmajor_all_sm_knl(
    const CscActNodes csc_inputs, const float *d_weights_rowmajor,
    const float *d_biases, const CscActNodes cmprs_labels,
    const int weight_col_num, const int max_in_num, const int max_out_num,
    const int max_label_num, CscActNodes csc_outputs, float *d_cmprs_bp_deltas);

__global__ void bp_knl(const CscActNodes csc_acts, const CscActNodes csc_prev,
                       const float *d_weights_colmajor,
                       const float *d_cmpr_bp_deltas, const int weight_row_num,
                       const int max_act_num, float *d_cmpr_prev_bp_deltas,
                       float *d_adam_ts, float *d_bias_adam_ts);

__global__ void bp_rowmajor_knl(const CscActNodes csc_acts,
                                const CscActNodes csc_prev,
                                const float *d_weights_rowmajor,
                                const float *d_cmpr_bp_deltas,
                                const int weight_col_num, const int max_act_num,
                                float *d_cmpr_prev_bp_deltas, float *d_adam_ts,
                                float *d_bias_adam_ts);

__global__ void bp_rowmajor_no_sm_knl(const CscActNodes csc_acts,
                                      const CscActNodes csc_prev,
                                      const float *d_weights_rowmajor,
                                      const float *d_cmprs_bp_deltas,
                                      const int weight_col_num,
                                      float *d_cmprs_prev_bp_deltas,
                                      float *d_adam_ts, float *d_bias_adam_ts);

__global__ void bp_rowmajor_slide_knl(
    const CscActNodes csc_acts, const CscActNodes csc_prev,
    const float *d_weights_rowmajor, const float *d_cmprs_bp_deltas,
    const int weight_col_num, const int max_prev_num,
    float *d_cmprs_prev_bp_deltas, float *d_adam_ts, float *d_bias_adam_ts);

__global__ void bp_first_layer_knl(const CscActNodes csc_acts,
                                   const CscActNodes csc_prev,
                                   const float *d_cmpr_bp_deltas,
                                   const int weight_row_num,
                                   const int max_act_num, float *d_adam_ts,
                                   float *d_bias_adam_ts);

__global__ void update_weights_knl(float *d_weights, float *d_adam_ts,
                                   float *d_adam_moms, float *d_adam_vels,
                                   const float lr, const int weight_size);
