#ifndef MEFORMER_H
#define MEFORMER_H

#include "utils.h"
#include <arm_neon.h>

#ifdef __cplusplus
	extern "C" {
#endif

// L1 = 64KB, L2 = 512KB, deprecate soon
#define MR 5
#define NR 16

// b1%r1==0;
// b3%r2==0; b2%b3==0
#define r1 5
#define r2 16

// #define b1 160
// #define b2 128
// #define b3 128

#define b1 160
#define b2 160
#define b3 128

void fused_scalexqxkt_mask_max_kernel(long m, long n, long k/*(head_dim==64)*/,
					float *scale, float *q, float *kt, float *buffer_kt, float *score, long lds,
					float *mask, long ldm,
					float *max_per_line);
void fused_exp_sum_scorexv_kernel(long m, long n/*(head_dim==64)*/, long k,
					float *score, float *v, float *buffer_v, float *out, long ldo,
					float *max_4_per_line, float *exp_sum_per_line);
void scaled_dot_product_attention_kernel(long m, long k, long l, long n,
						float *scale,
						float *q, float *kt, float *buffer_kt, float *score, long lds,
						float *mask, long ldm,
						float *v, float *buffer_v, float *out, long ldo,
						float *max_4_per_line,
						float *exp_sum_per_line);

void fused_exp_sum_scorexv_norm_kernel_1(long is_last_block,
					long m, long n/*(head_dim==64)*/, long k, long ldk/*careful,todo*/,
					float *score, float *v, float *out, long ldo,
					float *max_per_line, float *exp_sum_per_line);
void fused_exp_sum_scorexv_norm_kernel_2(long is_last_block,
					long m, long n/*(head_dim==64)*/, long k, long ldk/*careful,todo*/,
					float *score, float *v, float *out, long ldo,
					float *max_per_line, float *max_for_update,
					float *exp_sum_per_line);
void scaled_dot_product_attention(long n_batch, long n_head, long seq_len, long head_dim,
						float *q_data, float *kt_data, float *buffer_kt_out, float *score_data_out,
						float *mask_data,
						float *v_data, float *out_data, long ldo,
						float *max_per_line_out,
						float *exp_sum_per_line_out);

#ifdef __cplusplus
	}
#endif

#endif