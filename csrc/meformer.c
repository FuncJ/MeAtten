#include <math.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include "meformer.h"

void scaled_dot_product_attention_kernel(long m, long k, long l, long n,
						float *scale,
						float *q, float *kt, float *buffer_kt, float *score, long lds,
						float *mask, long ldm,
						float *v, float *buffer_v, float *out, long ldo,
						float *max_4_per_line,
						float *exp_sum_per_line)
{
	fused_scalexqxkt_mask_max_kernel(m, l, k, scale, q, kt, buffer_kt, score, lds, mask, ldm, max_4_per_line);
	fused_exp_sum_scorexv_kernel(m, n, l, score, v, buffer_v, out, ldo, max_4_per_line,exp_sum_per_line);
}

// parallel batch and seq_len
// q_data:         [n_batch, n_head, seq_len, head_dim]
// kt_data:        [n_batch, n_head, seq_len, head_dim]    buffer_kt_out :       [n_threads, head_dim, r2]
// score_data_out: [n_batch, n_head, seq_len, seq_len]     mask_data:            [seq_len, seq_len]
//   													   max_per_line_out:     [n_batch, n_head, seq_len*2]
// 	      												   exp_sum_per_line_out: [n_batch, n_head, seq_len]
// v_data:         [n_batch, n_head, seq_len, head_dim]
// out:            [n_batch, n_head, seq_len, head_dim]
#define MAX_THREADS 64
double start[MAX_THREADS],
	bmm_1_cost[MAX_THREADS]={0},   bmm_2_cost[MAX_THREADS]={0},
	bmm_1_ops[MAX_THREADS]={0},    bmm_2_ops[MAX_THREADS]={0},
	bmm_1_gflops[MAX_THREADS]={0}, bmm_2_gflops[MAX_THREADS]={0};

void scaled_dot_product_attention(long n_batch, long n_head, long seq_len, long head_dim,
						float *q_data, float *kt_data, float *buffer_kt_out/*todo: buffer management*/, float *score_data_out/*todo: buffer management*/,
						float *mask_data,
						float *v_data, float *out_data, long ldo,
						float *max_per_line_out/*todo: buffer management*/,
						float *exp_sum_per_line_out/*todo: buffer management*/)
{
	long batch_size, m, k, l, n;
	batch_size=(n_batch*n_head), m=seq_len, k=head_dim, l=seq_len, n=head_dim;

	long size_head, lds, size_score, size_buffer;
	size_head=(seq_len*head_dim), lds=min(l, b2)/*careful*/, size_score=(seq_len*lds)/*careful*/, size_buffer=(head_dim*r2);

	float scale=1.0/sqrt(head_dim);

	int num_T=64;
	#pragma omp parallel
		num_T=omp_get_num_threads();

	// work partition: determine block size
	int mb, intra_partitions, partitions;
	mb=b1, intra_partitions=( (m+mb-1)/mb ), partitions=( intra_partitions*batch_size );
	int res;
	res=( partitions%num_T );

	while(res>=8 && mb>=25)/*careful*/ {
		mb-=5;
		intra_partitions=( (m+mb-1)/mb );
		partitions=( intra_partitions*batch_size );
		res=( partitions%num_T );
	}
	mb=( (m+intra_partitions-1)/intra_partitions )/*careful*/;
	mb=( (mb+5-1)/5*5 )/*careful*/;
	// printf("mb:%d, intra_partitions:%d partitions:%d\n", mb, intra_partitions, partitions);

	// work distribution: determine thread num and data address
	#pragma omp parallel for num_threads(num_T) schedule(static, 1)
	for(int part=0; part<partitions; part++) {
		float *q, *kt, *buffer_kt, *score,
			*mask,
			*v, *out,
			*max_per_line, *max_for_update,
			*exp_sum_per_line;

		int tid=omp_get_thread_num();

		int bs=(part/intra_partitions), intra_index=(part%intra_partitions);
		int thread_num=(part%num_T);

		if(tid==thread_num) {

			long is, is_stride;
			is=(intra_index*mb), is_stride=min(m-is, mb);

			exp_sum_per_line=((exp_sum_per_line_out+bs*m)+is);
			memset(exp_sum_per_line, 0, sizeof(float)*is_stride); /*todo: simd*/

			// if(is_stride==0 | is%r1!=0 | is_stride%r1!=0 )
			// 	printf("is_stride:%d, is:%d is_mod_mb:%d is_stride_mod_mb:%d\n", is_stride, is, is%r1, is_stride%r1);
			assert(is_stride!=0 && is%r1==0 && is_stride%r1==0);

			long num_block=((l+b2-1)/b2), block_index=0;

			for(long js=0; js<l; js+=b2) {

				long js_stride=min(l-js, b2);
				block_index++;

				char is_even=( ((js/b2)%2==0)?1:0 );

				// 1 scale*q*kt+mask (fuse max_per_line)
				{
					q=((q_data+bs*size_head)+is*k), kt=((kt_data+bs*size_head)+k*js), buffer_kt=(buffer_kt_out+(tid*size_buffer)),
											// score=((score_data_out+bs*size_score)+is*lds);
											size_score=(mb*lds); score=( score_data_out+(tid*size_score) );
					mask=(mask_data+is*l+js);
					max_per_line=((max_per_line_out+bs*2*m)+(!is_even*m)+is);
					memset(max_per_line, 0, sizeof(float)*is_stride); /*todo: simd*/

					int eventset=-1;
					// if(tid==1) {
						// trace_cache_start(&eventset);
						// start[tid]=dClock();
					// }

					fused_scalexqxkt_mask_max_kernel(is_stride, js_stride, k,
						&scale,
						q, kt, buffer_kt, score, lds,
						mask, l,
						max_per_line);

					// if(tid==1) {
						// trace_cache_stop(&eventset, "bmm_1");
						// bmm_1_cost[tid]+=(dClock()-start[tid]);
						// bmm_1_ops[tid]+=(2.0*is_stride*js_stride*k);
					// }
				}

				// 2 score*v+norm
				{
					v=((v_data+bs*size_head)+js*n), out=((out_data+bs*size_head)+is*n);
					max_for_update=((max_per_line_out+bs*2*m)+(is_even*m)+is);
					exp_sum_per_line=((exp_sum_per_line_out+bs*m)+is);

					// int eventset=-1;
					// if(tid==1) {
						// trace_cache_start(&eventset);
						// start[tid]=dClock();
					// }

					if(block_index==1)
					{
						fused_exp_sum_scorexv_norm_kernel_1((block_index==num_block),
									is_stride, n, js_stride, lds,
									score, v, out, n,
									max_per_line, exp_sum_per_line);
					}
					else
					{
						fused_exp_sum_scorexv_norm_kernel_2((block_index==num_block),
									is_stride, n, js_stride, lds,
									score, v, out, n,
									max_per_line, max_for_update,
									exp_sum_per_line);
					}

					// if(tid==1) {
						// trace_cache_stop(&eventset, "bmm_2");
						// bmm_2_cost[tid]+=(dClock()-start[tid]);
						// bmm_2_ops[tid]+=(2.0*is_stride*n*js_stride);
					// }
				}
			}
		}
	}

	// for(int i=0; i<num_T; i++) {
	// 	bmm_1_gflops[i]=(bmm_1_ops[i]/bmm_1_cost[i]/1.0e9);
	// 	bmm_2_gflops[i]=(bmm_2_ops[i]/bmm_2_cost[i]/1.0e9);
	// 	printf("tid=%d, bmm_1_gflops=,%lf, bmm_2_gflops=,%lf\n",
	// 					i, bmm_1_gflops[i], bmm_2_gflops[i]);
	// }
}


// // parallel batch
// // q_data:         [n_batch, n_head, seq_len, head_dim]
// // kt_data:        [n_batch, n_head, seq_len, head_dim]    buffer_kt_out :       [n_threads, head_dim, r2]
// // score_data_out: [n_batch, n_head, seq_len, seq_len]     mask_data:            [seq_len, seq_len]
// //   													   max_per_line_out:     [n_batch, n_head, seq_len*2]
// // 	      												   exp_sum_per_line_out: [n_batch, n_head, seq_len]
// // v_data:         [n_batch, n_head, seq_len, head_dim]
// // out:            [n_batch, n_head, seq_len, head_dim]
// void scaled_dot_product_attention(long n_batch, long n_head, long seq_len, long head_dim,
// 						float *q_data, float *kt_data, float *buffer_kt_out/*todo: buffer management*/, float *score_data_out,
// 						float *mask_data,
// 						float *v_data, float *out_data, long ldo,
// 						float *max_per_line_out/*todo: buffer management*/,
// 						float *exp_sum_per_line_out/*todo: buffer management*/)
// {
// 	long batch_size, m, k, l, n;
// 	batch_size=(n_batch*n_head), m=seq_len, k=head_dim, l=seq_len, n=head_dim;

// 	long size_head, lds, size_score, size_buffer;
// 	size_head=(seq_len*head_dim), lds=min(l, b2), size_score=(seq_len*lds)/*careful*/, size_buffer=(head_dim*r2);

// 	float scale=sqrt(head_dim);

// 	int num_T=0;
// 	#pragma omp parallel
// 		num_T=omp_get_num_threads();

// 	#pragma omp parallel for num_threads(num_T)
// 	for(long bs=0; bs<batch_size; bs++) {
// 		int tid=omp_get_thread_num();

// 		float *q, *kt, *buffer_kt, *score,
// 			*mask,
// 			*v, *out,
// 			*max_per_line, *max_for_update,
// 			*exp_sum_per_line;

// 		for(long is=0; is<m; is+=b1) {

// 			long is_stride=min(m-is, b1); /*todo: tuning b1,b2,b3 in context*/
// 			long num_block=((l+b2-1)/b2), block_index=0;

// 			for(long js=0; js<l; js+=b2) {

// 				long js_stride=min(l-js, b2);
// 				block_index++;

// 				char is_even=( ((js/b2)%2==0)?1:0 );

// 				// 1 scale*q*kt+mask (fuse max_per_line)
// 				{
// 					q=((q_data+bs*size_head)+is*k), kt=((kt_data+bs*size_head)+k*js), buffer_kt=(buffer_kt_out+(tid*size_buffer)),
// 											score=((score_data_out+bs*size_score)+is*lds/*careful*/);
// 					mask=(mask_data+is*l+js);
// 					max_per_line=((max_per_line_out+bs*2*m)+(!is_even*m)+is);
// 					memset(max_per_line, 0, sizeof(float)*is_stride); /*todo: simd*/

// 					fused_scalexqxkt_mask_max_kernel(is_stride, js_stride, k,
// 						&scale,
// 						q, kt, buffer_kt, score, lds/*careful*/,
// 						mask, l,
// 						max_per_line);
// 				}

// 				// 2 score*v+norm
// 				{
// 					v=((v_data+bs*size_head)+js*n), out=((out_data+bs*size_head)+is*n);
// 					max_for_update=((max_per_line_out+bs*2*m)+(is_even*m)+is);
// 					exp_sum_per_line=((exp_sum_per_line_out+bs*m)+is);

// 					if(block_index==1)
// 					{
// 						fused_exp_sum_scorexv_norm_kernel_1((block_index==num_block),
// 									is_stride, n, js_stride, lds/*careful*/,
// 									score, v, out, n,
// 									max_per_line, exp_sum_per_line);
// 					}
// 					else
// 					{
// 						fused_exp_sum_scorexv_norm_kernel_2((block_index==num_block),
// 									is_stride, n, js_stride, lds/*careful*/,
// 									score, v, out, n,
// 									max_per_line, max_for_update,
// 									exp_sum_per_line);
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// // no parallelism
// // q_data:         [n_batch, n_head, seq_len, head_dim]
// // kt_data:        [n_batch, n_head, seq_len, head_dim]    buffer_kt_out :       [n_threads, head_dim, r2]
// // score_data_out: [n_batch, n_head, seq_len, seq_len]     mask_data:            [seq_len, seq_len]
// //   													   max_per_line_out:     [n_batch, n_head, seq_len*2]
// // 	      												   exp_sum_per_line_out: [n_batch, n_head, seq_len]
// // v_data:         [n_batch, n_head, seq_len, head_dim]
// // out:            [n_batch, n_head, seq_len, head_dim]
// void scaled_dot_product_attention(long n_batch, long n_head, long seq_len, long head_dim,
// 						float *q_data, float *kt_data, float *buffer_kt_out, float *score_data_out,
// 						float *mask_data,
// 						float *v_data, float *out_data, long ldo,
// 						float *max_per_line_out,
// 						float *exp_sum_per_line_out)
// {
// 	long batch_size, m, k, l, n;
// 	batch_size=(n_batch*n_head), m=seq_len, k=head_dim, l=seq_len, n=head_dim;

// 	long size_head, size_score;
// 	size_head=(seq_len*head_dim), size_score=(seq_len*seq_len)/*careful*/;

// 	float scale=sqrt(head_dim);

// 	for(long bs=0; bs<batch_size; bs++) {

// 		float *q, *kt, *buffer_kt, *score,
// 			*mask,
// 			*v, *out,
// 			*max_per_line, *max_for_update,
// 			*exp_sum_per_line;

// 		for(long is=0; is<m; is+=b1) {

// 			long is_stride=min(m-is, b1); /*todo: b1,b2,b3 in context*/
// 			long num_block=((l+b2-1)/b2), block_index=0;

// 			for(long js=0; js<l; js+=b2) {

// 				long js_stride=min(l-js, b2);
// 				block_index++;

// 				char is_even=( ((js/b2)%2==0)?1:0 );

// 				// 1 scale*q*kt+mask (fuse max_per_line)
// 				{
// 					q=((q_data+bs*size_head)+is*k), kt=((kt_data+bs*size_head)+k*js), buffer_kt=buffer_kt_out/*todo: buffer management*/,
// 											score=((score_data_out+bs*size_score)+is*l/*todo: buffer management*/+(r1*b2)*(js/b2));
// 					mask=(mask_data+is*l+js);
// 					max_per_line=((max_per_line_out+bs*2*m)+(!is_even*m)+is);
// 					memset(max_per_line, 0, sizeof(float)*is_stride); /*todo: simd*/

// 					fused_scalexqxkt_mask_max_kernel(is_stride, js_stride, k,
// 						&scale,
// 						q, kt, buffer_kt, score, l/*todo*/,
// 						mask, l,
// 						max_per_line);
// 				}

// 				// 2 score*v+norm
// 				{
// 					v=((v_data+bs*size_head)+js*n);
// 					out=((out_data+bs*size_head)+is*n);
// 					max_for_update=((max_per_line_out+bs*2*m)+(is_even*m)+is);
// 					exp_sum_per_line=((exp_sum_per_line_out+bs*m)+is);

// 					if(block_index==1)
// 					{
// 						fused_exp_sum_scorexv_norm_kernel_1((block_index==num_block),
// 									is_stride, n, js_stride, l/*todo*/,
// 									score, v, out, n,
// 									max_per_line, exp_sum_per_line);
// 					}
// 					else
// 					{
// 						fused_exp_sum_scorexv_norm_kernel_2((block_index==num_block),
// 									is_stride, n, js_stride, l/*todo*/,
// 									score, v, out, n,
// 									max_per_line, max_for_update,
// 									exp_sum_per_line);
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// // no batch_size
// void scaled_dot_product_attention(long n_batch, long n_head, long seq_len, long head_dim,
// 						float *q_data, float *kt_data, float *buffer_kt_out, float *score_data_out,
// 						float *mask_data,
// 						float *v_data, float *out_data, long ldo,
// 						float *max_per_line_out,
// 						float *exp_sum_per_line_out)
// {
// 	long m, k, l, n;
// 	m=seq_len, k=head_dim, l=seq_len, n=head_dim;

// 	float *q, *kt, *buffer_kt, *score,
// 		*mask,
// 		*v, *out,
// 		*max_per_line, *max_for_update,
// 		*exp_sum_per_line;

// 	float scale=sqrt(head_dim);

// 	for(long is=0; is<m; is+=b1)
// 	{
// 		long is_stride=min(m-is, b1); // todo: adjust for performance
// 		long num_block=((l+b2-1)/b2), block_index=0;

// 		for(long js=0; js<l; js+=b2)
// 		{
// 			long js_stride=min(l-js, b2); // todo: adjust for performance
// 			block_index++;

// 			char is_even=( ((js/b2)%2==0)?1:0 );

// 			// 1 scale*q*kt+mask (fuse max_per_line)
// 			{
// 				q=(q_data+is*k), kt=(kt_data+k*js), buffer_kt=buffer_kt_out,
// 										score=(score_data_out+is*l/*todo*/+(r1*b2)*(js/b2));
// 				mask=(mask_data+is*l+js);
// 				max_per_line=(max_per_line_out+(!is_even*m)+is); // round robin
// 				memset(max_per_line, 0, sizeof(float)*is_stride);

// 				fused_scalexqxkt_mask_max_kernel(is_stride, js_stride, k,
// 					&scale,
// 					q, kt, buffer_kt, score, l/*todo*/,
// 					mask, l,
// 					max_per_line);
// 			}

// 			// 2 score*v+norm
// 			{
// 				v=(v_data+js*n);
// 				out=(out_data+is*n);
// 				max_for_update=(max_per_line_out+(is_even*m)+is);
// 				exp_sum_per_line=(exp_sum_per_line_out+is);

// 				if(block_index==1)
// 				{
// 					fused_exp_sum_scorexv_norm_kernel_1((block_index==num_block),
// 								is_stride, n, js_stride, l/*todo*/,
// 								score, v, out, n,
// 								max_per_line, exp_sum_per_line);
// 				}
// 				else
// 				{
// 					fused_exp_sum_scorexv_norm_kernel_2((block_index==num_block),
// 								is_stride, n, js_stride, l/*todo*/,
// 								score, v, out, n,
// 								max_per_line, max_for_update,
// 								exp_sum_per_line);
// 				}
// 			}
// 		}
// 	}
// }

