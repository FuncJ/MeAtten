#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <arm_neon.h>

#include <cblas.h>
#include "meformer.h"

#define REPEAT 50

// benchmark 2
int main(int argc, const char *argv[])
{
	int m,n,k;
	{
		int len=0;
		m=160,n=64,k=(128*4);

		float *a = (float *) malloc (sizeof(float)*m*k);
		float *a_panel = (float *) malloc (sizeof(float)*m*k);
		float *a_exp = (float *) malloc (sizeof(float)*m*k);
		float *b = (float *) malloc (sizeof(float)*k*n);
		float *buffer_b = (float *) malloc (sizeof(float)*k*r2);
		float *c = (float *) malloc (sizeof(float)*m*n);
		float *c_bak = (float *) malloc (sizeof(float)*m*n);

		float *max_per_line = (float *) malloc (sizeof(float)*m*4);
		float *exp_sum_per_line = (float *) malloc (sizeof(float)*m*1);

		float *flush_llc = (float *) malloc (sizeof(float)*4096*4096);

		randMatrix(m, k, a, k);
		randMatrix(k, n, b, n);

		float alpha=1.0;
		for(int i=0; i<5; i++) {
			// fused_exp_sum_scorexv_kernel(
			// 			m, n, k,
			// 			a_panel, b, buffer_b, c, n,
			// 			max_per_line, exp_sum_per_line);

			fused_exp_sum_scorexv_norm_kernel_1(1,
							m, n, k, k,
							a_panel, b, c, n,
							max_per_line, exp_sum_per_line);
			// fused_exp_sum_scorexv_norm_kernel_2(0,
			// 				m, n, 10*b3, k,
			// 				a_panel+(r1*b3)*10, b+(b3*n)*10, c, n,
			// 				max_per_line+m, max_per_line,
			// 				exp_sum_per_line);
			// fused_exp_sum_scorexv_norm_kernel_2(0,
			// 				m, n, 10*b3, k,
			// 				a_panel+(r1*b3)*20, b+(b3*n)*20, c, n,
			// 				max_per_line+2*m, max_per_line+m,
			// 				exp_sum_per_line);
			// fused_exp_sum_scorexv_norm_kernel_2(1,
			// 				m, n, 10*b3, k,
			// 				a_panel+(r1*b3)*30, b+(b3*n)*30, c, n,
			// 				max_per_line+3*m, max_per_line+2*m,
			// 				exp_sum_per_line);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a_exp, k, b, n, 0.0, c_bak, n);
		}

		exp_init
		double start, cost=0;
		for(int r=0; r<REPEAT; r++) {
			packMatrix(m, k, a, k, a_panel);
			copyMatrix(m, k, a, k, a_exp, k);

			{
				for(int i=0; i<m; i++)
					// max_per_line[i]=((i+1)+0.1);
					max_per_line[i]=(0.1);
				// for(int i=0; i<m; i++)
				// 	// max_per_line[m+i]=((i+1)+0.2);
				// 	max_per_line[m+i]=(0.2);
				// for(int i=0; i<m; i++)
				// 	// max_per_line[m*2+i]=((i+1)+0.3);
				// 	max_per_line[2*m+i]=(0.3);
				// for(int i=0; i<m; i++)
				// 	// max_per_line[m*3+i]=((i+1)+0.4);
				// 	max_per_line[3*m+i]=(0.4);

				for(int i=0; i<m; i++)
					exp_sum_per_line[i]=0.0;

				// for(int i=0; i<4*m; i++)
				// 	max_per_line[i]=0.1;
				// for(int i=0; i<4*m; i++)
				// 	exp_sum_per_line[i]=0;
			}

			for(int i=0; i<4096*4096; i++)
				flush_llc[i]=0.1;

			start = dClock();
			// fused_exp_sum_scorexv_kernel(
			// 			m, n, k,
			// 			a_panel, b, buffer_b, c, n,
			// 			max_per_line, exp_sum_per_line);

			fused_exp_sum_scorexv_norm_kernel_1(1,
							m, n, k, k,
							a_panel, b, c, n,
							max_per_line, exp_sum_per_line);
			// fused_exp_sum_scorexv_norm_kernel_2(0,
			// 				m, n, 10*b3, k,
			// 				a_panel+(r1*b3)*10, b+(b3*n)*10, c, n,
			// 				max_per_line+m, max_per_line,
			// 				exp_sum_per_line);
			// fused_exp_sum_scorexv_norm_kernel_2(0,
			// 				m, n, 10*b3, k,
			// 				a_panel+(r1*b3)*20, b+(b3*n)*20, c, n,
			// 				max_per_line+2*m, max_per_line+m,
			// 				exp_sum_per_line);
			// fused_exp_sum_scorexv_norm_kernel_2(1,
			// 				m, n, 10*b3, k,
			// 				a_panel+(r1*b3)*30, b+(b3*n)*30, c, n,
			// 				max_per_line+3*m, max_per_line+2*m,
			// 				exp_sum_per_line);
			cost += (dClock()-start);

			{
				for(int i=0; i<m; i++)
					// max_per_line[i]=((i+1)+0.3);
					max_per_line[i]=(0.1);
				for(int i=0; i<m; i++)
					exp_sum_per_line[i]=0.0;

				// for(int i=0; i<4*m; i++)
				// 	max_per_line[i]=0.1;
				// for(int i=0; i<4*m; i++)
				// 	exp_sum_per_line[i]=0;
			}

			{
				float *addr_a=a_exp;
				
				// minus max, exp, sum of line
				for(int i=0; i<m; i+=5) {
					for(int ii=i; ii<i+5; ii++) {
						v_max=vld1q_dup_f32(max_per_line+ii);
						v_sum=vld1q_dup_f32(exp_sum_per_line+ii);

						int ks=(k/4);
						while(ks>0) {
							vx=vld1q_f32(addr_a);
							vx=vsubq_f32(vx, v_max);
							exp_4
							vst1q_f32(addr_a, vf);
							v_sum=vaddq_f32(v_sum, vf);
							addr_a+=4;
							ks--;
						}
						v_sum=vmovq_n_f32(vaddvq_f32(v_sum));
						vst1q_lane_f32(exp_sum_per_line+ii, v_sum, 0);
					}
				}
				
				// norm
				addr_a=a_exp;
				for(int i=0; i<m; i+=5) {
					for(int ii=i; ii<i+5; ii++) {
						float32x4_t valpha=vld1q_dup_f32(exp_sum_per_line+ii);
						float32x4_t vone=vmovq_n_f32(1.0);
						valpha=vdivq_f32(vone, valpha);

						int ks=(k/4);
						while (ks>0)
						{
							vx=vld1q_f32(addr_a);
							vx=vmulq_f32(vx, valpha);
							vst1q_f32(addr_a, vx);
							addr_a+=4;
							ks--;
						}
					}
				}
			}
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a_exp, k, b, n, 0.0, c_bak, n);
		}

		// printPanel(m, k, a_panel);
		// printMatrixRowMajor(m, k, a_exp, k);
		// printMatrixRowMajor(m, n, c, n);
		// printMatrixRowMajor(m, n, c_bak, n);

		compareMatrix(m, n, c, n,
							c_bak, n);
		printf("m=%d,n=%d,k=%d,cost=%.9lf,gflops=%.2lf\n", m,n,k, cost,
												REPEAT*2.0*m*n*k/cost/1.0e9);
	}

	return 0;
}
