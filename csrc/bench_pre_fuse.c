#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <arm_neon.h>

#include <cblas.h>
#include "meformer.h"


// benchmark 1
int main(int argc, const char *argv[])
{
	// float settings[] = {8, 24, 40, 56, 72, 88, 104, 120};
	long m,n,k;

	// for(int i=0; i<sizeof(settings)/sizeof(float); i++)
	{
		// m=n=k=settings[i];
		// m=170,n=256,k=64;
		m=120,n=(128*2),k=64;
		scanf("%d", &n);

		float *a = (float *) malloc (sizeof(float)*m*k);
		float *b = (float *) malloc (sizeof(float)*n*k);
		float *buffer_b = (float *) malloc (sizeof(float)*k*r2);
		float *c = (float *) malloc (sizeof(float)*m*n);
		float *c_bak = (float *) malloc (sizeof(float)*m*n);
		float *c_panel = (float *) malloc (sizeof(float)*m*n);

		float *mask = (float *) malloc (sizeof(float)*m*n);
		float *max_per_line = (float *) malloc (sizeof(float)*m*1);

		randMatrix(m, k, a, k);
		randMatrix(n, k, b, k);
		randMatrix(m, n, mask, n);

		float alpha=0.1;
		for(int i=0; i<5; i++) {
			// sgemm_nt(a, b, c, m, n, k, &alpha, buffer_b);
			// SGEMM_NT(c, a, b, m, n, k);

			fused_scalexqxkt_mask_max_kernel(m, n, k, &alpha, a, b, buffer_b, c, n,
													mask, n, max_per_line);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k,
													alpha, a, k, b, k, 0.0, c_bak, n);
			addMatrix(m, n, c_bak, n, mask, n);
		}

		double start, cost=0;
		int rep=50;
		float *flush_llc = (float *) malloc (sizeof(float)*4096*4096);

		for(int r=0; r<rep; r++) {
			memset(max_per_line, 0, sizeof(float)*m*1);

			for(int i=0; i<4096*4096; i++)
				flush_llc[i]=0.1;

			start = dClock();
			// sgemm_nt(a, b, c, m, n, k, &alpha, buffer_b);
			// SGEMM_NT(c, a, b, m, n, k);

			fused_scalexqxkt_mask_max_kernel(m, n, k, &alpha, a, b, buffer_b, c, n,
													mask, n, max_per_line);
			cost += (dClock()-start);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k,
													alpha, a, k, b, k, 0.0, c_bak, n);
			addMatrix(m, n, c_bak, n, mask, n);
		}

		// printMatrixRowMajor(m, k, a, k);
		// printMatrixColMajor(k, n, b, k);

		// printPanel(m, n, c);
		// printMatrixRowMajor(m, n, c_bak, n);

		packMatrix(m, n, c_bak, n, c_panel);
		compareMatrix(m, n, c, n, c_panel, n);

		// compareMatrix(m, n, c, n, c_bak, n);

		for(int i=0; i<m; i++) {
			// float max_f=FLT_MIN;
			// for(int j=0; j<n; j++) {
			// 	if(c_bak[i*n+j]>max_f)
			// 		max_f=c_bak[i*n+j];
			// }

			// float diff=(max_f-max_per_line[i]);
			// printf("line %d, max_f=%.6lf, max_per_line=%.6lf, %d\n", i, max_f, max_per_line[i],
			// 									(-1.0e-5<diff) && (diff<1.0e-5));
		}

		printf("m=%d,n=%d,k=%d,cost=%.9lf,gflops=%.2lf\n", m,n,k, cost, rep*2.0*m*n*k/cost/1.0e9);
	}

	return 0;
}
