#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include <arm_neon.h>

#ifdef __cplusplus
	extern "C" {
#endif

#define min(a, b) ((a)<(b)?(a):(b))

#define exp_init \
    float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p+23f); \
    float32x4_t vzero_cutoff = vmovq_n_f32(-0x1.9FE368p+6f); \
    float32x4_t vinf_cutoff = vmovq_n_f32(0x1.62E42Ep+6f); \
    float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f); \
    float32x4_t vminus_ln2_hi = vmovq_n_f32(-0x1.62E43p-1f); \
    float32x4_t vminus_ln2_lo = vmovq_n_f32(0x1.05C61p-29f); \
    float32x4_t vplus_inf = vmovq_n_f32(INFINITY); \
	\
    float32x4_t vc1 = vmovq_n_f32(0x1.FFFFF6p-1f); \
    float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f); \
    float32x4_t vc3 = vmovq_n_f32(0x1.555A80p-3f); \
    float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f); \
    float32x4_t vc5 = vmovq_n_f32(0x1.0F9F9Cp-7f); \
    \
    int32x4_t vmin_exponent = vmovq_n_s32(INT32_C(0xC1000000)); \
    int32x4_t vmax_exponent = vmovq_n_s32(INT32_C(0x3F800000)); \
	int32x4_t vdefault_exponent = vmax_exponent; \
    \
	float32x4_t vx, v_max, v_sum, vn, vsn, vso, vt, vp, vf; \
	int32x4_t veo, ven;

// input:vx, output:vf=exp(vx)
#define exp_4 \
	vn = vfmaq_f32(vmagic_bias, vx, vlog2e); \
	veo = vshlq_n_s32(vreinterpretq_s32_f32(vn), 23); \
	ven = vmaxq_s32(veo, vmin_exponent); \
	ven = vminq_s32(ven, vmax_exponent); \
	veo = vsubq_s32(veo, ven); \
	vsn = vreinterpretq_f32_s32(vaddq_s32(ven, vdefault_exponent)); \
	vso = vreinterpretq_f32_s32(vaddq_s32(veo, vdefault_exponent)); \
	vn = vsubq_f32(vn, vmagic_bias); \
	vt = vfmaq_f32(vx, vn, vminus_ln2_hi); \
	vt = vfmaq_f32(vt, vn, vminus_ln2_lo); \
	vp = vfmaq_f32(vc4, vc5, vt); \
	vp = vfmaq_f32(vc3, vp, vt); \
	vp = vfmaq_f32(vc2, vp, vt); \
	vp = vfmaq_f32(vc1, vp, vt); \
	vt = vmulq_f32(vt, vso); \
	vf = vmulq_f32(vsn, vfmaq_f32(vso, vt, vp)); \
	vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32 (vf), vcltq_f32(vx, vzero_cutoff))); \
	vf = vbslq_f32(vcgtq_f32(vx, vinf_cutoff), vplus_inf, vf);

static inline float *fastMalloc(int elems)
{
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, 64, elems*sizeof(float));
    assert(ret == 0);

    return (float *)ptr;
}

static inline void randMatrix(int m, int n, float *x, int ldx)
{
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            x[i*ldx+j] = (float)drand48()*2.0-1.0;
            // x[i*ldx+j] = (i+1);
            // x[i*ldx+j] = (i+1)*0.1;
}

static inline void compareMatrix(int m, int n, float *get, int ldg, float *expect, int lde)
{
    int flag=0;
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++) {
            float diff = (get[i*ldg+j]-expect[i*lde+j]);
            if((diff > 1.0e-3) || (diff < -1.0e-3)) {
                printf("(%d, %d), diff=%lf, get=%lf, expect=%lf\n", i, j, diff,
                                                get[i*ldg+j], expect[i*lde+j]);
                flag = 1;
                // break;
            }
        }
        // if(flag == 1)
            // break;
    }
    if(flag == 0)
        printf("no diff\n");
}

static inline void printMatrixRowMajor(int m, int n, float *x, int ldx)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            printf("%.6lf ", x[i*ldx+j]);
        }
        printf("\n");
    }
    printf("\n");
}

static inline void printMatrixColMajor(int m, int n, float *x, int ldx)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            printf("%.6lf ", x[i+j*ldx]);
        }
        printf("\n");
    }
    printf("\n");
}

static inline void copyMatrix(int m, int n, float *a, int lda, float *b, int ldb)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            b[i*ldb+j] = a[i*lda+j];
        }
    }
}

static inline void packMatrix(int m, int n, float *a, int lda, float *a_panel)
{
    // float *a0, *a1, *a2, *a3, *a4;
    // float *b=a_panel;
    
    // for(int i=0; i<m; i+=5) {
    //     a0=a+i*lda;
    //     a1=a0+lda;
    //     a2=a1+lda;
    //     a3=a2+lda;
    //     a4=a3+lda;

    //     for(int j=0; j<n; j+=4) {
    //         *b=*a0, b++, a0++;  *b=*a0, b++, a0++;  *b=*a0, b++, a0++;  *b=*a0, b++, a0++;
    //         *b=*a1, b++, a1++;  *b=*a1, b++, a1++;  *b=*a1, b++, a1++;  *b=*a1, b++, a1++;
    //         *b=*a2, b++, a2++;  *b=*a2, b++, a2++;  *b=*a2, b++, a2++;  *b=*a2, b++, a2++;
    //         *b=*a3, b++, a3++;  *b=*a3, b++, a3++;  *b=*a3, b++, a3++;  *b=*a3, b++, a3++;
    //         *b=*a4, b++, a4++;  *b=*a4, b++, a4++;  *b=*a4, b++, a4++;  *b=*a4, b++, a4++;
    //     }
    // }

	float *a0, *a1, *a2, *a3, *a4;
	float32x4_t a0_v, a1_v, a2_v, a3_v, a4_v;
	float *b=a_panel;

	for(int i=0; i<m; i+=5) {
		a0=a+i*lda;
		a1=a0+lda;
		a2=a1+lda;
		a3=a2+lda;
		a4=a3+lda;

		for(int j=0; j<n; j+=16) {
			a0_v=vld1q_f32(a0), vst1q_f32(b, a0_v), a0+=4, b+=4;
			a1_v=vld1q_f32(a1), vst1q_f32(b, a1_v), a1+=4, b+=4;
			a2_v=vld1q_f32(a2), vst1q_f32(b, a2_v), a2+=4, b+=4;
			a3_v=vld1q_f32(a3), vst1q_f32(b, a3_v), a3+=4, b+=4;
			a4_v=vld1q_f32(a4), vst1q_f32(b, a4_v), a4+=4, b+=4;

			a0_v=vld1q_f32(a0), vst1q_f32(b, a0_v), a0+=4, b+=4;
			a1_v=vld1q_f32(a1), vst1q_f32(b, a1_v), a1+=4, b+=4;
			a2_v=vld1q_f32(a2), vst1q_f32(b, a2_v), a2+=4, b+=4;
			a3_v=vld1q_f32(a3), vst1q_f32(b, a3_v), a3+=4, b+=4;
			a4_v=vld1q_f32(a4), vst1q_f32(b, a4_v), a4+=4, b+=4;

			a0_v=vld1q_f32(a0), vst1q_f32(b, a0_v), a0+=4, b+=4;
			a1_v=vld1q_f32(a1), vst1q_f32(b, a1_v), a1+=4, b+=4;
			a2_v=vld1q_f32(a2), vst1q_f32(b, a2_v), a2+=4, b+=4;
			a3_v=vld1q_f32(a3), vst1q_f32(b, a3_v), a3+=4, b+=4;
			a4_v=vld1q_f32(a4), vst1q_f32(b, a4_v), a4+=4, b+=4;

			a0_v=vld1q_f32(a0), vst1q_f32(b, a0_v), a0+=4, b+=4;
			a1_v=vld1q_f32(a1), vst1q_f32(b, a1_v), a1+=4, b+=4;
			a2_v=vld1q_f32(a2), vst1q_f32(b, a2_v), a2+=4, b+=4;
			a3_v=vld1q_f32(a3), vst1q_f32(b, a3_v), a3+=4, b+=4;
			a4_v=vld1q_f32(a4), vst1q_f32(b, a4_v), a4+=4, b+=4;
		}
	}
}

static inline void printPanel(int m, int n, float *a)
{
    float *a0, *a1, *a2, *a3, *a4;
    for(int i=0; i<m; i+=5) {
        a0=a+i*n;
        a1=a0+4;
        a2=a1+4;
        a3=a2+4;
        a4=a3+4;

        for(int j=0; j<n; j+=4) {
            printf("%.6lf %.6lf %.6lf %.6lf ", a0[0], a0[1], a0[2], a0[3]);
            a0+=20;
        }
        printf("\n");
        for(int j=0; j<n; j+=4) {
            printf("%.6lf %.6lf %.6lf %.6lf ", a1[0], a1[1], a1[2], a1[3]);
            a1+=20;
        }
        printf("\n");
        for(int j=0; j<n; j+=4) {
            printf("%.6lf %.6lf %.6lf %.6lf ", a2[0], a2[1], a2[2], a2[3]);
            a2+=20;
        }
        printf("\n");
        for(int j=0; j<n; j+=4) {
            printf("%.6lf %.6lf %.6lf %.6lf ", a3[0], a3[1], a3[2], a3[3]);
            a3+=20;
        }
        printf("\n");
        for(int j=0; j<n; j+=4) {
            printf("%.6lf %.6lf %.6lf %.6lf ", a4[0], a4[1], a4[2], a4[3]);
            a4+=20;
        }
        printf("\n");
    }
    printf("\n");
}

static inline void addMatrix(int m, int n, float *dst, int lddst, float *src, int ldsrc)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            dst[i*n+j] += src[i*n+j];
        }
    }
}

static double gtod_ref_time_sec=0.0;
static double dClock()
{
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    if (gtod_ref_time_sec==0.0)
		gtod_ref_time_sec=(double)tv.tv_sec;

    norm_sec=(double)tv.tv_sec-gtod_ref_time_sec;
    the_time=norm_sec+tv.tv_usec*1.0e-6;

    return the_time;
}

#ifdef __cplusplus
	}
#endif

#endif