#include "meformer.h"

void cblas_saxpy(int32_t N, float alpha, const float *X, int32_t incX,
							float *Y, int32_t incY);

static const float meformer_table_exp2_k_over_64[64] = {
	0x1.000000p+0f, 0x1.02C9A4p+0f, 0x1.059B0Ep+0f, 0x1.087452p+0f,
	0x1.0B5586p+0f, 0x1.0E3EC4p+0f, 0x1.11301Ep+0f, 0x1.1429AAp+0f,
	0x1.172B84p+0f, 0x1.1A35BEp+0f, 0x1.1D4874p+0f, 0x1.2063B8p+0f,
	0x1.2387A6p+0f, 0x1.26B456p+0f, 0x1.29E9E0p+0f, 0x1.2D285Ap+0f,
	0x1.306FE0p+0f, 0x1.33C08Cp+0f, 0x1.371A74p+0f, 0x1.3A7DB4p+0f,
	0x1.3DEA64p+0f, 0x1.4160A2p+0f, 0x1.44E086p+0f, 0x1.486A2Cp+0f,
	0x1.4BFDAEp+0f, 0x1.4F9B28p+0f, 0x1.5342B6p+0f, 0x1.56F474p+0f,
	0x1.5AB07Ep+0f, 0x1.5E76F2p+0f, 0x1.6247ECp+0f, 0x1.662388p+0f,
	0x1.6A09E6p+0f, 0x1.6DFB24p+0f, 0x1.71F75Ep+0f, 0x1.75FEB6p+0f,
	0x1.7A1148p+0f, 0x1.7E2F34p+0f, 0x1.82589Ap+0f, 0x1.868D9Ap+0f,
	0x1.8ACE54p+0f, 0x1.8F1AEAp+0f, 0x1.93737Cp+0f, 0x1.97D82Ap+0f,
	0x1.9C4918p+0f, 0x1.A0C668p+0f, 0x1.A5503Cp+0f, 0x1.A9E6B6p+0f,
	0x1.AE89FAp+0f, 0x1.B33A2Cp+0f, 0x1.B7F770p+0f, 0x1.BCC1EAp+0f,
	0x1.C199BEp+0f, 0x1.C67F12p+0f, 0x1.CB720Ep+0f, 0x1.D072D4p+0f,
	0x1.D5818Ep+0f, 0x1.DA9E60p+0f, 0x1.DFC974p+0f, 0x1.E502EEp+0f,
	0x1.EA4AFAp+0f, 0x1.EFA1BEp+0f, 0x1.F50766p+0f, 0x1.FA7C18p+0f,
};

#define meformer_exp_init \
	const float32x4_t vlog2e = vmovq_n_f32(1.44269502); \
	const float32x4_t vmagic_bias = vmovq_n_f32(196608); \
	const int32x4_t vindex_mask = vmovq_n_s32(63); \
	const float32x4_t vminus_ln2 = vmovq_n_f32(-0.693147182); \
	const float32x4_t vc2 = vmovq_n_f32(0.499996334); \
	const float32x4_t vdenorm_cutoff = vmovq_n_f32(-87.3365402);

// int batch, float *input, float32x4_t vi_max, float *output, float sum, int stride
#define meformer_minusmax_exp_sum(batch, input, vi_max, output, sum, stride) \
	float32x4_t vacc0 = vmovq_n_f32(0.0f); \
	\
	for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) { \
		const float32x4_t vi0123 = vld1q_f32(input); input += stride; \
		const float32x4_t vi4567 = vld1q_f32(input); input += stride; \
		const float32x4_t vi89AB = vld1q_f32(input); input += stride; \
		const float32x4_t viCDEF = vld1q_f32(input); input += stride; \
		\
		const float32x4_t vx0123 = vsubq_f32(vi0123, vi_max); \
		const float32x4_t vx4567 = vsubq_f32(vi4567, vi_max); \
		const float32x4_t vx89AB = vsubq_f32(vi89AB, vi_max); \
		const float32x4_t vxCDEF = vsubq_f32(viCDEF, vi_max); \
		\
		float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vx0123, vlog2e); \
		float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vx4567, vlog2e); \
		float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vx89AB, vlog2e); \
		float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vxCDEF, vlog2e); \
		\
		const int32x4_t ve0123 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn0123), vmovq_n_s32(INT32_C(0x3F))), 17); \
		const int32x4_t ve4567 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn4567), vmovq_n_s32(INT32_C(0x3F))), 17); \
		const int32x4_t ve89AB = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn89AB), vmovq_n_s32(INT32_C(0x3F))), 17); \
		const int32x4_t veCDEF = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vnCDEF), vmovq_n_s32(INT32_C(0x3F))), 17); \
		\
		const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask)); \
		const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0); \
		const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1); \
		const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask)); \
		const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0); \
		const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1); \
		const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask)); \
		const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0); \
		const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1); \
		const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask)); \
		const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0); \
		const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1); \
		\
		float32x2_t vl01 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx01]); \
		float32x2_t vl23 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx23]); \
		float32x2_t vl45 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx45]); \
		float32x2_t vl67 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx67]); \
		float32x2_t vl89 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx89]); \
		float32x2_t vlAB = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidxAB]); \
		float32x2_t vlCD = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidxCD]); \
		float32x2_t vlEF = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidxEF]); \
		\
		vl01 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1); \
		vl23 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1); \
		const float32x4_t vl0123 = vcombine_f32(vl01, vl23); \
		vl45 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1); \
		vl67 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1); \
		const float32x4_t vl4567 = vcombine_f32(vl45, vl67); \
		vl89 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx89 >> 32)], vl89, 1); \
		vlAB = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidxAB >> 32)], vlAB, 1); \
		const float32x4_t vl89AB = vcombine_f32(vl89, vlAB); \
		vlCD = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidxCD >> 32)], vlCD, 1); \
		vlEF = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidxEF >> 32)], vlEF, 1); \
		const float32x4_t vlCDEF = vcombine_f32(vlCD, vlEF); \
		\
		const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123)); \
		const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567)); \
		const float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl89AB), ve89AB)); \
		const float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlCDEF), veCDEF)); \
		\
		vn0123 = vsubq_f32(vn0123, vmagic_bias); \
		vn4567 = vsubq_f32(vn4567, vmagic_bias); \
		vn89AB = vsubq_f32(vn89AB, vmagic_bias); \
		vnCDEF = vsubq_f32(vnCDEF, vmagic_bias); \
		\
		float32x4_t vt0123 = vfmaq_f32(vx0123, vn0123, vminus_ln2); \
		float32x4_t vt4567 = vfmaq_f32(vx4567, vn4567, vminus_ln2); \
		float32x4_t vt89AB = vfmaq_f32(vx89AB, vn89AB, vminus_ln2); \
		float32x4_t vtCDEF = vfmaq_f32(vxCDEF, vnCDEF, vminus_ln2); \
		\
		float32x4_t vp0123 = vmulq_f32(vt0123, vc2); \
		float32x4_t vp4567 = vmulq_f32(vt4567, vc2); \
		float32x4_t vp89AB = vmulq_f32(vt89AB, vc2); \
		float32x4_t vpCDEF = vmulq_f32(vtCDEF, vc2); \
		\
		vp0123 = vfmaq_f32(vt0123, vt0123, vp0123); \
		vp4567 = vfmaq_f32(vt4567, vt4567, vp4567); \
		vp89AB = vfmaq_f32(vt89AB, vt89AB, vp89AB); \
		vpCDEF = vfmaq_f32(vtCDEF, vtCDEF, vpCDEF); \
		\
		float32x4_t vf0123 = vfmaq_f32(vs0123, vs0123, vp0123); \
		float32x4_t vf4567 = vfmaq_f32(vs4567, vs4567, vp4567); \
		float32x4_t vf89AB = vfmaq_f32(vs89AB, vs89AB, vp89AB); \
		float32x4_t vfCDEF = vfmaq_f32(vsCDEF, vsCDEF, vpCDEF); \
		\
		vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vx0123, vdenorm_cutoff))); \
		vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vx4567, vdenorm_cutoff))); \
		vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcltq_f32(vx89AB, vdenorm_cutoff))); \
		vfCDEF = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfCDEF), vcltq_f32(vxCDEF, vdenorm_cutoff))); \
		\
		vst1q_f32(output, vf0123); output += stride; \
		vst1q_f32(output, vf4567); output += stride; \
		vst1q_f32(output, vf89AB); output += stride; \
		vst1q_f32(output, vfCDEF); output += stride; \
		\
		vacc0 = vaddq_f32(vacc0, vf0123); \
		vacc0 = vaddq_f32(vacc0, vf4567); \
		vacc0 = vaddq_f32(vacc0, vf89AB); \
		vacc0 = vaddq_f32(vacc0, vfCDEF); \
	} \
	\
	float32x4_t vacc = vacc0; \
	for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) { \
		const float32x4_t vi = vld1q_f32(input); input += stride; \
		\
		const float32x4_t vx = vsubq_f32(vi, vi_max); \
		\
		float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e); \
		\
		const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17); \
		\
		const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask)); \
		const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0); \
		const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1); \
		float32x2_t vl_lo = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_lo]); \
		float32x2_t vl_hi = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_hi]); \
		vl_lo = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1); \
		vl_hi = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1); \
		const float32x4_t vl = vcombine_f32(vl_lo, vl_hi); \
		const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve)); \
		\
		vn = vsubq_f32(vn, vmagic_bias); \
		\
		float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2); \
		\
		float32x4_t vp = vmulq_f32(vt, vc2); \
		vp = vfmaq_f32(vt, vt, vp); \
		\
		float32x4_t vf = vfmaq_f32(vs, vs, vp); \
		\
		vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff))); \
		\
		vst1q_f32(output, vf); output += stride; \
		\
		vacc = vaddq_f32(vacc, vf); \
	} \
	\
	float vacc_lo = vaddvq_f32(vacc); \
	\
	if (batch != 0) { \
		assert(batch >= 1 * sizeof(float)); \
		assert(batch <= 3 * sizeof(float)); \
		const float32x4_t vi = vld1q_f32(input); \
		\
		const float32x4_t vx = vsubq_f32(vi, vi_max); \
		\
		float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e); \
		\
		const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17); \
		\
		const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask)); \
		const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0); \
		const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1); \
		float32x2_t vl_lo = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_lo]); \
		float32x2_t vl_hi = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_hi]); \
		vl_lo = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1); \
		vl_hi = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1); \
		const float32x4_t vl = vcombine_f32(vl_lo, vl_hi); \
		const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve)); \
		\
		vn = vsubq_f32(vn, vmagic_bias); \
		\
		float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2); \
		\
		float32x4_t vp = vmulq_f32(vt, vc2); \
		vp = vfmaq_f32(vt, vt, vp); \
		\
		float32x4_t vf = vfmaq_f32(vs, vs, vp); \
		\
		vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff))); \
		\
		float32x2_t vf_lo = vget_low_f32(vf); \
		\
		if (batch & (2 * sizeof(float))) { \
			vst1_f32(output, vf_lo); output += 2; \
			\
			vacc_lo += vaddv_f32(vf_lo); \
			\
			vf_lo = vget_high_f32(vf); \
		} \
		if (batch & (1 * sizeof(float))) { \
			vst1_lane_f32(output, vf_lo, 0); \
			\
			vacc_lo += vget_lane_f32(vf_lo, 0); \
		} \
	} \
	\
	*sum += vacc_lo;

// int batch, float *input, float *output, int stride
#define meformer_exp(batch, input, output, stride) \
	for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) { \
		const float32x4_t vi0123 = vld1q_f32(input); input += stride; \
		const float32x4_t vi4567 = vld1q_f32(input); input += stride; \
		const float32x4_t vi89AB = vld1q_f32(input); input += stride; \
		const float32x4_t viCDEF = vld1q_f32(input); input += stride; \
		\
		float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vi0123, vlog2e); \
		float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vi4567, vlog2e); \
		float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vi89AB, vlog2e); \
		float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, viCDEF, vlog2e); \
		\
		const int32x4_t ve0123 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn0123), vmovq_n_s32(INT32_C(0x3F))), 17); \
		const int32x4_t ve4567 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn4567), vmovq_n_s32(INT32_C(0x3F))), 17); \
		const int32x4_t ve89AB = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn89AB), vmovq_n_s32(INT32_C(0x3F))), 17); \
		const int32x4_t veCDEF = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vnCDEF), vmovq_n_s32(INT32_C(0x3F))), 17); \
		\
		const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask)); \
		const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0); \
		const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1); \
		const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask)); \
		const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0); \
		const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1); \
		const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask)); \
		const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0); \
		const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1); \
		const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask)); \
		const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0); \
		const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1); \
		\
		float32x2_t vl01 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx01]); \
		float32x2_t vl23 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx23]); \
		float32x2_t vl45 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx45]); \
		float32x2_t vl67 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx67]); \
		float32x2_t vl89 = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx89]); \
		float32x2_t vlAB = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidxAB]); \
		float32x2_t vlCD = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidxCD]); \
		float32x2_t vlEF = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidxEF]); \
		\
		vl01 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1); \
		vl23 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1); \
		const float32x4_t vl0123 = vcombine_f32(vl01, vl23); \
		vl45 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1); \
		vl67 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1); \
		const float32x4_t vl4567 = vcombine_f32(vl45, vl67); \
		vl89 = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx89 >> 32)], vl89, 1); \
		vlAB = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidxAB >> 32)], vlAB, 1); \
		const float32x4_t vl89AB = vcombine_f32(vl89, vlAB); \
		vlCD = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidxCD >> 32)], vlCD, 1); \
		vlEF = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidxEF >> 32)], vlEF, 1); \
		const float32x4_t vlCDEF = vcombine_f32(vlCD, vlEF); \
		\
		const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123)); \
		const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567)); \
		const float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl89AB), ve89AB)); \
		const float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlCDEF), veCDEF)); \
		\
		vn0123 = vsubq_f32(vn0123, vmagic_bias); \
		vn4567 = vsubq_f32(vn4567, vmagic_bias); \
		vn89AB = vsubq_f32(vn89AB, vmagic_bias); \
		vnCDEF = vsubq_f32(vnCDEF, vmagic_bias); \
		\
		float32x4_t vt0123 = vfmaq_f32(vi0123, vn0123, vminus_ln2); \
		float32x4_t vt4567 = vfmaq_f32(vi4567, vn4567, vminus_ln2); \
		float32x4_t vt89AB = vfmaq_f32(vi89AB, vn89AB, vminus_ln2); \
		float32x4_t vtCDEF = vfmaq_f32(viCDEF, vnCDEF, vminus_ln2); \
		\
		float32x4_t vp0123 = vmulq_f32(vt0123, vc2); \
		float32x4_t vp4567 = vmulq_f32(vt4567, vc2); \
		float32x4_t vp89AB = vmulq_f32(vt89AB, vc2); \
		float32x4_t vpCDEF = vmulq_f32(vtCDEF, vc2); \
		\
		vp0123 = vfmaq_f32(vt0123, vt0123, vp0123); \
		vp4567 = vfmaq_f32(vt4567, vt4567, vp4567); \
		vp89AB = vfmaq_f32(vt89AB, vt89AB, vp89AB); \
		vpCDEF = vfmaq_f32(vtCDEF, vtCDEF, vpCDEF); \
		\
		float32x4_t vf0123 = vfmaq_f32(vs0123, vs0123, vp0123); \
		float32x4_t vf4567 = vfmaq_f32(vs4567, vs4567, vp4567); \
		float32x4_t vf89AB = vfmaq_f32(vs89AB, vs89AB, vp89AB); \
		float32x4_t vfCDEF = vfmaq_f32(vsCDEF, vsCDEF, vpCDEF); \
		\
		vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vi0123, vdenorm_cutoff))); \
		vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vi4567, vdenorm_cutoff))); \
		vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcltq_f32(vi89AB, vdenorm_cutoff))); \
		vfCDEF = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfCDEF), vcltq_f32(viCDEF, vdenorm_cutoff))); \
		\
		vst1q_f32(output, vf0123); output += stride; \
		vst1q_f32(output, vf4567); output += stride; \
		vst1q_f32(output, vf89AB); output += stride; \
		vst1q_f32(output, vfCDEF); output += stride; \
	} \
	\
	for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) { \
		const float32x4_t vi = vld1q_f32(input); input += stride; \
		\
		float32x4_t vn = vfmaq_f32(vmagic_bias, vi, vlog2e); \
		\
		const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17); \
		\
		const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask)); \
		const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0); \
		const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1); \
		float32x2_t vl_lo = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_lo]); \
		float32x2_t vl_hi = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_hi]); \
		vl_lo = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1); \
		vl_hi = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1); \
		const float32x4_t vl = vcombine_f32(vl_lo, vl_hi); \
		const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve)); \
		\
		vn = vsubq_f32(vn, vmagic_bias); \
		\
		float32x4_t vt = vfmaq_f32(vi, vn, vminus_ln2); \
		\
		float32x4_t vp = vmulq_f32(vt, vc2); \
		vp = vfmaq_f32(vt, vt, vp); \
		\
		float32x4_t vf = vfmaq_f32(vs, vs, vp); \
		\
		vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vi, vdenorm_cutoff))); \
		\
		vst1q_f32(output, vf); output += stride; \
	} \
	\
	if (batch != 0) { \
		assert(batch >= 1 * sizeof(float)); \
		assert(batch <= 3 * sizeof(float)); \
		const float32x4_t vi = vld1q_f32(input); \
		\
		float32x4_t vn = vfmaq_f32(vmagic_bias, vi, vlog2e); \
		\
		const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17); \
		\
		const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask)); \
		const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0); \
		const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1); \
		float32x2_t vl_lo = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_lo]); \
		float32x2_t vl_hi = vld1_dup_f32(&meformer_table_exp2_k_over_64[(uint32_t) vidx_hi]); \
		vl_lo = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1); \
		vl_hi = vld1_lane_f32(&meformer_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1); \
		const float32x4_t vl = vcombine_f32(vl_lo, vl_hi); \
		const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve)); \
		\
		vn = vsubq_f32(vn, vmagic_bias); \
		\
		float32x4_t vt = vfmaq_f32(vi, vn, vminus_ln2); \
		\
		float32x4_t vp = vmulq_f32(vt, vc2); \
		vp = vfmaq_f32(vt, vt, vp); \
		\
		float32x4_t vf = vfmaq_f32(vs, vs, vp); \
		\
		vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vi, vdenorm_cutoff))); \
		\
		float32x2_t vf_lo = vget_low_f32(vf); \
		\
		if (batch & (2 * sizeof(float))) { \
			vst1_f32(output, vf_lo); output += 2; \
			\
			vf_lo = vget_high_f32(vf); \
		} \
		if (batch & (1 * sizeof(float))) { \
			vst1_lane_f32(output, vf_lo, 0); \
		} \
	}

#define gemm_ukernel_5x16_u8_first_k \
	"	ldr q8, [x20]					\n" \
	"	ldr q0, [x15], #16				\n" \
	"	ldr q1, [x15], #16				\n" \
	"	ldr q2, [x15], #16				\n" \
	\
	"	fmul v12.4s, v8.4s, v0.s[0]		\n" \
	"	ldr q3, [x15], #16				\n" \
	"	fmul v13.4s, v8.4s, v1.s[0]		\n" \
	"	ldr q4, [x15], #16				\n" \
	"	ldr q9, [x20, #16]				\n" \
	"	fmul v14.4s, v8.4s, v2.s[0]		\n" \
	"	fmul v15.4s, v8.4s, v3.s[0]		\n" \
	"	fmul v16.4s, v8.4s, v4.s[0]		\n" \
	"	ldr q10, [x20, #32]				\n" \
	"	ldr q11, [x20, #48]				\n" \
	"	add x20, x20, x1      			\n" \
	\
	"	fmul v17.4s, v9.4s, v0.s[0]		\n" \
	"	fmul v18.4s, v9.4s, v1.s[0]		\n" \
	"	fmul v19.4s, v9.4s, v2.s[0]		\n" \
	"	fmul v20.4s, v9.4s, v3.s[0]		\n" \
	"	fmul v21.4s, v9.4s, v4.s[0]		\n" \
	"	ldr q8, [x20]					\n" \
	"	ldr q9, [x20, #16]				\n" \
	\
	"	fmul v22.4s, v10.4s, v0.s[0]	\n" \
	"	fmul v23.4s, v10.4s, v1.s[0]	\n" \
	"	fmul v24.4s, v10.4s, v2.s[0]	\n" \
	"	fmul v25.4s, v10.4s, v3.s[0]	\n" \
	"	fmul v26.4s, v10.4s, v4.s[0]	\n" \
	"	ldr q10, [x20, #32]				\n" \
	"	ldr q5, [x15], #16				\n" \
	\
	"	fmul v27.4s, v11.4s, v0.s[0]	\n" \
	"	fmul v28.4s, v11.4s, v1.s[0]	\n" \
	"	fmul v29.4s, v11.4s, v2.s[0]	\n" \
	"	fmul v30.4s, v11.4s, v3.s[0]	\n" \
	"	fmul v31.4s, v11.4s, v4.s[0]	\n" \
	"	ldr q11, [x20, #48]				\n" \
	"	add x20, x20, x1      			\n"

#define gemm_ukernel_5x16_u8_k0 \
	"	fmla v12.4s, v8.4s, v0.s[0]		\n" \
	"	fmla v13.4s, v8.4s, v1.s[0]		\n" \
	"	fmla v14.4s, v8.4s, v2.s[0]		\n" \
	"	fmla v15.4s, v8.4s, v3.s[0]		\n" \
	"	fmla v16.4s, v8.4s, v4.s[0]		\n" \
	"	ldr q8, [x20]					\n" \
	\
	"	fmla v17.4s, v9.4s, v0.s[0]		\n" \
	"	fmla v18.4s, v9.4s, v1.s[0]		\n" \
	"	fmla v19.4s, v9.4s, v2.s[0]		\n" \
	"	fmla v20.4s, v9.4s, v3.s[0]		\n" \
	"	fmla v21.4s, v9.4s, v4.s[0]		\n" \
	"	ldr q9, [x20, #16]				\n" \
	\
	"	fmla v22.4s, v10.4s, v0.s[0]	\n" \
	"	fmla v23.4s, v10.4s, v1.s[0]	\n" \
	"	fmla v24.4s, v10.4s, v2.s[0]	\n" \
	"	fmla v25.4s, v10.4s, v3.s[0]	\n" \
	"	fmla v26.4s, v10.4s, v4.s[0]	\n" \
	"	ldr q10, [x20, #32]				\n" \
	"	ldr q5, [x15], #16				\n" \
	\
	"	fmla v27.4s, v11.4s, v0.s[0]	\n" \
	"	fmla v28.4s, v11.4s, v1.s[0]	\n" \
	"	fmla v29.4s, v11.4s, v2.s[0]	\n" \
	"	fmla v30.4s, v11.4s, v3.s[0]	\n" \
	"	fmla v31.4s, v11.4s, v4.s[0]	\n" \
	"	ldr q11, [x20, #48]				\n" \
	"	add x20, x20, x1      			\n"

#define gemm_ukernel_5x16_u8_k1 \
	"	fmla v12.4s, v8.4s, v0.s[1]		\n" \
	"	fmla v13.4s, v8.4s, v1.s[1]		\n" \
	"	fmla v14.4s, v8.4s, v2.s[1]		\n" \
	"	fmla v15.4s, v8.4s, v3.s[1]		\n" \
	"	fmla v16.4s, v8.4s, v4.s[1]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v0.s[1]		\n" \
	"	fmla v18.4s, v9.4s, v1.s[1]		\n" \
	"	fmla v19.4s, v9.4s, v2.s[1]		\n" \
	"	fmla v20.4s, v9.4s, v3.s[1]		\n" \
	"	fmla v21.4s, v9.4s, v4.s[1]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	"	ldr q6, [x15], #16              \n" \
	\
	"	fmla v22.4s, v10.4s, v0.s[1]	\n" \
	"	fmla v23.4s, v10.4s, v1.s[1]	\n" \
	"	fmla v24.4s, v10.4s, v2.s[1]	\n" \
	"	fmla v25.4s, v10.4s, v3.s[1]	\n" \
	"	fmla v26.4s, v10.4s, v4.s[1]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v27.4s, v11.4s, v0.s[1]	\n" \
	"	fmla v28.4s, v11.4s, v1.s[1]	\n" \
	"	fmla v29.4s, v11.4s, v2.s[1]	\n" \
	"	fmla v30.4s, v11.4s, v3.s[1]	\n" \
	"	fmla v31.4s, v11.4s, v4.s[1]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_k2 \
	"	fmla v12.4s, v8.4s, v0.s[2]		\n" \
	"	fmla v13.4s, v8.4s, v1.s[2]		\n" \
	"	fmla v14.4s, v8.4s, v2.s[2]		\n" \
	"	fmla v15.4s, v8.4s, v3.s[2]		\n" \
	"	fmla v16.4s, v8.4s, v4.s[2]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v0.s[2]		\n" \
	"	fmla v18.4s, v9.4s, v1.s[2]		\n" \
	"	fmla v19.4s, v9.4s, v2.s[2]		\n" \
	"	fmla v20.4s, v9.4s, v3.s[2]		\n" \
	"	fmla v21.4s, v9.4s, v4.s[2]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	"	ldr q7, [x15], #16              \n" \
	\
	"	fmla v22.4s, v10.4s, v0.s[2]	\n" \
	"	fmla v23.4s, v10.4s, v1.s[2]	\n" \
	"	fmla v24.4s, v10.4s, v2.s[2]	\n" \
	"	fmla v25.4s, v10.4s, v3.s[2]	\n" \
	"	fmla v26.4s, v10.4s, v4.s[2]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v27.4s, v11.4s, v0.s[2]	\n" \
	"	fmla v28.4s, v11.4s, v1.s[2]	\n" \
	"	fmla v29.4s, v11.4s, v2.s[2]	\n" \
	"	fmla v30.4s, v11.4s, v3.s[2]	\n" \
	"	fmla v31.4s, v11.4s, v4.s[2]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_k3 \
	"	fmla v12.4s, v8.4s, v0.s[3]		\n" \
	"	fmla v13.4s, v8.4s, v1.s[3]		\n" \
	"	fmla v14.4s, v8.4s, v2.s[3]		\n" \
	"	fmla v15.4s, v8.4s, v3.s[3]		\n" \
	"	fmla v16.4s, v8.4s, v4.s[3]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v0.s[3]		\n" \
	"	fmla v18.4s, v9.4s, v1.s[3]		\n" \
	"	fmla v19.4s, v9.4s, v2.s[3]		\n" \
	"	fmla v20.4s, v9.4s, v3.s[3]		\n" \
	"	fmla v21.4s, v9.4s, v4.s[3]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	\
	"	fmla v22.4s, v10.4s, v0.s[3]	\n" \
	"	fmla v27.4s, v11.4s, v0.s[3]	\n" \
	"	ldr q0, [x15], #16				\n" \
	"	fmla v23.4s, v10.4s, v1.s[3]	\n" \
	"	fmla v28.4s, v11.4s, v1.s[3]	\n" \
	"	ldr q1, [x15], #16				\n" \
	"	mov x16, x15					\n" \
	"	fmla v24.4s, v10.4s, v2.s[3]	\n" \
	"	fmla v25.4s, v10.4s, v3.s[3]	\n" \
	"	fmla v26.4s, v10.4s, v4.s[3]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v29.4s, v11.4s, v2.s[3]	\n" \
	"	fmla v30.4s, v11.4s, v3.s[3]	\n" \
	"	fmla v31.4s, v11.4s, v4.s[3]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_k4 \
	"	fmla v12.4s, v8.4s, v5.s[0]		\n" \
	"	fmla v13.4s, v8.4s, v6.s[0]		\n" \
	"	fmla v14.4s, v8.4s, v7.s[0]		\n" \
	"	fmla v15.4s, v8.4s, v0.s[0]		\n" \
	"	fmla v16.4s, v8.4s, v1.s[0]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v5.s[0]		\n" \
	"	fmla v18.4s, v9.4s, v6.s[0]		\n" \
	"	fmla v19.4s, v9.4s, v7.s[0]		\n" \
	"	fmla v20.4s, v9.4s, v0.s[0]		\n" \
	"	fmla v21.4s, v9.4s, v1.s[0]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	"	ldr q2, [x16, #32]              \n" \
	\
	"	fmla v22.4s, v10.4s, v5.s[0]	\n" \
	"	fmla v23.4s, v10.4s, v6.s[0]	\n" \
	"	fmla v24.4s, v10.4s, v7.s[0]	\n" \
	"	fmla v25.4s, v10.4s, v0.s[0]	\n" \
	"	fmla v26.4s, v10.4s, v1.s[0]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v27.4s, v11.4s, v5.s[0]	\n" \
	"	fmla v28.4s, v11.4s, v6.s[0]	\n" \
	"	fmla v29.4s, v11.4s, v7.s[0]	\n" \
	"	fmla v30.4s, v11.4s, v0.s[0]	\n" \
	"	fmla v31.4s, v11.4s, v1.s[0]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_k5 \
	"	fmla v12.4s, v8.4s, v5.s[1]		\n" \
	"	fmla v13.4s, v8.4s, v6.s[1]		\n" \
	"	fmla v14.4s, v8.4s, v7.s[1]		\n" \
	"	fmla v15.4s, v8.4s, v0.s[1]		\n" \
	"	fmla v16.4s, v8.4s, v1.s[1]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v5.s[1]		\n" \
	"	fmla v18.4s, v9.4s, v6.s[1]		\n" \
	"	fmla v19.4s, v9.4s, v7.s[1]		\n" \
	"	fmla v20.4s, v9.4s, v0.s[1]		\n" \
	"	fmla v21.4s, v9.4s, v1.s[1]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	"	ldr q3, [x16, #48]              \n" \
	\
	"	fmla v22.4s, v10.4s, v5.s[1]	\n" \
	"	fmla v23.4s, v10.4s, v6.s[1]	\n" \
	"	fmla v24.4s, v10.4s, v7.s[1]	\n" \
	"	fmla v25.4s, v10.4s, v0.s[1]	\n" \
	"	fmla v26.4s, v10.4s, v1.s[1]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v27.4s, v11.4s, v5.s[1]	\n" \
	"	fmla v28.4s, v11.4s, v6.s[1]	\n" \
	"	fmla v29.4s, v11.4s, v7.s[1]	\n" \
	"	fmla v30.4s, v11.4s, v0.s[1]	\n" \
	"	fmla v31.4s, v11.4s, v1.s[1]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_k6 \
	"	fmla v12.4s, v8.4s, v5.s[2]		\n" \
	"	fmla v13.4s, v8.4s, v6.s[2]		\n" \
	"	fmla v14.4s, v8.4s, v7.s[2]		\n" \
	"	fmla v15.4s, v8.4s, v0.s[2]		\n" \
	"	fmla v16.4s, v8.4s, v1.s[2]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v5.s[2]		\n" \
	"	fmla v18.4s, v9.4s, v6.s[2]		\n" \
	"	fmla v19.4s, v9.4s, v7.s[2]		\n" \
	"	fmla v20.4s, v9.4s, v0.s[2]		\n" \
	"	fmla v21.4s, v9.4s, v1.s[2]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	"	ldr q4, [x16, #64]              \n" \
	\
	"	fmla v22.4s, v10.4s, v5.s[2]	\n" \
	"	fmla v23.4s, v10.4s, v6.s[2]	\n" \
	"	fmla v24.4s, v10.4s, v7.s[2]	\n" \
	"	fmla v25.4s, v10.4s, v0.s[2]	\n" \
	"	fmla v26.4s, v10.4s, v1.s[2]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v27.4s, v11.4s, v5.s[2]	\n" \
	"	fmla v28.4s, v11.4s, v6.s[2]	\n" \
	"	fmla v29.4s, v11.4s, v7.s[2]	\n" \
	"	fmla v30.4s, v11.4s, v0.s[2]	\n" \
	"	fmla v31.4s, v11.4s, v1.s[2]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_k7 \
	"	fmla v12.4s, v8.4s, v5.s[3]		\n" \
	"	fmla v13.4s, v8.4s, v6.s[3]		\n" \
	"	fmla v14.4s, v8.4s, v7.s[3]		\n" \
	"	fmla v15.4s, v8.4s, v0.s[3]		\n" \
	"	fmla v16.4s, v8.4s, v1.s[3]		\n" \
	"	ldr q8, [x20]                   \n" \
	\
	"	fmla v17.4s, v9.4s, v5.s[3]		\n" \
	"	fmla v18.4s, v9.4s, v6.s[3]		\n" \
	"	fmla v19.4s, v9.4s, v7.s[3]		\n" \
	"	fmla v20.4s, v9.4s, v0.s[3]		\n" \
	"	fmla v21.4s, v9.4s, v1.s[3]		\n" \
	"	ldr q9, [x20, #16]              \n" \
	\
	"	fmla v30.4s, v11.4s, v0.s[3]	\n" \
	"	fmla v25.4s, v10.4s, v0.s[3]	\n" \
	"	ldr q0, [x16]				    \n" \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n" \
	"	fmla v26.4s, v10.4s, v1.s[3]	\n" \
	"	ldr q1, [x16, #16]			    \n" \
	"	add x16, x16, #80               \n" \
	"	mov x15, x16                    \n" \
	"	fmla v22.4s, v10.4s, v5.s[3]	\n" \
	"	fmla v23.4s, v10.4s, v6.s[3]	\n" \
	"	fmla v24.4s, v10.4s, v7.s[3]	\n" \
	"	ldr q10, [x20, #32]             \n" \
	\
	"	fmla v27.4s, v11.4s, v5.s[3]	\n" \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n" \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n" \
	"	ldr q11, [x20, #48]             \n" \
	"	add x20, x20, x1  				\n"

#define gemm_ukernel_5x16_u8_last_k \
	"	prfm PLDL1STRM, [x24]           \n" \
	"	prfm PLDL1STRM, [x25]           \n" \
	"	prfm PLDL1STRM, [x26]           \n" \
	"	prfm PLDL1STRM, [x27]           \n" \
	"	prfm PLDL1STRM, [x28]           \n" \
	\
	"	fmla v12.4s, v8.4s, v5.s[3]		\n" \
	"	fmla v17.4s, v9.4s, v5.s[3]		\n" \
	"	str q12, [x24]                  \n" \
	"	str q17, [x24, #16]             \n" \
	"	fmla v22.4s, v10.4s, v5.s[3]	\n" \
	"	fmla v27.4s, v11.4s, v5.s[3]	\n" \
	"	str q22, [x24, #32]             \n" \
	"	str q27, [x24, #48]             \n" \
	\
	"	fmla v13.4s, v8.4s, v6.s[3]		\n" \
	"	fmla v18.4s, v9.4s, v6.s[3]		\n" \
	"	str q13, [x25]                  \n" \
	"	str q18, [x25, #16]             \n" \
	"	fmla v23.4s, v10.4s, v6.s[3]	\n" \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n" \
	"	str q23, [x25, #32]             \n" \
	"	str q28, [x25, #48]             \n" \
	\
	"	fmla v14.4s, v8.4s, v7.s[3]		\n" \
	"	fmla v19.4s, v9.4s, v7.s[3]		\n" \
	"	str q14, [x26]                  \n" \
	"	str q19, [x26, #16]             \n" \
	"	fmla v24.4s, v10.4s, v7.s[3]	\n" \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n" \
	"	str q24, [x26, #32]             \n" \
	"	str q29, [x26, #48]             \n" \
	\
	"	fmla v15.4s, v8.4s, v0.s[3]		\n" \
	"	fmla v20.4s, v9.4s, v0.s[3]		\n" \
	"	str q15, [x27]                  \n" \
	"	str q20, [x27, #16]             \n" \
	"	fmla v25.4s, v10.4s, v0.s[3]	\n" \
	"	fmla v30.4s, v11.4s, v0.s[3]	\n" \
	"	str q25, [x27, #32]             \n" \
	"	str q30, [x27, #48]             \n" \
	\
	"	fmla v16.4s, v8.4s, v1.s[3]		\n" \
	"	fmla v21.4s, v9.4s, v1.s[3]		\n" \
	"	str q16, [x28]                  \n" \
	"	str q21, [x28, #16]             \n" \
	"	fmla v26.4s, v10.4s, v1.s[3]	\n" \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n" \
	"	str q26, [x28, #32]             \n" \
	"	str q31, [x28, #48]             \n"

#define gemm_ukernel_5x16_u8_last_k_with_norm \
	"	prfm PLDL1STRM, [x24]           \n"  \
	"	prfm PLDL1STRM, [x25]           \n"  \
	"	prfm PLDL1STRM, [x26]           \n"  \
	"	prfm PLDL1STRM, [x27]           \n"  \
	"	prfm PLDL1STRM, [x28]           \n"  \
	"	fmov v2.4s, #1.0                \n"  \
	\
	/*sum: v3,v4,v12,v17,v22*/\
	"	ld1r {v3.4s}, [x4]              \n" \
	"   add x4, x4, #4                  \n" \
	"	fdiv v3.4s, v2.4s, v3.4s        \n" \
	"	ld1r {v4.4s}, [x4]              \n" \
	"   add x4, x4, #4                  \n" \
	"	fdiv v4.4s, v2.4s, v4.4s        \n" \
	\
	"	fmla v12.4s, v8.4s, v5.s[3]		\n" \
	"	fmla v17.4s, v9.4s, v5.s[3]		\n" \
	"	fmla v22.4s, v10.4s, v5.s[3]	\n" \
	"	fmla v27.4s, v11.4s, v5.s[3]	\n" \
	"	fmul v12.4s, v12.4s, v3.4s      \n" \
	"	fmul v17.4s, v17.4s, v3.4s      \n" \
	"	str q12, [x24]                  \n" \
	"	str q17, [x24, #16]             \n" \
	"	fmul v22.4s, v22.4s, v3.4s      \n" \
	"	fmul v27.4s, v27.4s, v3.4s      \n" \
	"	str q22, [x24, #32]             \n" \
	"	str q27, [x24, #48]             \n" \
	\
	"	ld1r {v12.4s}, [x4]             \n" \
	"   add x4, x4, #4                  \n" \
	"	fdiv v12.4s, v2.4s, v12.4s      \n" \
	"	ld1r {v17.4s}, [x4]             \n" \
	"   add x4, x4, #4                  \n" \
	"	fdiv v17.4s, v2.4s, v17.4s      \n" \
	"	ld1r {v22.4s}, [x4]             \n" \
	"	fdiv v22.4s, v2.4s, v22.4s      \n" \
	\
	"	fmla v13.4s, v8.4s, v6.s[3]		\n" \
	"	fmla v18.4s, v9.4s, v6.s[3]		\n" \
	"	fmla v23.4s, v10.4s, v6.s[3]	\n" \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n" \
	"	fmul v13.4s, v13.4s, v4.4s      \n" \
	"	fmul v18.4s, v18.4s, v4.4s      \n" \
	"	str q13, [x25]                  \n" \
	"	str q18, [x25, #16]             \n" \
	"	fmul v23.4s, v23.4s, v4.4s      \n" \
	"	fmul v28.4s, v28.4s, v4.4s      \n" \
	"	str q23, [x25, #32]             \n" \
	"	str q28, [x25, #48]             \n" \
	\
	"	fmla v14.4s, v8.4s, v7.s[3]		\n" \
	"	fmla v19.4s, v9.4s, v7.s[3]		\n" \
	"	fmla v24.4s, v10.4s, v7.s[3]	\n" \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n" \
	"	fmul v14.4s, v14.4s, v12.4s     \n" \
	"	fmul v19.4s, v19.4s, v12.4s     \n" \
	"	str q14, [x26]                  \n" \
	"	str q19, [x26, #16]             \n" \
	"	fmul v24.4s, v24.4s, v12.4s     \n" \
	"	fmul v29.4s, v29.4s, v12.4s     \n" \
	"	str q24, [x26, #32]             \n" \
	"	str q29, [x26, #48]             \n" \
	\
	"	fmla v15.4s, v8.4s, v0.s[3]		\n" \
	"	fmla v20.4s, v9.4s, v0.s[3]		\n" \
	"	fmla v25.4s, v10.4s, v0.s[3]	\n" \
	"	fmla v30.4s, v11.4s, v0.s[3]	\n" \
	"	fmul v15.4s, v15.4s, v17.4s     \n" \
	"	fmul v20.4s, v20.4s, v17.4s     \n" \
	"	str q15, [x27]                  \n" \
	"	str q20, [x27, #16]             \n" \
	"	fmul v25.4s, v25.4s, v17.4s     \n" \
	"	fmul v30.4s, v30.4s, v17.4s     \n" \
	"	str q25, [x27, #32]             \n" \
	"	str q30, [x27, #48]             \n" \
	\
	"	fmla v16.4s, v8.4s, v1.s[3]		\n" \
	"	fmla v21.4s, v9.4s, v1.s[3]		\n" \
	"	fmla v26.4s, v10.4s, v1.s[3]	\n" \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n" \
	"	fmul v16.4s, v16.4s, v22.4s     \n" \
	"	fmul v21.4s, v21.4s, v22.4s     \n" \
	"	str q16, [x28]                  \n" \
	"	str q21, [x28, #16]             \n" \
	"	fmul v26.4s, v26.4s, v22.4s     \n" \
	"	fmul v31.4s, v31.4s, v22.4s     \n" \
	"	str q26, [x28, #32]             \n" \
	"	str q31, [x28, #48]             \n"

#define gemm_ukernel_5x16_u8_last_k_with_ldr \
	"	prfm PLDL1STRM, [x24]           \n"  \
	"	prfm PLDL1STRM, [x25]           \n"  \
	"	prfm PLDL1STRM, [x26]           \n"  \
	"	prfm PLDL1STRM, [x27]           \n"  \
	"	prfm PLDL1STRM, [x28]           \n"  \
	\
	"	fmla v12.4s, v8.4s, v5.s[3]		\n"/*row 1*/\
	"	fmla v17.4s, v9.4s, v5.s[3]		\n"  \
	"	ldr q4, [x24]                   \n"  \
	"	fadd v12.4s, v12.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #16]              \n"  \
	"	str q12, [x24]                  \n"  \
	"   ldr q12, [x25]                  \n"  \
	"	fadd v17.4s, v17.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #32]              \n"  \
	"	str q17, [x24, #16]             \n"  \
	"   ldr q17, [x25, #16]             \n"  \
	\
	"	fmla v22.4s, v10.4s, v5.s[3]	\n"  \
	"	fmla v27.4s, v11.4s, v5.s[3]	\n"  \
	"   fadd v22.4s, v22.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #48]              \n"  \
	"	str q22, [x24, #32]             \n"  \
	"   ldr q22, [x25, #32]             \n"  \
	"   fadd v27.4s, v27.4s, v4.4s      \n"  \
	"	str q27, [x24, #48]             \n"  \
	"   ldr q27, [x25, #48]             \n"  \
	\
	\
	"	fmla v13.4s, v8.4s, v6.s[3]		\n"/*row 2*/\
	"	fmla v18.4s, v9.4s, v6.s[3]		\n"  \
	"   fadd v13.4s, v13.4s, v12.4s     \n"  \
	"   ldr q12, [x26]                  \n"  \
	"	str q13, [x25]                  \n"  \
	"   fadd v18.4s, v18.4s, v17.4s     \n"  \
	"   ldr q17, [x26, #16]             \n"  \
	"	str q18, [x25, #16]             \n"  \
	\
	"	fmla v23.4s, v10.4s, v6.s[3]	\n"  \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n"  \
	"   fadd v23.4s, v23.4s, v22.4s     \n"  \
	"   ldr q22, [x26, #32]             \n"  \
	"	str q23, [x25, #32]             \n"  \
	"   fadd v28.4s, v28.4s, v27.4s     \n"  \
	"   ldr q27, [x26, #48]             \n"  \
	"	str q28, [x25, #48]             \n"  \
	\
	\
	"	fmla v14.4s, v8.4s, v7.s[3]		\n"/*row 3*/\
	"	fmla v19.4s, v9.4s, v7.s[3]		\n"  \
	"   fadd v14.4s, v14.4s, v12.4s     \n"  \
	"   ldr q12, [x27]                  \n"  \
	"	str q14, [x26]                  \n"  \
	"   fadd v19.4s, v19.4s, v17.4s     \n"  \
	"   ldr q17, [x27, #16]             \n"  \
	"	str q19, [x26, #16]             \n"  \
	\
	"	fmla v24.4s, v10.4s, v7.s[3]	\n"  \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n"  \
	"   fadd v24.4s, v24.4s, v22.4s     \n"  \
	"   ldr q22, [x27, #32]             \n"  \
	"	str q24, [x26, #32]             \n"  \
	"   fadd v29.4s, v29.4s, v27.4s     \n"  \
	"   ldr q27, [x27, #48]             \n"  \
	"	str q29, [x26, #48]             \n"  \
	\
	\
	"	fmla v15.4s, v8.4s, v0.s[3]		\n"/*row 4*/\
	"	fmla v20.4s, v9.4s, v0.s[3]		\n"  \
	"   fadd v15.4s, v15.4s, v12.4s     \n"  \
	"   ldr q12, [x28]                  \n"  \
	"	str q15, [x27]                  \n"  \
	"   fadd v20.4s, v20.4s, v17.4s     \n"  \
	"   ldr q17, [x28, #16]             \n"  \
	"	str q20, [x27, #16]             \n"  \
	\
	"	fmla v25.4s, v10.4s, v0.s[3]	\n"  \
	"	fmla v30.4s, v11.4s, v0.s[3]	\n"  \
	"   fadd v25.4s, v25.4s, v22.4s     \n"  \
	"   ldr q22, [x28, #32]             \n"  \
	"	str q25, [x27, #32]             \n"  \
	"   fadd v30.4s, v30.4s, v27.4s     \n"  \
	"   ldr q27, [x28, #48]             \n"  \
	"	str q30, [x27, #48]             \n"  \
	\
	\
	"	fmla v16.4s, v8.4s, v1.s[3]		\n"/*row 5*/\
	"	fmla v21.4s, v9.4s, v1.s[3]		\n"  \
	"   fadd v16.4s, v16.4s, v12.4s     \n"  \
	"	str q16, [x28]                  \n"  \
	"   fadd v21.4s, v21.4s, v17.4s     \n"  \
	"	str q21, [x28, #16]             \n"  \
	\
	"	fmla v26.4s, v10.4s, v1.s[3]	\n"  \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n"  \
	"   fadd v26.4s, v26.4s, v22.4s     \n"  \
	"	str q26, [x28, #32]             \n"  \
	"   fadd v31.4s, v31.4s, v27.4s     \n"  \
	"	str q31, [x28, #48]             \n"

#define gemm_ukernel_5x16_u8_last_k_with_ldr_norm \
	"	prfm PLDL1STRM, [x24]           \n"  \
	"	prfm PLDL1STRM, [x25]           \n"  \
	"	prfm PLDL1STRM, [x26]           \n"  \
	"	prfm PLDL1STRM, [x27]           \n"  \
	"	prfm PLDL1STRM, [x28]           \n"  \
	"	prfm PLDL1STRM, [x4]            \n"  \
	\
	"	fmov v2.4s, #1.0                \n"  \
	\
	"	ldr q4, [x24]                   \n"  \
	"	fmla v12.4s, v8.4s, v5.s[3]		\n"/*row 1*/\
	"	fmla v17.4s, v9.4s, v5.s[3]		\n"  \
	"	fmla v22.4s, v10.4s, v5.s[3]	\n"  \
	"	fmla v27.4s, v11.4s, v5.s[3]	\n"  \
	"	ld1r {v5.4s}, [x4]              \n"  \
	"	add x4, x4, #4                  \n"  \
	"	fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"	fadd v12.4s, v12.4s, v4.4s      \n"  \
	"	fmul v12.4s, v12.4s, v5.4s      \n"  \
	"	ldr q4, [x24, #16]              \n"  \
	"	str q12, [x24]                  \n"  \
	"   ldr q12, [x25]                  \n"  \
	"	fadd v17.4s, v17.4s, v4.4s      \n"  \
	"	fmul v17.4s, v17.4s, v5.4s      \n"  \
	"	ldr q4, [x24, #32]              \n"  \
	"	str q17, [x24, #16]             \n"  \
	"   ldr q17, [x25, #16]             \n"  \
	\
	"   fadd v22.4s, v22.4s, v4.4s      \n"  \
	"	fmul v22.4s, v22.4s, v5.4s      \n"  \
	"	ldr q4, [x24, #48]              \n"  \
	"	str q22, [x24, #32]             \n"  \
	"   ldr q22, [x25, #32]             \n"  \
	"   fadd v27.4s, v27.4s, v4.4s      \n"  \
	"	fmul v27.4s, v27.4s, v5.4s      \n"  \
	"	ld1r {v5.4s}, [x4]              \n"  \
	"	add x4, x4, #4                  \n"  \
	"	fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"	str q27, [x24, #48]             \n"  \
	"   ldr q27, [x25, #48]             \n"  \
	\
	\
	"	fmla v13.4s, v8.4s, v6.s[3]		\n"/*row 2*/\
	"	fmla v18.4s, v9.4s, v6.s[3]		\n"  \
	"   fadd v13.4s, v13.4s, v12.4s     \n"  \
	"	fmul v13.4s, v13.4s, v5.4s      \n"  \
	"   ldr q12, [x26]                  \n"  \
	"	str q13, [x25]                  \n"  \
	"   fadd v18.4s, v18.4s, v17.4s     \n"  \
	"	fmul v18.4s, v18.4s, v5.4s      \n"  \
	"   ldr q17, [x26, #16]             \n"  \
	"	str q18, [x25, #16]             \n"  \
	\
	"	fmla v23.4s, v10.4s, v6.s[3]	\n"  \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n"  \
	"   fadd v23.4s, v23.4s, v22.4s     \n"  \
	"	fmul v23.4s, v23.4s, v5.4s      \n"  \
	"   ldr q22, [x26, #32]             \n"  \
	"	str q23, [x25, #32]             \n"  \
	"   fadd v28.4s, v28.4s, v27.4s     \n"  \
	"	fmul v28.4s, v28.4s, v5.4s      \n"  \
	"	ld1r {v5.4s}, [x4]              \n"  \
	"	add x4, x4, #4                  \n"  \
	"	fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"   ldr q27, [x26, #48]             \n"  \
	"	str q28, [x25, #48]             \n"  \
	\
	\
	"	fmla v14.4s, v8.4s, v7.s[3]		\n"/*row 3*/\
	"	fmla v19.4s, v9.4s, v7.s[3]		\n"  \
	"   fadd v14.4s, v14.4s, v12.4s     \n"  \
	"	fmul v14.4s, v14.4s, v5.4s      \n"  \
	"   ldr q12, [x27]                  \n"  \
	"	str q14, [x26]                  \n"  \
	"   fadd v19.4s, v19.4s, v17.4s     \n"  \
	"	fmul v19.4s, v19.4s, v5.4s      \n"  \
	"   ldr q17, [x27, #16]             \n"  \
	"	str q19, [x26, #16]             \n"  \
	\
	"	fmla v24.4s, v10.4s, v7.s[3]	\n"  \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n"  \
	"   fadd v24.4s, v24.4s, v22.4s     \n"  \
	"	fmul v24.4s, v24.4s, v5.4s      \n"  \
	"   ldr q22, [x27, #32]             \n"  \
	"	str q24, [x26, #32]             \n"  \
	"   fadd v29.4s, v29.4s, v27.4s     \n"  \
	"	fmul v29.4s, v29.4s, v5.4s      \n"  \
	"	ld1r {v5.4s}, [x4]              \n"  \
	"	add x4, x4, #4                  \n"  \
	"	fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"   ldr q27, [x27, #48]             \n"  \
	"	str q29, [x26, #48]             \n"  \
	\
	\
	"	fmla v15.4s, v8.4s, v0.s[3]		\n"/*row 4*/\
	"	fmla v20.4s, v9.4s, v0.s[3]		\n"  \
	"   fadd v15.4s, v15.4s, v12.4s     \n"  \
	"	fmul v15.4s, v15.4s, v5.4s      \n"  \
	"   ldr q12, [x28]                  \n"  \
	"	str q15, [x27]                  \n"  \
	"   fadd v20.4s, v20.4s, v17.4s     \n"  \
	"	fmul v20.4s, v20.4s, v5.4s      \n"  \
	"   ldr q17, [x28, #16]             \n"  \
	"	str q20, [x27, #16]             \n"  \
	\
	"	fmla v25.4s, v10.4s, v0.s[3]	\n"  \
	"	fmla v30.4s, v11.4s, v0.s[3]	\n"  \
	"   fadd v25.4s, v25.4s, v22.4s     \n"  \
	"	fmul v25.4s, v25.4s, v5.4s      \n"  \
	"   ldr q22, [x28, #32]             \n"  \
	"	str q25, [x27, #32]             \n"  \
	"   fadd v30.4s, v30.4s, v27.4s     \n"  \
	"	fmul v30.4s, v30.4s, v5.4s      \n"  \
	"	ld1r {v5.4s}, [x4]              \n"  \
	"	fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"   ldr q27, [x28, #48]             \n"  \
	"	str q30, [x27, #48]             \n"  \
	\
	\
	"	fmla v16.4s, v8.4s, v1.s[3]		\n"/*row 5*/\
	"	fmla v21.4s, v9.4s, v1.s[3]		\n"  \
	"   fadd v16.4s, v16.4s, v12.4s     \n"  \
	"	fmul v16.4s, v16.4s, v5.4s      \n"  \
	"	str q16, [x28]                  \n"  \
	"   fadd v21.4s, v21.4s, v17.4s     \n"  \
	"	fmul v21.4s, v21.4s, v5.4s      \n"  \
	"	str q21, [x28, #16]             \n"  \
	\
	"	fmla v26.4s, v10.4s, v1.s[3]	\n"  \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n"  \
	"   fadd v26.4s, v26.4s, v22.4s     \n"  \
	"	fmul v26.4s, v26.4s, v5.4s      \n"  \
	"	str q26, [x28, #32]             \n"  \
	"   fadd v31.4s, v31.4s, v27.4s     \n"  \
	"	fmul v31.4s, v31.4s, v5.4s      \n"  \
	"	str q31, [x28, #48]             \n"

#define gemm_ukernel_5x16_u8_last_k_with_ldr_update \
	"	prfm PLDL1STRM, [x24]           \n"  \
	"	prfm PLDL1STRM, [x25]           \n"  \
	"	prfm PLDL1STRM, [x26]           \n"  \
	"	prfm PLDL1STRM, [x27]           \n"  \
	"	prfm PLDL1STRM, [x28]           \n"  \
	"	prfm PLDL1STRM, [x3]            \n"  \
	\
	/*v3: exp(m1-m2) v4: ldr*/\
	"   ld1r {v3.4s}, [x3]              \n"  \
	"   add x3, x3, #4                  \n"  \
	"	ldr q4, [x24]                   \n"  \
	"	fmla v12.4s, v8.4s, v5.s[3]		\n"/*row 1*/\
	"	fmla v17.4s, v9.4s, v5.s[3]		\n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"	fadd v12.4s, v12.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #16]              \n"  \
	"	str q12, [x24]                  \n"  \
	"   ldr q12, [x25]                  \n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"	fadd v17.4s, v17.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #32]              \n"  \
	"	str q17, [x24, #16]             \n"  \
	"   ldr q17, [x25, #16]             \n"  \
	\
	"	fmla v22.4s, v10.4s, v5.s[3]	\n"  \
	"	fmla v27.4s, v11.4s, v5.s[3]	\n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"   fadd v22.4s, v22.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #48]              \n"  \
	"	str q22, [x24, #32]             \n"  \
	"   ldr q22, [x25, #32]             \n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	"   fadd v27.4s, v27.4s, v4.4s      \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	"	str q27, [x24, #48]             \n"  \
	"   ldr q27, [x25, #48]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	\
	\
	\
	"	fmla v13.4s, v8.4s, v6.s[3]		\n"/*row 2*/\
	"	fmla v18.4s, v9.4s, v6.s[3]		\n"  \
	"   fadd v13.4s, v13.4s, v12.4s     \n"  \
	"   ldr q12, [x26]                  \n"  \
	"   fadd v18.4s, v18.4s, v17.4s     \n"  \
	"   ldr q17, [x26, #16]             \n"  \
	"	str q13, [x25]                  \n"  \
	"	str q18, [x25, #16]             \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	\
	"	fmla v23.4s, v10.4s, v6.s[3]	\n"  \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n"  \
	"   fadd v23.4s, v23.4s, v22.4s     \n"  \
	"   ldr q22, [x26, #32]             \n"  \
	"   fadd v28.4s, v28.4s, v27.4s     \n"  \
	"   ldr q27, [x26, #48]             \n"  \
	"	str q23, [x25, #32]             \n"  \
	"	str q28, [x25, #48]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	\
	\
	\
	"	fmla v14.4s, v8.4s, v7.s[3]		\n"/*row 3*/\
	"	fmla v19.4s, v9.4s, v7.s[3]		\n"  \
	"   fadd v14.4s, v14.4s, v12.4s     \n"  \
	"   ldr q12, [x27]                  \n"  \
	"   fadd v19.4s, v19.4s, v17.4s     \n"  \
	"   ldr q17, [x27, #16]             \n"  \
	"	str q14, [x26]                  \n"  \
	"	str q19, [x26, #16]             \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	\
	"	fmla v24.4s, v10.4s, v7.s[3]	\n"  \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n"  \
	"   fadd v24.4s, v24.4s, v22.4s     \n"  \
	"   ldr q22, [x27, #32]             \n"  \
	"   fadd v29.4s, v29.4s, v27.4s     \n"  \
	"   ldr q27, [x27, #48]             \n"  \
	"	str q24, [x26, #32]             \n"  \
	"	str q29, [x26, #48]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	\
	\
	\
	"	fmla v15.4s, v8.4s, v0.s[3]		\n"/*row 4*/\
	"	fmla v20.4s, v9.4s, v0.s[3]		\n"  \
	"   fadd v15.4s, v15.4s, v12.4s     \n"  \
	"   ldr q12, [x28]                  \n"  \
	"   fadd v20.4s, v20.4s, v17.4s     \n"  \
	"   ldr q17, [x28, #16]             \n"  \
	"	str q15, [x27]                  \n"  \
	"	str q20, [x27, #16]             \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	\
	"	fmla v25.4s, v10.4s, v0.s[3]	\n"  \
	"	fmla v30.4s, v11.4s, v0.s[3]	\n"  \
	"   fadd v25.4s, v25.4s, v22.4s     \n"  \
	"   ldr q22, [x28, #32]             \n"  \
	"   fadd v30.4s, v30.4s, v27.4s     \n"  \
	"   ldr q27, [x28, #48]             \n"  \
	"	str q25, [x27, #32]             \n"  \
	"	str q30, [x27, #48]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	\
	\
	\
	"	fmla v16.4s, v8.4s, v1.s[3]		\n"/*row 5*/\
	"	fmla v21.4s, v9.4s, v1.s[3]		\n"  \
	"   fadd v16.4s, v16.4s, v12.4s     \n"  \
	"   fadd v21.4s, v21.4s, v17.4s     \n"  \
	"	str q16, [x28]                  \n"  \
	"	str q21, [x28, #16]             \n"  \
	\
	"	fmla v26.4s, v10.4s, v1.s[3]	\n"  \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n"  \
	"   fadd v26.4s, v26.4s, v22.4s     \n"  \
	"   fadd v31.4s, v31.4s, v27.4s     \n"  \
	"	str q26, [x28, #32]             \n"  \
	"	str q31, [x28, #48]             \n"

#define gemm_ukernel_5x16_u8_last_k_with_ldr_update_norm \
	"	prfm PLDL1STRM, [x24]           \n"  \
	"	prfm PLDL1STRM, [x25]           \n"  \
	"	prfm PLDL1STRM, [x26]           \n"  \
	"	prfm PLDL1STRM, [x27]           \n"  \
	"	prfm PLDL1STRM, [x28]           \n"  \
	"	prfm PLDL1STRM, [x3]            \n"  \
	"   prfm PLDL1STRM, [x4]            \n"  \
	\
	/*v2:1.0  v3:exp(m1-m2)  v4:ldr  v5: sum*/\
	"   fmov v2.4s, #1.0                \n"  \
	"   ld1r {v3.4s}, [x3]              \n"  \
	"   add x3, x3, #4                  \n"  \
	"	ldr q4, [x24]                   \n"  \
	\
	"	fmla v12.4s, v8.4s, v5.s[3]		\n"/*row 1*/\
	"	fmla v17.4s, v9.4s, v5.s[3]		\n"  \
	"	fmla v22.4s, v10.4s, v5.s[3]	\n"  \
	"	fmla v27.4s, v11.4s, v5.s[3]	\n"  \
	"   ld1r {v5.4s}, [x4]              \n"  \
	"   add x4, x4, #4                  \n"  \
	"   fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"	fadd v12.4s, v12.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #16]              \n"  \
	"   fmul v12.4s, v12.4s, v5.4s      \n"  \
	"	str q12, [x24]                  \n"  \
	"   ldr q12, [x25]                  \n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"	fadd v17.4s, v17.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #32]              \n"  \
	"   fmul v17.4s, v17.4s, v5.4s      \n"  \
	"	str q17, [x24, #16]             \n"  \
	"   ldr q17, [x25, #16]             \n"  \
	\
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"   fadd v22.4s, v22.4s, v4.4s      \n"  \
	"	ldr q4, [x24, #48]              \n"  \
	"   fmul v22.4s, v22.4s, v5.4s      \n"  \
	"	str q22, [x24, #32]             \n"  \
	"   ldr q22, [x25, #32]             \n"  \
	"   fmul v4.4s, v4.4s, v3.4s        \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	"   fadd v27.4s, v27.4s, v4.4s      \n"  \
	"   fmul v27.4s, v27.4s, v5.4s      \n"  \
	"   ld1r {v5.4s}, [x4]              \n"  \
	"   add x4, x4, #4                  \n"  \
	"   fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	"	str q27, [x24, #48]             \n"  \
	"   ldr q27, [x25, #48]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	\
	\
	\
	"	fmla v13.4s, v8.4s, v6.s[3]		\n"/*row 2*/\
	"	fmla v18.4s, v9.4s, v6.s[3]		\n"  \
	"   fadd v13.4s, v13.4s, v12.4s     \n"  \
	"   ldr q12, [x26]                  \n"  \
	"   fadd v18.4s, v18.4s, v17.4s     \n"  \
	"   ldr q17, [x26, #16]             \n"  \
	"   fmul v13.4s, v13.4s, v5.4s      \n"  \
	"   fmul v18.4s, v18.4s, v5.4s      \n"  \
	"	str q13, [x25]                  \n"  \
	"	str q18, [x25, #16]             \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	\
	"	fmla v23.4s, v10.4s, v6.s[3]	\n"  \
	"	fmla v28.4s, v11.4s, v6.s[3]	\n"  \
	"   fadd v23.4s, v23.4s, v22.4s     \n"  \
	"   ldr q22, [x26, #32]             \n"  \
	"   fadd v28.4s, v28.4s, v27.4s     \n"  \
	"   ldr q27, [x26, #48]             \n"  \
	"   fmul v23.4s, v23.4s, v5.4s      \n"  \
	"   fmul v28.4s, v28.4s, v5.4s      \n"  \
	"   ld1r {v5.4s}, [x4]              \n"  \
	"   add x4, x4, #4                  \n"  \
	"   fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"	str q23, [x25, #32]             \n"  \
	"	str q28, [x25, #48]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	\
	\
	\
	"	fmla v14.4s, v8.4s, v7.s[3]		\n"/*row 3*/\
	"	fmla v19.4s, v9.4s, v7.s[3]		\n"  \
	"   fadd v14.4s, v14.4s, v12.4s     \n"  \
	"   ldr q12, [x27]                  \n"  \
	"   fadd v19.4s, v19.4s, v17.4s     \n"  \
	"   ldr q17, [x27, #16]             \n"  \
	"   fmul v14.4s, v14.4s, v5.4s      \n"  \
	"   fmul v19.4s, v19.4s, v5.4s      \n"  \
	"	str q19, [x26, #16]             \n"  \
	"	str q14, [x26]                  \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	\
	"	fmla v24.4s, v10.4s, v7.s[3]	\n"  \
	"	fmla v29.4s, v11.4s, v7.s[3]	\n"  \
	"   fadd v24.4s, v24.4s, v22.4s     \n"  \
	"   ldr q22, [x27, #32]             \n"  \
	"   fadd v29.4s, v29.4s, v27.4s     \n"  \
	"   ldr q27, [x27, #48]             \n"  \
	"   fmul v24.4s, v24.4s, v5.4s      \n"  \
	"   fmul v29.4s, v29.4s, v5.4s      \n"  \
	"   ld1r {v5.4s}, [x4]              \n"  \
	"   add x4, x4, #4                  \n"  \
	"   fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"	str q29, [x26, #48]             \n"  \
	"	str q24, [x26, #32]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	"	ld1r {v3.4s}, [x3]              \n"  \
	"	add x3, x3, #4                  \n"  \
	\
	\
	\
	"	fmla v15.4s, v8.4s, v0.s[3]		\n"/*row 4*/\
	"	fmla v20.4s, v9.4s, v0.s[3]		\n"  \
	"   fadd v15.4s, v15.4s, v12.4s     \n"  \
	"   ldr q12, [x28]                  \n"  \
	"   fadd v20.4s, v20.4s, v17.4s     \n"  \
	"   ldr q17, [x28, #16]             \n"  \
	"   fmul v15.4s, v15.4s, v5.4s      \n"  \
	"   fmul v20.4s, v20.4s, v5.4s      \n"  \
	"	str q15, [x27]                  \n"  \
	"	str q20, [x27, #16]             \n"  \
	"	fmul v12.4s, v12.4s, v3.4s      \n"  \
	"	fmul v17.4s, v17.4s, v3.4s      \n"  \
	\
	"	fmla v25.4s, v10.4s, v0.s[3]	\n"  \
	"	fmla v30.4s, v11.4s, v0.s[3]	\n"  \
	"   fadd v25.4s, v25.4s, v22.4s     \n"  \
	"   ldr q22, [x28, #32]             \n"  \
	"   fadd v30.4s, v30.4s, v27.4s     \n"  \
	"   ldr q27, [x28, #48]             \n"  \
	"   fmul v25.4s, v25.4s, v5.4s      \n"  \
	"   fmul v30.4s, v30.4s, v5.4s      \n"  \
	"   ld1r {v5.4s}, [x4]              \n"  \
	"   add x4, x4, #4                  \n"  \
	"   fdiv v5.4s, v2.4s, v5.4s        \n"  \
	"	str q30, [x27, #48]             \n"  \
	"	str q25, [x27, #32]             \n"  \
	"	fmul v22.4s, v22.4s, v3.4s      \n"  \
	"	fmul v27.4s, v27.4s, v3.4s      \n"  \
	\
	\
	\
	"	fmla v16.4s, v8.4s, v1.s[3]		\n"/*row 5*/\
	"	fmla v21.4s, v9.4s, v1.s[3]		\n"  \
	"   fadd v16.4s, v16.4s, v12.4s     \n"  \
	"   fadd v21.4s, v21.4s, v17.4s     \n"  \
	"   fmul v16.4s, v16.4s, v5.4s      \n"  \
	"   fmul v21.4s, v21.4s, v5.4s      \n"  \
	"	str q16, [x28]                  \n"  \
	"	str q21, [x28, #16]             \n"  \
	\
	"	fmla v26.4s, v10.4s, v1.s[3]	\n"  \
	"	fmla v31.4s, v11.4s, v1.s[3]	\n"  \
	"   fadd v26.4s, v26.4s, v22.4s     \n"  \
	"   fadd v31.4s, v31.4s, v27.4s     \n"  \
	"   fmul v26.4s, v26.4s, v5.4s      \n"  \
	"   fmul v31.4s, v31.4s, v5.4s      \n"  \
	"	str q26, [x28, #32]             \n"  \
	"	str q31, [x28, #48]             \n"


void fused_exp_sum_scorexv_norm_kernel_1(long is_last_block,
					long m, long n/*(head_dim==64)*/, long k, long ldk/*careful,todo*/,
					float *score, float *v, float *out, long ldo,
					float *max_per_line, float *exp_sum_per_line)
{
	float *addr_a, *addr_b, *addr_c, *addr_max, *addr_sum;
	long ldb_in_bytes=sizeof(float)*n, ldc_in_bytes=sizeof(float)*ldo;

	long num_tile=((k+b3-1)/b3), tile_index=0;

	for(long ps=0; ps<k; ps+=b3) {
		long k_stride=min(k-ps, b3);
		addr_a=(score+r1*ps), addr_b=(v+ps*n), addr_c=out;

		tile_index++;
		for(long is=0; is<m; is+=r1) {
			addr_a=(score+r1*ps+is*ldk); addr_b=(v+ps*n); addr_c=(out+is*ldo);
						addr_max=(max_per_line+is); addr_sum=(exp_sum_per_line+is);

			// exp sum
			{
				meformer_exp_init
				int batch;
				float32x4_t vi_max;
				float *input, *output, *sum;
				for(int i=0; i<r1; i++) {
					batch=sizeof(float)*k_stride;
					vi_max=vld1q_dup_f32(addr_max+i);
					input=addr_a, output=addr_a;
					sum=addr_sum+i;
					meformer_minusmax_exp_sum(batch, input, vi_max, output, sum, 20);
					addr_a+=4;
				}
			}
			addr_a=(score+r1*ps+is*ldk);

			for(long js=0; js<n; js+=r2) {
				// kernel
				{
					asm volatile(
						"ldr x0, %[ldc_in_bytes]               \n"
						"ldr x1, %[ldb_in_bytes]               \n"
						"ldr x4, %[addr_sum]                   \n"

						"ldr x3, %[num_tile]                   \n"
						"ldr x5, %[tile_index]                 \n"
						"ldr x6, %[is_last_block]              \n"

						"ldr x15, %[a]                         \n"
						"ldr x20, %[b]                         \n"

						"ldr x24, %[c]                         \n"
						"add x25, x24, x0                      \n"
						"add x26, x25, x0                      \n"
						"add x27, x26, x0                      \n"
						"add x28, x27, x0                      \n"

						"gemm_5x16_start:                      \n"
						"   ldr x2, %[k_stride]                \n"
						"   lsr x2, x2, #3                     \n"
							gemm_ukernel_5x16_u8_first_k
						"   b gemm_5x16_body                   \n"

						"gemm_5x16_mid:                        \n"
							gemm_ukernel_5x16_u8_k0

						"gemm_5x16_body:                       \n"
							gemm_ukernel_5x16_u8_k1
							gemm_ukernel_5x16_u8_k2
							gemm_ukernel_5x16_u8_k3
							gemm_ukernel_5x16_u8_k4
							gemm_ukernel_5x16_u8_k5
							gemm_ukernel_5x16_u8_k6
						"   subs x2, x2, #1                   \n"
						"   beq gemm_5x16_end                 \n"
							gemm_ukernel_5x16_u8_k7
						"   b gemm_5x16_mid                   \n"

						// -------------------------------------
						"gemm_5x16_end:                       \n"
						"	cmp x6, #1                        \n" // last block
						"	bne gemm_5x16_1bxb_end            \n"

						"	cmp x3, #1                        \n" // not last block
						"	beq branch_1blb_1t                \n"

						"	cmp x3, #2                        \n"
						"	beq branch_1blb_2t                \n"

						"	b branch_1blb_xt                  \n"

						"gemm_5x16_1bxb_end:                  \n"
						"	cmp x5, #1                        \n"
						"	beq branch_1bxb_1t                \n"

						"	b branch_1bxb_xt                  \n"

						// -------------------------------------
						"branch_1blb_1t:                      \n"
							gemm_ukernel_5x16_u8_last_k_with_norm
						"	b branch_nop                      \n"

						"branch_1blb_2t:                      \n"
						"	cmp x5, #1                        \n"
						"	beq branch_1blb_2t_1              \n"
						"	b branch_1blb_2t_2                \n"

						"branch_1blb_xt:                      \n"
						"	cmp x5, #1                        \n"
						"	beq branch_1blb_xt_1              \n"

						"	cmp x5, x3                        \n"
						"	beq branch_1blb_xt_3              \n"

						"	b branch_1blb_xt_2                \n"

						"branch_1bxb_1t:                      \n"
							gemm_ukernel_5x16_u8_last_k
						"	b branch_nop                      \n"

						"branch_1bxb_xt:                      \n"
							gemm_ukernel_5x16_u8_last_k_with_ldr
						"	b branch_nop                      \n"

						// -------------------------------------
						"branch_1blb_2t_1:                    \n"
							gemm_ukernel_5x16_u8_last_k
						"	b branch_nop                      \n"

						"branch_1blb_2t_2:                    \n"
							gemm_ukernel_5x16_u8_last_k_with_ldr_norm
						"	b branch_nop                      \n"

						"branch_1blb_xt_1:                    \n"
							gemm_ukernel_5x16_u8_last_k
						"	b branch_nop                      \n"

						"branch_1blb_xt_2:                    \n"
							gemm_ukernel_5x16_u8_last_k_with_ldr
						"	b branch_nop                      \n"

						"branch_1blb_xt_3:                    \n"
							gemm_ukernel_5x16_u8_last_k_with_ldr_norm
						"	b branch_nop                      \n"

						"branch_nop:                          \n"
						"	nop                               \n"
						:
						: [a]  "m" (addr_a),
						  [b]  "m" (addr_b),
						  [c]  "m" (addr_c),
						  [k_stride] "m" (k_stride),
						  [ldb_in_bytes] "m" (ldb_in_bytes),
						  [ldc_in_bytes] "m" (ldc_in_bytes),
						  [addr_sum] "m" (addr_sum),
						  [tile_index] "m" (tile_index),
						  [num_tile] "m" (num_tile),
						  [is_last_block] "m" (is_last_block)
						: "cc", "memory",
						  "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
						  "x9", "x10", "x11", "x12", "x13","x14", "x15", "x16",
						  "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24","x25",
						  "x26", "x27", "x28", "x29", "x30",
						  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
						  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
						  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
						  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
					);
				}
				addr_b+=r2, addr_c+=r2;
			}
		}
	}
}


void fused_exp_sum_scorexv_norm_kernel_2(long is_last_block,
					long m, long n/*(head_dim==64)*/, long k, long ldk/*careful,todo*/,
					float *score, float *v, float *out, long ldo,
					float *max_per_line, float *max_for_update,
					float *exp_sum_per_line)
{
	float *addr_a, *addr_b, *addr_c, *addr_max, *addr_update, *addr_sum;
	long ldb_in_bytes=sizeof(float)*n, ldc_in_bytes=sizeof(float)*ldo;

	// update max, update sum
	addr_max=max_per_line, addr_update=max_for_update; addr_sum=exp_sum_per_line;
	{
		// m1-=m2
		cblas_saxpy(m,
				-1.0, addr_max, 1,
					  addr_update, 1);

		// m1=exp(m1-m2)
		meformer_exp_init            
		int batch=m*sizeof(float);
		float *input=addr_update, *output=addr_update;
		meformer_exp(batch, input, output, 4);

		// sum=sum*exp(m1-m2)
		int m_left=m, m_count=0;
		for(;m_left>=32;m_left-=32,m_count+=32) {
			float32x4_t vsum_0, vsum_1, vsum_2, vsum_3,
						vsum_4, vsum_5, vsum_6, vsum_7;
			float32x4_t vexp_0, vexp_1, vexp_2, vexp_3,
						vexp_4, vexp_5, vexp_6, vexp_7;

			vsum_0=vld1q_f32(addr_sum+m_count+0);
			vexp_0=vld1q_f32(addr_update+m_count+0);
			vsum_0=vmulq_f32(vsum_0, vexp_0);
			vst1q_f32(addr_sum+m_count+0, vsum_0);

			vsum_1=vld1q_f32(addr_sum+m_count+4);
			vexp_1=vld1q_f32(addr_update+m_count+4);
			vsum_1=vmulq_f32(vsum_1, vexp_1);
			vst1q_f32(addr_sum+m_count+4, vsum_1);

			vsum_2=vld1q_f32(addr_sum+m_count+8);
			vexp_2=vld1q_f32(addr_update+m_count+8);
			vsum_2=vmulq_f32(vsum_2, vexp_2);
			vst1q_f32(addr_sum+m_count+8, vsum_2);

			vsum_3=vld1q_f32(addr_sum+m_count+12);
			vexp_3=vld1q_f32(addr_update+m_count+12);
			vsum_3=vmulq_f32(vsum_3, vexp_3);
			vst1q_f32(addr_sum+m_count+12, vsum_3);

			vsum_4=vld1q_f32(addr_sum+m_count+16);
			vexp_4=vld1q_f32(addr_update+m_count+16);
			vsum_4=vmulq_f32(vsum_4, vexp_4);
			vst1q_f32(addr_sum+m_count+16, vsum_4);

			vsum_5=vld1q_f32(addr_sum+m_count+20);
			vexp_5=vld1q_f32(addr_update+m_count+20);
			vsum_5=vmulq_f32(vsum_5, vexp_5);
			vst1q_f32(addr_sum+m_count+20, vsum_5);

			vsum_6=vld1q_f32(addr_sum+m_count+24);
			vexp_6=vld1q_f32(addr_update+m_count+24);
			vsum_6=vmulq_f32(vsum_6, vexp_6);
			vst1q_f32(addr_sum+m_count+24, vsum_6);

			vsum_7=vld1q_f32(addr_sum+m_count+28);
			vexp_7=vld1q_f32(addr_update+m_count+28);
			vsum_7=vmulq_f32(vsum_7, vexp_7);
			vst1q_f32(addr_sum+m_count+28, vsum_7);
		}
		for(;m_left>=16;m_left-=16,m_count+=16) {
			float32x4_t vsum_0, vsum_1, vsum_2, vsum_3;
			float32x4_t vexp_0, vexp_1, vexp_2, vexp_3;

			vsum_0=vld1q_f32(addr_sum+m_count+0);
			vexp_0=vld1q_f32(addr_update+m_count+0);
			vsum_0=vmulq_f32(vsum_0, vexp_0);
			vst1q_f32(addr_sum+m_count+0, vsum_0);

			vsum_1=vld1q_f32(addr_sum+m_count+4);
			vexp_1=vld1q_f32(addr_update+m_count+4);
			vsum_1=vmulq_f32(vsum_1, vexp_1);
			vst1q_f32(addr_sum+m_count+4, vsum_1);

			vsum_2=vld1q_f32(addr_sum+m_count+8);
			vexp_2=vld1q_f32(addr_update+m_count+8);
			vsum_2=vmulq_f32(vsum_2, vexp_2);
			vst1q_f32(addr_sum+m_count+8, vsum_2);

			vsum_3=vld1q_f32(addr_sum+m_count+12);
			vexp_3=vld1q_f32(addr_update+m_count+12);
			vsum_3=vmulq_f32(vsum_3, vexp_3);
			vst1q_f32(addr_sum+m_count+12, vsum_3);
		}
		for(;m_left>=8;m_left-=8,m_count+=8) {
			float32x4_t vsum_0, vsum_1;
			float32x4_t vexp_0, vexp_1;

			vsum_0=vld1q_f32(addr_sum+m_count+0);
			vexp_0=vld1q_f32(addr_update+m_count+0);
			vsum_0=vmulq_f32(vsum_0, vexp_0);
			vst1q_f32(addr_sum+m_count+0, vsum_0);

			vsum_1=vld1q_f32(addr_sum+m_count+4);
			vexp_1=vld1q_f32(addr_update+m_count+4);
			vsum_1=vmulq_f32(vsum_1, vexp_1);
			vst1q_f32(addr_sum+m_count+4, vsum_1);
		}
		for(;m_left>=4;m_left-=4,m_count+=4) {
			float32x4_t vsum_0;
			float32x4_t vexp_0;

			vsum_0=vld1q_f32(addr_sum+m_count+0);
			vexp_0=vld1q_f32(addr_update+m_count+0);
			vsum_0=vmulq_f32(vsum_0, vexp_0);
			vst1q_f32(addr_sum+m_count+0, vsum_0);
		}
		for(;m_left>0;m_left--,m_count++) {
			addr_sum[m_count]=(addr_sum[m_count]*addr_update[m_count]);
		}
	}

	long num_tile=((k+b3-1)/b3), tile_index=0;

	for(long ps=0; ps<k; ps+=b3) {
		long k_stride=min(k-ps, b3);
		addr_a=(score+r1*ps), addr_b=(v+ps*n), addr_c=out;

		tile_index++;
		for(long is=0; is<m; is+=r1) {
			addr_a=(score+r1*ps+is*ldk), addr_b=(v+ps*n), addr_c=(out+is*ldo);
						addr_max=(max_per_line+is);
						addr_sum=(exp_sum_per_line+is);

			// exp sum
			{
				meformer_exp_init
				int batch;
				float32x4_t vi_max;
				float *input, *output, *sum;
				for(int i=0; i<r1; i++) {
					batch=sizeof(float)*k_stride;
					vi_max=vld1q_dup_f32(addr_max+i);
					input=addr_a, output=addr_a;
					sum=addr_sum+i;
					meformer_minusmax_exp_sum(batch, input, vi_max, output, sum, 20);
					addr_a+=4;
				}
			}
			addr_a=(score+r1*ps+is*ldk);

			addr_update=(max_for_update+is);
			for(long js=0; js<n; js+=r2) {
				// kernel
				{
					asm volatile(
						"ldr x0, %[ldc_in_bytes]               \n"
						"ldr x1, %[ldb_in_bytes]               \n"
						"ldr x3, %[addr_update]                \n"
						"ldr x4, %[addr_sum]                   \n"

						"ldr x5, %[num_tile]                   \n"
						"ldr x6, %[tile_index]                 \n"
						"ldr x7, %[is_last_block]              \n"

						"ldr x15, %[a]                         \n"
						"ldr x20, %[b]                         \n"

						"ldr x24, %[c]                         \n"
						"add x25, x24, x0                      \n"
						"add x26, x25, x0                      \n"
						"add x27, x26, x0                      \n"
						"add x28, x27, x0                      \n"

						"gemm_5x16_start_1103:                 \n"
						"   ldr x2, %[k_stride]                \n"
						"   lsr x2, x2, #3                     \n"

							gemm_ukernel_5x16_u8_first_k
						"   b gemm_5x16_body_1103              \n"

						"gemm_5x16_mid_1103:                   \n"
							gemm_ukernel_5x16_u8_k0

						"gemm_5x16_body_1103:                  \n"
							gemm_ukernel_5x16_u8_k1
							gemm_ukernel_5x16_u8_k2
							gemm_ukernel_5x16_u8_k3
							gemm_ukernel_5x16_u8_k4
							gemm_ukernel_5x16_u8_k5
							gemm_ukernel_5x16_u8_k6
						"   subs x2, x2, #1                   \n"
						"   beq gemm_5x16_end_1103            \n"
							gemm_ukernel_5x16_u8_k7
						"   b gemm_5x16_mid_1103              \n"

						// -------------------------------------
						"gemm_5x16_end_1103:                  \n"
						"	cmp x7, #1                        \n"
						"	bne branch_xbxb_1103              \n"

						"	cmp x5, #1                        \n"
						"	beq branch_xblb_1t_1103           \n"

						"	cmp x5, #2                        \n"
						"	beq branch_xblb_2t_1103           \n"

						"	b branch_xblb_xt_1103             \n"

						// -------------------------------------
						"branch_xblb_1t_1103:                 \n"
							gemm_ukernel_5x16_u8_last_k_with_ldr_update_norm
						"	b branch_nop_1103                 \n"

						"branch_xblb_2t_1103:                 \n"
						"	cmp x6, #1                        \n"
						"	beq branch_xblb_2t_1_1103         \n"

						"	b branch_xblb_2t_2_1103           \n"

						"branch_xblb_xt_1103:                 \n"
						"	cmp x6, #1                        \n"
						"	beq branch_xblb_xt_1_1103         \n"

						"	cmp x6, x5                        \n"
						"	beq branch_xblb_xt_3_1103         \n"

						"	b branch_xblb_xt_2_1103           \n"

						"branch_xbxb_1103:                    \n"
						"	cmp x6, #1                        \n"
						"	beq branch_xbxb_1_1103            \n"

						"	b branch_xbxb_2_1103              \n"

						// -------------------------------------
						"branch_xblb_2t_1_1103:               \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr_update
						"	b branch_nop_1103                 \n"

						"branch_xblb_2t_2_1103:               \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr_norm
						"	b branch_nop_1103                 \n"

						"branch_xblb_xt_1_1103:               \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr_update
						"	b branch_nop_1103                 \n"

						"branch_xblb_xt_2_1103:               \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr
						"	b branch_nop_1103                 \n"

						"branch_xblb_xt_3_1103:               \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr_norm
						"	b branch_nop_1103                 \n"

						"branch_xbxb_1_1103:                  \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr_update
						"	b branch_nop_1103                 \n"

						"branch_xbxb_2_1103:                  \n"
						gemm_ukernel_5x16_u8_last_k_with_ldr
						"	b branch_nop_1103                 \n"

						"branch_nop_1103:                     \n"
						"	nop                               \n"

						:
						: [a] "m" (addr_a),
						  [b] "m" (addr_b),
						  [c] "m" (addr_c),
						  [k_stride] "m" (k_stride),
						  [ldb_in_bytes] "m" (ldb_in_bytes),
						  [ldc_in_bytes] "m" (ldc_in_bytes),
						  [ps] "m" (ps),
						  [addr_update] "m" (addr_update),
						  [addr_sum] "m" (addr_sum),
						  [num_tile] "m" (num_tile),
						  [tile_index] "m" (tile_index),
						  [is_last_block] "m" (is_last_block)
						: "cc", "memory",
						  "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
						  "x9", "x10", "x11", "x12", "x13","x14", "x15", "x16",
						  "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24","x25",
						  "x26", "x27", "x28", "x29", "x30",
						  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
						  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
						  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
						  "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
					);
				}
				addr_b+=r2, addr_c+=r2;
			}
		}
	}
}

