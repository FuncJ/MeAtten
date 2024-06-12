#include "meformer.h"

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
	*sum = vacc_lo;

#define pack_ukernel_5x16_u8_first_k        \
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
    "	add x20, x20, x6    			\n" \
    \
    "	fmul v17.4s, v9.4s, v0.s[0]		\n" \
    "	fmul v18.4s, v9.4s, v1.s[0]		\n" \
    "	str q8, [x21], #16              \n" \
    "	str q9, [x21], #16              \n" \
    "	fmul v19.4s, v9.4s, v2.s[0]		\n" \
    "	fmul v20.4s, v9.4s, v3.s[0]		\n" \
    "	fmul v21.4s, v9.4s, v4.s[0]		\n" \
    "	ldr q8, [x20]					\n" \
    "	ldr q9, [x20, #16]				\n" \
    \
    "	fmul v22.4s, v10.4s, v0.s[0]	\n" \
    "	str q10, [x21], #16             \n" \
    "	str q11, [x21], #16             \n" \
    "	fmul v23.4s, v10.4s, v1.s[0]	\n" \
    "	fmul v24.4s, v10.4s, v2.s[0]	\n" \
    "	str q8, [x21], #16              \n" \
    "	str q9, [x21], #16              \n" \
    "	fmul v25.4s, v10.4s, v3.s[0]	\n" \
    "	fmul v26.4s, v10.4s, v4.s[0]	\n" \
    "	ldr q10, [x20, #32]				\n" \
    "	str q10, [x21], #16             \n" \
    "	ldr q5, [x15], #16				\n" \
    \
    "	fmul v27.4s, v11.4s, v0.s[0]	\n" \
    "	fmul v28.4s, v11.4s, v1.s[0]	\n" \
    "	fmul v29.4s, v11.4s, v2.s[0]	\n" \
    "	fmul v30.4s, v11.4s, v3.s[0]	\n" \
    "	fmul v31.4s, v11.4s, v4.s[0]	\n" \
    "	ldr q11, [x20, #48]				\n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6    			\n"

#define pack_ukernel_5x16_u8_k0 			\
    "	fmla v12.4s, v8.4s, v0.s[0]		\n" \
    "	fmla v13.4s, v8.4s, v1.s[0]		\n" \
    "	fmla v14.4s, v8.4s, v2.s[0]		\n" \
    "	fmla v15.4s, v8.4s, v3.s[0]		\n" \
    "	fmla v16.4s, v8.4s, v4.s[0]		\n" \
    "	ldr q8, [x20]					\n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v0.s[0]		\n" \
    "	fmla v18.4s, v9.4s, v1.s[0]		\n" \
    "	fmla v19.4s, v9.4s, v2.s[0]		\n" \
    "	fmla v20.4s, v9.4s, v3.s[0]		\n" \
    "	fmla v21.4s, v9.4s, v4.s[0]		\n" \
    "	ldr q9, [x20, #16]				\n" \
    "	str q9, [x21], #16              \n" \
    \
    "	fmla v22.4s, v10.4s, v0.s[0]	\n" \
    "	fmla v23.4s, v10.4s, v1.s[0]	\n" \
    "	fmla v24.4s, v10.4s, v2.s[0]	\n" \
    "	fmla v25.4s, v10.4s, v3.s[0]	\n" \
    "	fmla v26.4s, v10.4s, v4.s[0]	\n" \
    "	ldr q10, [x20, #32]				\n" \
    "	str q10, [x21], #16              \n" \
    "	ldr q5, [x15], #16				\n" \
    \
    "	fmla v27.4s, v11.4s, v0.s[0]	\n" \
    "	fmla v28.4s, v11.4s, v1.s[0]	\n" \
    "	fmla v29.4s, v11.4s, v2.s[0]	\n" \
    "	fmla v30.4s, v11.4s, v3.s[0]	\n" \
    "	fmla v31.4s, v11.4s, v4.s[0]	\n" \
    "	ldr q11, [x20, #48]				\n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6    			\n"

#define pack_ukernel_5x16_u8_k1 			\
    "	fmla v12.4s, v8.4s, v0.s[1]		\n" \
    "	fmla v13.4s, v8.4s, v1.s[1]		\n" \
    "	fmla v14.4s, v8.4s, v2.s[1]		\n" \
    "	fmla v15.4s, v8.4s, v3.s[1]		\n" \
    "	fmla v16.4s, v8.4s, v4.s[1]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v0.s[1]		\n" \
    "	fmla v18.4s, v9.4s, v1.s[1]		\n" \
    "	fmla v19.4s, v9.4s, v2.s[1]		\n" \
    "	fmla v20.4s, v9.4s, v3.s[1]		\n" \
    "	fmla v21.4s, v9.4s, v4.s[1]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
    "	ldr q6, [x15], #16              \n" \
    \
    "	fmla v22.4s, v10.4s, v0.s[1]	\n" \
    "	fmla v23.4s, v10.4s, v1.s[1]	\n" \
    "	fmla v24.4s, v10.4s, v2.s[1]	\n" \
    "	fmla v25.4s, v10.4s, v3.s[1]	\n" \
    "	fmla v26.4s, v10.4s, v4.s[1]	\n" \
    "	ldr q10, [x20, #32]             \n" \
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v27.4s, v11.4s, v0.s[1]	\n" \
    "	fmla v28.4s, v11.4s, v1.s[1]	\n" \
    "	fmla v29.4s, v11.4s, v2.s[1]	\n" \
    "	fmla v30.4s, v11.4s, v3.s[1]	\n" \
    "	fmla v31.4s, v11.4s, v4.s[1]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

#define pack_ukernel_5x16_u8_k2 			\
    "	fmla v12.4s, v8.4s, v0.s[2]		\n" \
    "	fmla v13.4s, v8.4s, v1.s[2]		\n" \
    "	fmla v14.4s, v8.4s, v2.s[2]		\n" \
    "	fmla v15.4s, v8.4s, v3.s[2]		\n" \
    "	fmla v16.4s, v8.4s, v4.s[2]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v0.s[2]		\n" \
    "	fmla v18.4s, v9.4s, v1.s[2]		\n" \
    "	fmla v19.4s, v9.4s, v2.s[2]		\n" \
    "	fmla v20.4s, v9.4s, v3.s[2]		\n" \
    "	fmla v21.4s, v9.4s, v4.s[2]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
    "	ldr q7, [x15], #16              \n" \
    \
    "	fmla v22.4s, v10.4s, v0.s[2]	\n" \
    "	fmla v23.4s, v10.4s, v1.s[2]	\n" \
    "	fmla v24.4s, v10.4s, v2.s[2]	\n" \
    "	fmla v25.4s, v10.4s, v3.s[2]	\n" \
    "	fmla v26.4s, v10.4s, v4.s[2]	\n" \
    "	ldr q10, [x20, #32]             \n" \
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v27.4s, v11.4s, v0.s[2]	\n" \
    "	fmla v28.4s, v11.4s, v1.s[2]	\n" \
    "	fmla v29.4s, v11.4s, v2.s[2]	\n" \
    "	fmla v30.4s, v11.4s, v3.s[2]	\n" \
    "	fmla v31.4s, v11.4s, v4.s[2]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

#define pack_ukernel_5x16_u8_k3 			\
    "	fmla v12.4s, v8.4s, v0.s[3]		\n" \
    "	fmla v13.4s, v8.4s, v1.s[3]		\n" \
    "	fmla v14.4s, v8.4s, v2.s[3]		\n" \
    "	fmla v15.4s, v8.4s, v3.s[3]		\n" \
    "	fmla v16.4s, v8.4s, v4.s[3]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v0.s[3]		\n" \
    "	fmla v18.4s, v9.4s, v1.s[3]		\n" \
    "	fmla v19.4s, v9.4s, v2.s[3]		\n" \
    "	fmla v20.4s, v9.4s, v3.s[3]		\n" \
    "	fmla v21.4s, v9.4s, v4.s[3]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
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
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v29.4s, v11.4s, v2.s[3]	\n" \
    "	fmla v30.4s, v11.4s, v3.s[3]	\n" \
    "	fmla v31.4s, v11.4s, v4.s[3]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

#define pack_ukernel_5x16_u8_k4 			\
    "	fmla v12.4s, v8.4s, v5.s[0]		\n" \
    "	fmla v13.4s, v8.4s, v6.s[0]		\n" \
    "	fmla v14.4s, v8.4s, v7.s[0]		\n" \
    "	fmla v15.4s, v8.4s, v0.s[0]		\n" \
    "	fmla v16.4s, v8.4s, v1.s[0]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v5.s[0]		\n" \
    "	fmla v18.4s, v9.4s, v6.s[0]		\n" \
    "	fmla v19.4s, v9.4s, v7.s[0]		\n" \
    "	fmla v20.4s, v9.4s, v0.s[0]		\n" \
    "	fmla v21.4s, v9.4s, v1.s[0]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
    "	ldr q2, [x16, #32]              \n" \
    \
    "	fmla v22.4s, v10.4s, v5.s[0]	\n" \
    "	fmla v23.4s, v10.4s, v6.s[0]	\n" \
    "	fmla v24.4s, v10.4s, v7.s[0]	\n" \
    "	fmla v25.4s, v10.4s, v0.s[0]	\n" \
    "	fmla v26.4s, v10.4s, v1.s[0]	\n" \
    "	ldr q10, [x20, #32]             \n" \
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v27.4s, v11.4s, v5.s[0]	\n" \
    "	fmla v28.4s, v11.4s, v6.s[0]	\n" \
    "	fmla v29.4s, v11.4s, v7.s[0]	\n" \
    "	fmla v30.4s, v11.4s, v0.s[0]	\n" \
    "	fmla v31.4s, v11.4s, v1.s[0]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

#define pack_ukernel_5x16_u8_k5 			\
    "	fmla v12.4s, v8.4s, v5.s[1]		\n" \
    "	fmla v13.4s, v8.4s, v6.s[1]		\n" \
    "	fmla v14.4s, v8.4s, v7.s[1]		\n" \
    "	fmla v15.4s, v8.4s, v0.s[1]		\n" \
    "	fmla v16.4s, v8.4s, v1.s[1]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v5.s[1]		\n" \
    "	fmla v18.4s, v9.4s, v6.s[1]		\n" \
    "	fmla v19.4s, v9.4s, v7.s[1]		\n" \
    "	fmla v20.4s, v9.4s, v0.s[1]		\n" \
    "	fmla v21.4s, v9.4s, v1.s[1]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
    "	ldr q3, [x16, #48]              \n" \
    \
    "	fmla v22.4s, v10.4s, v5.s[1]	\n" \
    "	fmla v23.4s, v10.4s, v6.s[1]	\n" \
    "	fmla v24.4s, v10.4s, v7.s[1]	\n" \
    "	fmla v25.4s, v10.4s, v0.s[1]	\n" \
    "	fmla v26.4s, v10.4s, v1.s[1]	\n" \
    "	ldr q10, [x20, #32]             \n" \
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v27.4s, v11.4s, v5.s[1]	\n" \
    "	fmla v28.4s, v11.4s, v6.s[1]	\n" \
    "	fmla v29.4s, v11.4s, v7.s[1]	\n" \
    "	fmla v30.4s, v11.4s, v0.s[1]	\n" \
    "	fmla v31.4s, v11.4s, v1.s[1]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

#define pack_ukernel_5x16_u8_k6 			\
    "	fmla v12.4s, v8.4s, v5.s[2]		\n" \
    "	fmla v13.4s, v8.4s, v6.s[2]		\n" \
    "	fmla v14.4s, v8.4s, v7.s[2]		\n" \
    "	fmla v15.4s, v8.4s, v0.s[2]		\n" \
    "	fmla v16.4s, v8.4s, v1.s[2]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v5.s[2]		\n" \
    "	fmla v18.4s, v9.4s, v6.s[2]		\n" \
    "	fmla v19.4s, v9.4s, v7.s[2]		\n" \
    "	fmla v20.4s, v9.4s, v0.s[2]		\n" \
    "	fmla v21.4s, v9.4s, v1.s[2]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
    "	ldr q4, [x16, #64]              \n" \
    \
    "	fmla v22.4s, v10.4s, v5.s[2]	\n" \
    "	fmla v23.4s, v10.4s, v6.s[2]	\n" \
    "	fmla v24.4s, v10.4s, v7.s[2]	\n" \
    "	fmla v25.4s, v10.4s, v0.s[2]	\n" \
    "	fmla v26.4s, v10.4s, v1.s[2]	\n" \
    "	ldr q10, [x20, #32]             \n" \
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v27.4s, v11.4s, v5.s[2]	\n" \
    "	fmla v28.4s, v11.4s, v6.s[2]	\n" \
    "	fmla v29.4s, v11.4s, v7.s[2]	\n" \
    "	fmla v30.4s, v11.4s, v0.s[2]	\n" \
    "	fmla v31.4s, v11.4s, v1.s[2]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

#define pack_ukernel_5x16_u8_k7 			\
    "	fmla v12.4s, v8.4s, v5.s[3]		\n" \
    "	fmla v13.4s, v8.4s, v6.s[3]		\n" \
    "	fmla v14.4s, v8.4s, v7.s[3]		\n" \
    "	fmla v15.4s, v8.4s, v0.s[3]		\n" \
    "	fmla v16.4s, v8.4s, v1.s[3]		\n" \
    "	ldr q8, [x20]                   \n" \
    "	str q8, [x21], #16              \n" \
    \
    "	fmla v17.4s, v9.4s, v5.s[3]		\n" \
    "	fmla v18.4s, v9.4s, v6.s[3]		\n" \
    "	fmla v19.4s, v9.4s, v7.s[3]		\n" \
    "	fmla v20.4s, v9.4s, v0.s[3]		\n" \
    "	fmla v21.4s, v9.4s, v1.s[3]		\n" \
    "	ldr q9, [x20, #16]              \n" \
    "	str q9, [x21], #16              \n" \
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
    "	str q10, [x21], #16             \n" \
    \
    "	fmla v27.4s, v11.4s, v5.s[3]	\n" \
    "	fmla v28.4s, v11.4s, v6.s[3]	\n" \
    "	fmla v29.4s, v11.4s, v7.s[3]	\n" \
    "	ldr q11, [x20, #48]             \n" \
    "	str q11, [x21], #16             \n" \
    "	add x20, x20, x6				\n"

// #define pack_ukernel_5x16_u8_last_k 		\
//     "	fmla v12.4s, v8.4s, v5.s[3]		\n" \
//     "	fmla v17.4s, v9.4s, v5.s[3]		\n" \
//     "	fmla v22.4s, v10.4s, v5.s[3]	\n" \
//     "	fmla v27.4s, v11.4s, v5.s[3]	\n" \
//     "	str q12, [x24]                  \n" \
//     "	str q17, [x24, #16]             \n" \
//     "	str q22, [x24, #32]             \n" \
//     "	str q27, [x24, #48]             \n" \
//     \
//     "	fmla v13.4s, v8.4s, v6.s[3]		\n" \
//     "	fmla v18.4s, v9.4s, v6.s[3]		\n" \
//     "	fmla v23.4s, v10.4s, v6.s[3]	\n" \
//     "	fmla v28.4s, v11.4s, v6.s[3]	\n" \
//     "	str q13, [x25]                  \n" \
//     "	str q18, [x25, #16]             \n" \
//     "	str q23, [x25, #32]             \n" \
//     "	str q28, [x25, #48]             \n" \
//     \
//     "	fmla v14.4s, v8.4s, v7.s[3]		\n" \
//     "	fmla v19.4s, v9.4s, v7.s[3]		\n" \
//     "	fmla v24.4s, v10.4s, v7.s[3]	\n" \
//     "	fmla v29.4s, v11.4s, v7.s[3]	\n" \
//     "	str q14, [x26]                  \n" \
//     "	str q19, [x26, #16]             \n" \
//     "	str q24, [x26, #32]             \n" \
//     "	str q29, [x26, #48]             \n" \
//     \
//     "	fmla v15.4s, v8.4s, v0.s[3]		\n" \
//     "	fmla v20.4s, v9.4s, v0.s[3]		\n" \
//     "	fmla v25.4s, v10.4s, v0.s[3]	\n" \
//     "	fmla v30.4s, v11.4s, v0.s[3]	\n" \
//     "	str q15, [x27]                  \n" \
//     "	str q20, [x27, #16]             \n" \
//     "	str q25, [x27, #32]             \n" \
//     "	str q30, [x27, #48]             \n" \
//     \
//     "	fmla v16.4s, v8.4s, v1.s[3]		\n" \
//     "	fmla v21.4s, v9.4s, v1.s[3]		\n" \
//     "	fmla v26.4s, v10.4s, v1.s[3]	\n" \
//     "	fmla v31.4s, v11.4s, v1.s[3]	\n" \
//     "	str q16, [x28]                  \n" \
//     "	str q21, [x28, #16]             \n" \
//     "	str q26, [x28, #32]             \n" \
//     "	str q31, [x28, #48]             \n"

// #define pack_ukernel_5x16_u8_last_k 		\
//     "	fmov v2.4s, #1.0                \n" \
//     \
//     "	ldr q3, [x2]                    \n" \
//     "	fdiv v3.4s, v2.4s, v3.4s        \n" \
//     "	ldr q4, [x2, #16]               \n" \
//     "	fdiv v4.4s, v2.4s, v4.4s        \n" \
//     \
//     "	fmla v12.4s, v8.4s, v5.s[3]		\n" \
//     "	fmla v17.4s, v9.4s, v5.s[3]		\n" \
//     "	fmla v22.4s, v10.4s, v5.s[3]	\n" \
//     "	fmla v27.4s, v11.4s, v5.s[3]	\n" \
//     "	fmul v12.4s, v12.4s, v3.4s      \n" \
//     "	fmul v17.4s, v17.4s, v3.4s      \n" \
//     "	str q12, [x24]                  \n" \
//     "	str q17, [x24, #16]             \n" \
//     "	fmul v22.4s, v22.4s, v3.4s      \n" \
//     "	fmul v27.4s, v27.4s, v3.4s      \n" \
//     "	str q22, [x24, #32]             \n" \
//     "	str q27, [x24, #48]             \n" \
//     \
//     "	ldr q12, [x2, #32]              \n" \
//     "	fdiv v12.4s, v2.4s, v12.4s      \n" \
//     "	ldr q17, [x2, #48]              \n" \
//     "	fdiv v17.4s, v2.4s, v17.4s      \n" \
//     "	ldr q22, [x2, #64]              \n" \
//     "	fdiv v22.4s, v2.4s, v22.4s      \n" \
//     \
//     "	fmla v13.4s, v8.4s, v6.s[3]		\n" \
//     "	fmla v18.4s, v9.4s, v6.s[3]		\n" \
//     "	fmla v23.4s, v10.4s, v6.s[3]	\n" \
//     "	fmla v28.4s, v11.4s, v6.s[3]	\n" \
//     "	fmul v13.4s, v13.4s, v4.4s      \n" \
//     "	fmul v18.4s, v18.4s, v4.4s      \n" \
//     "	str q13, [x25]                  \n" \
//     "	str q18, [x25, #16]             \n" \
//     "	fmul v23.4s, v23.4s, v4.4s      \n" \
//     "	fmul v28.4s, v28.4s, v4.4s      \n" \
//     "	str q23, [x25, #32]             \n" \
//     "	str q28, [x25, #48]             \n" \
//     \
//     "	fmla v14.4s, v8.4s, v7.s[3]		\n" \
//     "	fmla v19.4s, v9.4s, v7.s[3]		\n" \
//     "	fmla v24.4s, v10.4s, v7.s[3]	\n" \
//     "	fmla v29.4s, v11.4s, v7.s[3]	\n" \
//     "	fmul v14.4s, v14.4s, v12.4s     \n" \
//     "	fmul v19.4s, v19.4s, v12.4s     \n" \
//     "	str q14, [x26]                  \n" \
//     "	str q19, [x26, #16]             \n" \
//     "	fmul v24.4s, v24.4s, v12.4s     \n" \
//     "	fmul v29.4s, v29.4s, v12.4s     \n" \
//     "	str q24, [x26, #32]             \n" \
//     "	str q29, [x26, #48]             \n" \
//     \
//     "	fmla v15.4s, v8.4s, v0.s[3]		\n" \
//     "	fmla v20.4s, v9.4s, v0.s[3]		\n" \
//     "	fmla v25.4s, v10.4s, v0.s[3]	\n" \
//     "	fmla v30.4s, v11.4s, v0.s[3]	\n" \
//     "	fmul v15.4s, v15.4s, v17.4s     \n" \
//     "	fmul v20.4s, v20.4s, v17.4s     \n" \
//     "	str q15, [x27]                  \n" \
//     "	str q20, [x27, #16]             \n" \
//     "	fmul v25.4s, v25.4s, v17.4s     \n" \
//     "	fmul v30.4s, v30.4s, v17.4s     \n" \
//     "	str q25, [x27, #32]             \n" \
//     "	str q30, [x27, #48]             \n" \
//     \
//     "	fmla v16.4s, v8.4s, v1.s[3]		\n" \
//     "	fmla v21.4s, v9.4s, v1.s[3]		\n" \
//     "	fmla v26.4s, v10.4s, v1.s[3]	\n" \
//     "	fmla v31.4s, v11.4s, v1.s[3]	\n" \
//     "	fmul v16.4s, v16.4s, v22.4s     \n" \
//     "	fmul v21.4s, v21.4s, v22.4s     \n" \
//     "	str q16, [x28]                  \n" \
//     "	str q21, [x28, #16]             \n" \
//     "	fmul v26.4s, v26.4s, v22.4s     \n" \
//     "	fmul v31.4s, v31.4s, v22.4s     \n" \
//     "	str q26, [x28, #32]             \n" \
//     "	str q31, [x28, #48]             \n"

#define pack_ukernel_5x16_u8_last_k 		\
    "	fmov v2.4s, #1.0                \n" \
    \
    "	ld1r {v3.4s}, [x2]              \n" \
    "   add x2, x2, #16                 \n" \
    "	fdiv v3.4s, v2.4s, v3.4s        \n" \
    "	ld1r {v4.4s}, [x2]              \n" \
    "   add x2, x2, #16                 \n" \
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
    "	ld1r {v12.4s}, [x2]             \n" \
    "   add x2, x2, #16                 \n" \
    "	fdiv v12.4s, v2.4s, v12.4s      \n" \
    "	ld1r {v17.4s}, [x2]             \n" \
    "   add x2, x2, #16                 \n" \
    "	fdiv v17.4s, v2.4s, v17.4s      \n" \
    "	ld1r {v22.4s}, [x2]             \n" \
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
    "	add x20, x20, #64    			\n" \
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
    "	add x20, x20, #64    			\n"

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
    "	add x20, x20, #64    			\n"

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
    "	add x20, x20, #64				\n"

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
    "	add x20, x20, #64				\n"

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
    "	add x20, x20, #64				\n"

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
    "	add x20, x20, #64				\n"

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
    "	add x20, x20, #64				\n"

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
    "	add x20, x20, #64				\n"

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
    "	add x20, x20, #64				\n"

// #define gemm_ukernel_5x16_u8_last_k \
//     "	fmla v12.4s, v8.4s, v5.s[3]		\n" \
//     "	fmla v17.4s, v9.4s, v5.s[3]		\n" \
//     "	fmla v22.4s, v10.4s, v5.s[3]	\n" \
//     "	fmla v27.4s, v11.4s, v5.s[3]	\n" \
//     "	str q12, [x24]                  \n" \
//     "	str q17, [x24, #16]             \n" \
//     "	str q22, [x24, #32]             \n" \
//     "	str q27, [x24, #48]             \n" \
//     \
//     "	fmla v13.4s, v8.4s, v6.s[3]		\n" \
//     "	fmla v18.4s, v9.4s, v6.s[3]		\n" \
//     "	fmla v23.4s, v10.4s, v6.s[3]	\n" \
//     "	fmla v28.4s, v11.4s, v6.s[3]	\n" \
//     "	str q13, [x25]                  \n" \
//     "	str q18, [x25, #16]             \n" \
//     "	str q23, [x25, #32]             \n" \
//     "	str q28, [x25, #48]             \n" \
//     \
//     "	fmla v14.4s, v8.4s, v7.s[3]		\n" \
//     "	fmla v19.4s, v9.4s, v7.s[3]		\n" \
//     "	fmla v24.4s, v10.4s, v7.s[3]	\n" \
//     "	fmla v29.4s, v11.4s, v7.s[3]	\n" \
//     "	str q14, [x26]                  \n" \
//     "	str q19, [x26, #16]             \n" \
//     "	str q24, [x26, #32]             \n" \
//     "	str q29, [x26, #48]             \n" \
//     \
//     "	fmla v15.4s, v8.4s, v0.s[3]		\n" \
//     "	fmla v20.4s, v9.4s, v0.s[3]		\n" \
//     "	fmla v25.4s, v10.4s, v0.s[3]	\n" \
//     "	fmla v30.4s, v11.4s, v0.s[3]	\n" \
//     "	str q15, [x27]                  \n" \
//     "	str q20, [x27, #16]             \n" \
//     "	str q25, [x27, #32]             \n" \
//     "	str q30, [x27, #48]             \n" \
//     \
//     "	fmla v16.4s, v8.4s, v1.s[3]		\n" \
//     "	fmla v21.4s, v9.4s, v1.s[3]		\n" \
//     "	fmla v26.4s, v10.4s, v1.s[3]	\n" \
//     "	fmla v31.4s, v11.4s, v1.s[3]	\n" \
//     "	str q16, [x28]                  \n" \
//     "	str q21, [x28, #16]             \n" \
//     "	str q26, [x28, #32]             \n" \
//     "	str q31, [x28, #48]             \n"

// #define gemm_ukernel_5x16_u8_last_k \
//     "	fmov v2.4s, #1.0                \n" \
//     \
//     "	ldr q3, [x2]                    \n" \
//     "	fdiv v3.4s, v2.4s, v3.4s        \n" \
//     "	ldr q4, [x2, #16]               \n" \
//     "	fdiv v4.4s, v2.4s, v4.4s        \n" \
//     \
//     "	fmla v12.4s, v8.4s, v5.s[3]		\n" \
//     "	fmla v17.4s, v9.4s, v5.s[3]		\n" \
//     "	fmla v22.4s, v10.4s, v5.s[3]	\n" \
//     "	fmla v27.4s, v11.4s, v5.s[3]	\n" \
//     "	fmul v12.4s, v12.4s, v3.4s      \n" \
//     "	fmul v17.4s, v17.4s, v3.4s      \n" \
//     "	str q12, [x24]                  \n" \
//     "	str q17, [x24, #16]             \n" \
//     "	fmul v22.4s, v22.4s, v3.4s      \n" \
//     "	fmul v27.4s, v27.4s, v3.4s      \n" \
//     "	str q22, [x24, #32]             \n" \
//     "	str q27, [x24, #48]             \n" \
//     \
//     "	ldr q12, [x2, #32]              \n" \
//     "	fdiv v12.4s, v2.4s, v12.4s      \n" \
//     "	ldr q17, [x2, #48]              \n" \
//     "	fdiv v17.4s, v2.4s, v17.4s      \n" \
//     "	ldr q22, [x2, #64]              \n" \
//     "	fdiv v22.4s, v2.4s, v22.4s      \n" \
//     \
//     "	fmla v13.4s, v8.4s, v6.s[3]		\n" \
//     "	fmla v18.4s, v9.4s, v6.s[3]		\n" \
//     "	fmla v23.4s, v10.4s, v6.s[3]	\n" \
//     "	fmla v28.4s, v11.4s, v6.s[3]	\n" \
//     "	fmul v13.4s, v13.4s, v4.4s      \n" \
//     "	fmul v18.4s, v18.4s, v4.4s      \n" \
//     "	str q13, [x25]                  \n" \
//     "	str q18, [x25, #16]             \n" \
//     "	fmul v23.4s, v23.4s, v4.4s      \n" \
//     "	fmul v28.4s, v28.4s, v4.4s      \n" \
//     "	str q23, [x25, #32]             \n" \
//     "	str q28, [x25, #48]             \n" \
//     \
//     "	fmla v14.4s, v8.4s, v7.s[3]		\n" \
//     "	fmla v19.4s, v9.4s, v7.s[3]		\n" \
//     "	fmla v24.4s, v10.4s, v7.s[3]	\n" \
//     "	fmla v29.4s, v11.4s, v7.s[3]	\n" \
//     "	fmul v14.4s, v14.4s, v12.4s     \n" \
//     "	fmul v19.4s, v19.4s, v12.4s     \n" \
//     "	str q14, [x26]                  \n" \
//     "	str q19, [x26, #16]             \n" \
//     "	fmul v24.4s, v24.4s, v12.4s     \n" \
//     "	fmul v29.4s, v29.4s, v12.4s     \n" \
//     "	str q24, [x26, #32]             \n" \
//     "	str q29, [x26, #48]             \n" \
//     \
//     "	fmla v15.4s, v8.4s, v0.s[3]		\n" \
//     "	fmla v20.4s, v9.4s, v0.s[3]		\n" \
//     "	fmla v25.4s, v10.4s, v0.s[3]	\n" \
//     "	fmla v30.4s, v11.4s, v0.s[3]	\n" \
//     "	fmul v15.4s, v15.4s, v17.4s     \n" \
//     "	fmul v20.4s, v20.4s, v17.4s     \n" \
//     "	str q15, [x27]                  \n" \
//     "	str q20, [x27, #16]             \n" \
//     "	fmul v25.4s, v25.4s, v17.4s     \n" \
//     "	fmul v30.4s, v30.4s, v17.4s     \n" \
//     "	str q25, [x27, #32]             \n" \
//     "	str q30, [x27, #48]             \n" \
//     \
//     "	fmla v16.4s, v8.4s, v1.s[3]		\n" \
//     "	fmla v21.4s, v9.4s, v1.s[3]		\n" \
//     "	fmla v26.4s, v10.4s, v1.s[3]	\n" \
//     "	fmla v31.4s, v11.4s, v1.s[3]	\n" \
//     "	fmul v16.4s, v16.4s, v22.4s     \n" \
//     "	fmul v21.4s, v21.4s, v22.4s     \n" \
//     "	str q16, [x28]                  \n" \
//     "	str q21, [x28, #16]             \n" \
//     "	fmul v26.4s, v26.4s, v22.4s     \n" \
//     "	fmul v31.4s, v31.4s, v22.4s     \n" \
//     "	str q26, [x28, #32]             \n" \
//     "	str q31, [x28, #48]             \n"

#define gemm_ukernel_5x16_u8_last_k \
    "	fmov v2.4s, #1.0                \n" \
    \
    "	ld1r {v3.4s}, [x2]              \n" \
    "   add x2, x2, #16                 \n" \
    "	fdiv v3.4s, v2.4s, v3.4s        \n" \
    "	ld1r {v4.4s}, [x2]              \n" \
    "   add x2, x2, #16                 \n" \
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
    "	ld1r {v12.4s}, [x2]             \n" \
    "   add x2, x2, #16                 \n" \
    "	fdiv v12.4s, v2.4s, v12.4s      \n" \
    "	ld1r {v17.4s}, [x2]             \n" \
    "   add x2, x2, #16                 \n" \
    "	fdiv v17.4s, v2.4s, v17.4s      \n" \
    "	ld1r {v22.4s}, [x2]             \n" \
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

// exp(score-max), sum(score)
// out(mxn, row major) = score(mxk, panel(5x4)) * v(kxn, row major)
// score * (1/sum(score))
void fused_exp_sum_scorexv_kernel(long m, long n/*(head_dim==64)*/, long k,
                    float *score, float *v, float *buffer_v, float *out, long ldo,
                    float *max_4_per_line, float *exp_sum_per_line)
{
    int is, js;
    float *addr_a, *addr_b, *addr_c, *addr_max, *addr_sum;
    long ldb_in_bytes=sizeof(float)*n, ldc_in_bytes=sizeof(float)*ldo;

    for(js=0; js<n; js+=NR) {
        addr_a=score, addr_b=v+js, addr_c=out+js,
                addr_max=max_4_per_line, addr_sum=exp_sum_per_line;

        {
            // exp neon
            if(js == 0) {
                {
                    // exp_init
                    // for(int i=0; i<MR; i++) {
                    //     int ks=k/4,cnt=0;
                    //     v_max=vld1q_f32(addr_max+i*4);
                    //     v_max=vmovq_n_f32(vmaxvq_f32(v_max));

                    //     v_sum=vld1q_f32(addr_sum+i*4);

                    //     while(ks>0) {
                    //         vx=vld1q_f32(addr_a+cnt*20);
                    //         vx=vsubq_f32(vx, v_max);
                    //         exp_4
                    //         vst1q_f32(addr_a+cnt*20, vf);
                    //         v_sum=vaddq_f32(v_sum, vf);
                    //         ks--, cnt++;
                    //     }
                    //     v_sum=vmovq_n_f32(vaddvq_f32(v_sum));
                    //     vst1q_f32(addr_sum+i*4,v_sum);
                    //     addr_a+=4;
                    // }
                }
                
                {
                    meformer_exp_init
                    int batch;
                    float32x4_t vi_max;
                    float *input, *output, *sum;
                    for(int i=0; i<MR; i++) {
                        batch=sizeof(float)*k;
                        vi_max=vmovq_n_f32(vmaxvq_f32(vld1q_f32(addr_max+i*4)));
                        input=addr_a, output=addr_a;
                        sum=addr_sum+i*4;
                        meformer_minusmax_exp_sum(batch, input, vi_max, output, sum, 20);
                        addr_a+=4;
                    }
                }
            }
            addr_a=score;
        }

        {
            // pack m5xn16
            asm volatile (
                "pack_5x16_entry:				   \n"
                "	ldr x2, %[addr_sum]            \n"

                "	ldr x15, %[a]           	   \n"
                "	ldr x20, %[b]           	   \n"

                "	ldr x21, %[buffer_v]		   \n"
                "	ldr x6, %[ldb_in_bytes] 	   \n"

                "	ldr x5, %[ldc_in_bytes]        \n"
                "	ldr x24, %[c] 	        	   \n"
                "	add x25, x24, x5        	   \n"
                "	add x26, x25, x5        	   \n"
                "	add x27, x26, x5        	   \n"
                "	add x28, x27, x5        	   \n"
   
                "pack_5x16_start:           	   \n"
                "   ldr x3, %[k]		    	   \n"
                "	lsr x11, x3, #3         	   \n" // k/8
                    pack_ukernel_5x16_u8_first_k
                "	b pack_5x16_body               \n"

                "pack_5x16_mid:                   \n"
                    pack_ukernel_5x16_u8_k0

                "pack_5x16_body:				  \n"
                    pack_ukernel_5x16_u8_k1
                    pack_ukernel_5x16_u8_k2
                    pack_ukernel_5x16_u8_k3
                    pack_ukernel_5x16_u8_k4
                    pack_ukernel_5x16_u8_k5
                    pack_ukernel_5x16_u8_k6

                "	subs x11, x11, #1             \n"
                "	beq pack_5x16_end			  \n"
                    pack_ukernel_5x16_u8_k7
                "	b pack_5x16_mid               \n"

                "pack_5x16_end:					  \n"
                    pack_ukernel_5x16_u8_last_k

                : 
                : [a] "m" (addr_a),
                  [b] "m" (addr_b),
                  [c] "m" (addr_c),
                  [k] "m" (k),
                  [ldb_in_bytes] "m" (ldb_in_bytes),
                  [ldc_in_bytes] "m" (ldc_in_bytes),
                  [buffer_v] "m" (buffer_v),
                  [addr_sum] "m" (addr_sum)
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

        for(is=MR; is<m; is+=MR) {
            addr_a=(score+is*k), addr_c=(out+js+is*ldo),
                addr_max=(max_4_per_line+is*4), addr_sum=(exp_sum_per_line+is*4);

            {
                // exp neon
                if(js == 0) {
                    {
                        // exp_init
                        // for(int i=0; i<MR; i++) {
                        //     int ks=k/4,cnt=0;
                        //     v_max=vld1q_f32(addr_max+i*4);
                        //     v_max=vmovq_n_f32(vmaxvq_f32(v_max));

                        //     v_sum=vld1q_f32(addr_sum+i*4);

                        //     while(ks>0) {
                        //         vx=vld1q_f32(addr_a+cnt*20);
                        //         vx=vsubq_f32(vx, v_max);
                        //         exp_4
                        //         vst1q_f32(addr_a+cnt*20, vf);
                        //         v_sum=vaddq_f32(v_sum, vf);
                        //         ks--, cnt++;
                        //     }
                        //     v_sum=vmovq_n_f32(vaddvq_f32(v_sum));
                        //     vst1q_f32(addr_sum+i*4,v_sum);
                        //     addr_a+=4;
                        // }
                    }
                    
                    {
                        meformer_exp_init
                        int batch;
                        float32x4_t vi_max;
                        float *input, *output, *sum;
                        for(int i=0; i<MR; i++) {
                            batch=sizeof(float)*k;
                            vi_max=vmovq_n_f32(vmaxvq_f32(vld1q_f32(addr_max+i*4)));
                            input=addr_a, output=addr_a;
                            sum=addr_sum+i*4;
                            meformer_minusmax_exp_sum(batch, input, vi_max, output, sum, 20);
                            addr_a+=4;
                        }
                    }
                }
                addr_a=(score+is*k);
            }
            
            {
                // kernel
                asm volatile (
                    "gemm_5x16_entry:				   \n"
                    "	ldr x2, %[addr_sum]            \n"

                    "	ldr x15, %[a]           	   \n"
                    "	ldr x20, %[buffer_v]      	   \n"
                    
                    "	ldr x5, %[ldc_in_bytes]        \n"
                    "	ldr x24, %[c] 	        	   \n"
                    "	add x25, x24, x5        	   \n"
                    "	add x26, x25, x5        	   \n"
                    "	add x27, x26, x5        	   \n"
                    "	add x28, x27, x5        	   \n"
    
                    "gemm_5x16_start:           	   \n"
                    "   ldr x3, %[k]		    	   \n"
                    "	lsr x11, x3, #3         	   \n" // k/8
                        gemm_ukernel_5x16_u8_first_k
                    "	b gemm_5x16_body               \n"

                    "gemm_5x16_mid:                   \n"
                        gemm_ukernel_5x16_u8_k0

                    "gemm_5x16_body:				  \n"
                        gemm_ukernel_5x16_u8_k1
                        gemm_ukernel_5x16_u8_k2
                        gemm_ukernel_5x16_u8_k3
                        gemm_ukernel_5x16_u8_k4
                        gemm_ukernel_5x16_u8_k5
                        gemm_ukernel_5x16_u8_k6

                    "	subs x11, x11, #1             \n"
                    "	beq gemm_5x16_end			  \n"
                        gemm_ukernel_5x16_u8_k7
                    "	b gemm_5x16_mid               \n"

                    "gemm_5x16_end:					  \n"
                        gemm_ukernel_5x16_u8_last_k

                    : 
                    : [a] "m" (addr_a),
                      [b] "m" (addr_b),
                      [c] "m" (addr_c),
                      [k] "m" (k),
                      [ldb_in_bytes] "m" (ldb_in_bytes),
                      [ldc_in_bytes] "m" (ldc_in_bytes),
                      [buffer_v] "m" (buffer_v),
                      [addr_sum] "m" (addr_sum)
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
        }
    }
}

