#include "meformer.h"

#define pack_ukernel_5x4_u8_first_k                     \
    "   ldr q8, [x20], #16                          \n" \
    "   ldr q0, [x15], #16                          \n" \
    "   ldr q1, [x16], #16                          \n" \
    "   fmul v12.4s, v8.4s, v0.4s                   \n" \
    "   ldr q2, [x17], #16                          \n" \
    "   fmul v13.4s, v8.4s, v1.4s                   \n" \
    "   ldr q3, [x18], #16                          \n" \
    "   fmul v14.4s, v8.4s, v2.4s                   \n" \
    "   ldr q4, [x19], #16                          \n" \
    "   fmul v15.4s, v8.4s, v3.4s                   \n" \
    "   ldr q9, [x21], #16                          \n" \
    "   fmul v16.4s, v8.4s, v4.4s                   \n" \
    \
    "   ldr q10, [x22], #16                         \n" \
    "   fmul v17.4s, v9.4s, v0.4s                   \n" \
    "   ldr q11, [x23], #16                         \n" \
    "   fmul v18.4s, v9.4s, v1.4s                   \n" \
    "   fmul v19.4s, v9.4s, v2.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n" \
    "   add x12, x12, #64             			    \n" \
    "   fmul v20.4s, v9.4s, v3.4s     				\n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n" \
    "   add x12, x12, #64             			    \n" \
    "   fmul v21.4s, v9.4s, v4.4s     			    \n" \
    \
    "   ldr q5, [x15], #16                          \n" \
    "   fmul v22.4s, v10.4s, v0.4s                  \n" \
    "   fmul v23.4s, v10.4s, v1.4s                  \n" \
    "   ldr q6, [x16], #16                          \n" \
    "   fmul v24.4s, v10.4s, v2.4s                  \n" \
    "   fmul v25.4s, v10.4s, v3.4s                  \n" \
    "   ldr q7, [x17], #16                          \n" \
    "   fmul v26.4s, v10.4s, v4.4s                  \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    \
    "   fmul v27.4s, v11.4s, v0.4s    			    \n" \
    "   ldr q0, [x18], #16            			    \n" \
    "   fmul v28.4s, v11.4s, v1.4s    			    \n" \
    "   ldr q1, [x19], #16            			    \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   ldr q8, [x20], #16                          \n" \
    "   fmul v29.4s, v11.4s, v2.4s                  \n" \
    "   fmul v30.4s, v11.4s, v3.4s                  \n" \
    "   fmul v31.4s, v11.4s, v4.4s                  \n"

#define pack_ukernel_5x4_u8_k0                          \
    "   ldr q9, [x21], #16                          \n" \
    "   fmla v12.4s, v8.4s, v0.4s                   \n" \
    "   fmla v13.4s, v8.4s, v1.4s                   \n" \
    "   ldr q10, [x22], #16                         \n" \
    "   fmla v14.4s, v8.4s, v2.4s                   \n" \
    "   fmla v15.4s, v8.4s, v3.4s                   \n" \
    "   ldr q11, [x23], #16                         \n" \
    "   fmla v16.4s, v8.4s, v4.4s                   \n" \
    \
    "   ldr q5, [x15], #16                          \n" \
    "   fmla v17.4s, v9.4s, v0.4s                   \n" \
    "   fmla v18.4s, v9.4s, v1.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v19.4s, v9.4s, v2.4s                   \n" \
    "   fmla v20.4s, v9.4s, v3.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v21.4s, v9.4s, v4.4s                   \n" \
    \
    "   ldr q6, [x16], #16                          \n" \
    "   ldr q7, [x17], #16                          \n" \
    "   fmla v22.4s, v10.4s, v0.4s                  \n" \
    "   fmla v23.4s, v10.4s, v1.4s                  \n" \
    "   fmla v24.4s, v10.4s, v2.4s                  \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v25.4s, v10.4s, v3.4s                  \n" \
    "   fmla v26.4s, v10.4s, v4.4s                  \n" \
    \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   ldr q8, [x20], #16                          \n" \
    "   fmla v27.4s, v11.4s, v0.4s                  \n" \
    "   ldr q0, [x18], #16                          \n" \
    "   fmla v28.4s, v11.4s, v1.4s                  \n" \
    "   ldr q1, [x19], #16                          \n" \
    "   fmla v29.4s, v11.4s, v2.4s                  \n" \
    "   fmla v30.4s, v11.4s, v3.4s                  \n" \
    "   fmla v31.4s, v11.4s, v4.4s                  \n"

#define pack_ukernel_5x4_u8_k1                          \
    "   ldr q9, [x21], #16                          \n" \
    "   fmla v12.4s, v8.4s, v5.4s                   \n" \
    "   fmla v13.4s, v8.4s, v6.4s                   \n" \
    "   ldr q10, [x22], #16                         \n" \
    "   fmla v14.4s, v8.4s, v7.4s                   \n" \
    "   fmla v15.4s, v8.4s, v0.4s                   \n" \
    "   ldr q11, [x23], #16                         \n" \
    "   fmla v16.4s, v8.4s, v1.4s                   \n" \
    \
    "   ldr q2, [x17], #16                          \n" \
    "   fmla v17.4s, v9.4s, v5.4s                   \n" \
    "   fmla v18.4s, v9.4s, v6.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v19.4s, v9.4s, v7.4s                   \n" \
    "   fmla v20.4s, v9.4s, v0.4s                   \n" \
    "   fmla v21.4s, v9.4s, v1.4s                   \n" \
    \
    "   ldr q3, [x18], #16                          \n" \
    "   ldr q4, [x19], #16                          \n" \
    "   fmla v22.4s, v10.4s, v5.4s                  \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v23.4s, v10.4s, v6.4s                  \n" \
    "   fmla v24.4s, v10.4s, v7.4s                  \n" \
    "   fmla v25.4s, v10.4s, v0.4s                  \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v26.4s, v10.4s, v1.4s                  \n" \
    \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   ldr q8, [x20], #16                          \n" \
    "   fmla v30.4s, v11.4s, v0.4s                  \n" \
    "   ldr q0, [x15], #16                          \n" \
    "   fmla v31.4s, v11.4s, v1.4s                  \n" \
    "   ldr q1, [x16], #16                          \n" \
    "   fmla v27.4s, v11.4s, v5.4s                  \n" \
    "   fmla v28.4s, v11.4s, v6.4s                  \n" \
    "   fmla v29.4s, v11.4s, v7.4s                  \n"

#define pack_ukernel_5x4_u8_last_k                      \
	/*x9:max  x10:mask*/\
    "   prfm PLDL1STRM, [x9]                        \n" \
	\
    "   prfm PLDL1KEEP, [x10]                       \n" \
    "	prfm PLDL1KEEP, [x10, x11]                  \n" \
    \
    "   ldr q9, [x21], #16                          \n" \
    "   fmla v12.4s, v8.4s, v5.4s                   \n" \
    "   fmla v13.4s, v8.4s, v6.4s                   \n" \
    "   ldr q10, [x22], #16                         \n" \
    "   fmla v14.4s, v8.4s, v7.4s                   \n" \
    "   fmla v15.4s, v8.4s, v0.4s                   \n" \
    "   ldr q11, [x23], #16                         \n" \
    "   fmla v16.4s, v8.4s, v1.4s                   \n" \
    \
    "   ldr q3, [x9]                                \n" \
    "   fmla v17.4s, v9.4s, v5.4s                   \n" \
    "   fmla v18.4s, v9.4s, v6.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v19.4s, v9.4s, v7.4s                   \n" \
    "   fmla v20.4s, v9.4s, v0.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v21.4s, v9.4s, v1.4s                   \n" \
    \
    "   ld1r {v2.4s}, [x6]                          \n" \
    "   faddp v12.4s, v12.4s, v17.4s                \n" \
    "   ldr q17, [x9, #16]                          \n" \
    "   faddp v13.4s, v13.4s, v18.4s                \n" \
    "	ldr q18, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "   faddp v14.4s, v14.4s, v19.4s                \n" \
    "	ldr q19, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "   faddp v15.4s, v15.4s, v20.4s                \n" \
    "   prfm PLDL1KEEP, [x10]                       \n" \
    "	prfm PLDL1KEEP, [x10, x11]                  \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   faddp v16.4s, v16.4s, v21.4s                \n" \
    \
    "   fmla v22.4s, v10.4s, v5.4s                  \n" \
    "   fmla v23.4s, v10.4s, v6.4s                  \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v24.4s, v10.4s, v7.4s                  \n" \
    "   fmla v25.4s, v10.4s, v0.4s                  \n" \
    "	ldr q20, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "	ldr q21, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "   fmla v26.4s, v10.4s, v1.4s                  \n" \
    \
    "   ldr q4, [x9, #32]                           \n" \
    "   fmla v30.4s, v11.4s, v0.4s                  \n" \
    "   fmla v31.4s, v11.4s, v1.4s                  \n" \
    "   ldr q0, [x9, #48]                           \n" \
    "   ldr q1, [x9, #64]                           \n" \
    "   fmla v27.4s, v11.4s, v5.4s                  \n" \
    "   fmla v28.4s, v11.4s, v6.4s                  \n" \
    "   fmla v29.4s, v11.4s, v7.4s                  \n" \
    \
    "   faddp v22.4s, v22.4s, v27.4s                \n" \
    "   faddp v23.4s, v23.4s, v28.4s                \n" \
    "   faddp v12.4s, v12.4s, v22.4s                \n" \
    "	ldr q22, [x10]                              \n" \
    "   faddp v13.4s, v13.4s, v23.4s                \n" \
    "   fmul v12.4s, v12.4s, v2.s[0]                \n" \
    "   fmul v13.4s, v13.4s, v2.s[0]                \n" \
    "	fadd v12.4s, v12.4s, v18.4s                 \n" \
    "	fadd v13.4s, v13.4s, v19.4s                 \n" \
    "   str q12, [x24]                              \n" \
    "   add x24, x24, #80                           \n" \
    "   str q13, [x25]                              \n" \
    "   add x25, x25, #80                           \n" \
    "   fmax v3.4s, v3.4s, v12.4s                   \n" \
    "   fmax v17.4s, v17.4s, v13.4s                 \n" \
    \
    "   faddp v24.4s, v24.4s, v29.4s                \n" \
    "   faddp v25.4s, v25.4s, v30.4s                \n" \
    "   str q3, [x9]                                \n" \
    "   str q17, [x9, #16]                          \n" \
    "   faddp v14.4s, v14.4s, v24.4s                \n" \
    "   faddp v15.4s, v15.4s, v25.4s                \n" \
    "   fmul v14.4s, v14.4s, v2.s[0]                \n" \
    "   fmul v15.4s, v15.4s, v2.s[0]                \n" \
    "	fadd v14.4s, v14.4s, v20.4s                 \n" \
    "	fadd v15.4s, v15.4s, v21.4s                 \n" \
    "   str q14, [x26]                              \n" \
    "   add x26, x26, #80                           \n" \
    "   str q15, [x27]                              \n" \
    "   add x27, x27, #80                           \n" \
    "   fmax v4.4s, v4.4s, v14.4s                   \n" \
    "   fmax v0.4s, v0.4s, v15.4s                   \n" \
    \
    "   faddp v26.4s, v26.4s, v31.4s                \n" \
    "   faddp v16.4s, v16.4s, v26.4s                \n" \
    "   str q4, [x9, #32]                           \n" \
    "   str q0, [x9, #48]                           \n" \
    "   fmul v16.4s, v16.4s, v2.s[0]                \n" \
    "	fadd v16.4s, v16.4s, v22.4s                 \n" \
    "   str q16, [x28]                              \n" \
    "   add x28, x28, #80                           \n" \
    "   fmax v1.4s, v1.4s, v16.4s                   \n" \
    "   str q1, [x9, #64]                           \n"

#define pack_ukernel_5x4_u8_last_k_with_scale_mask_max  \
	"	prfm PLDL1STRM, [x24]                       \n" \
	"	prfm PLDL1STRM, [x25]                       \n" \
	"	prfm PLDL1STRM, [x26]                       \n" \
	"	prfm PLDL1STRM, [x27]                       \n" \
	"	prfm PLDL1STRM, [x28]                       \n" \
	\
	/*x6:scale  x9:max  x10:mask  x12:pack*/\
    "   prfm PLDL1STRM, [x9]                        \n" \
	\
    "   prfm PLDL1STRM, [x10]                       \n" \
    "	prfm PLDL1STRM, [x10, x11]                  \n" \
    \
	\
    "   ldr q9, [x21], #16                          \n" \
	"   fmla v12.4s, v8.4s, v5.4s                   \n"/*col 1*/\
    "   fmla v13.4s, v8.4s, v6.4s                   \n" \
    "   ldr q10, [x22], #16                         \n" \
	"   ldr q11, [x23], #16                         \n" \
	"   fmla v14.4s, v8.4s, v7.4s                   \n" \
    "   fmla v15.4s, v8.4s, v0.4s                   \n" \
    "   fmla v16.4s, v8.4s, v1.4s                   \n" \
	"   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    \
    "   ld1r {v2.4s}, [x6]                          \n" \
	"   ld1r {v3.4s}, [x9]                          \n" \
	"	add x9, x9, #4                              \n" \
    "   fmla v17.4s, v9.4s, v5.4s                   \n"/*col 2*/\
    "   fmla v18.4s, v9.4s, v6.4s                   \n" \
    "   fmla v19.4s, v9.4s, v7.4s                   \n" \
    "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
	"   fmla v20.4s, v9.4s, v0.4s                   \n" \
    "   fmla v21.4s, v9.4s, v1.4s                   \n" \
    \
    "   faddp v12.4s, v12.4s, v17.4s                \n" \
    "   ld1r {v17.4s}, [x9]                         \n" \
	"	add x9, x9, #4                              \n" \
    "   faddp v13.4s, v13.4s, v18.4s                \n" \
    "	ldr q18, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "   faddp v14.4s, v14.4s, v19.4s                \n" \
    "	ldr q19, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "   prfm PLDL1STRM, [x10]                       \n" \
    "	prfm PLDL1STRM, [x10, x11]                  \n" \
	"   faddp v15.4s, v15.4s, v20.4s                \n" \
    "   faddp v16.4s, v16.4s, v21.4s                \n" \
	"	ldr q20, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
    "	ldr q21, [x10]                              \n" \
    "	add x10, x10, x11                           \n" \
	"	prfm PLDL1STRM, [x10]                       \n" \
    \
    "   fmla v22.4s, v10.4s, v5.4s                  \n"/*col 3*/\
    "   fmla v23.4s, v10.4s, v6.4s                  \n" \
	"   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    "   fmla v24.4s, v10.4s, v7.4s                  \n" \
    "   fmla v25.4s, v10.4s, v0.4s                  \n" \
	"   fmla v26.4s, v10.4s, v1.4s                  \n" \
	"   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n" \
    "   add x12, x12, #64                           \n" \
    \
    "   ld1r {v4.4s}, [x9]                          \n" \
	"	add x9, x9, #4                              \n" \
    "   fmla v30.4s, v11.4s, v0.4s                  \n"/*col 4*/\
    "   fmla v31.4s, v11.4s, v1.4s                  \n" \
    "   ld1r {v0.4s}, [x9]                          \n" \
	"	add x9, x9, #4                              \n" \
    "   ld1r {v1.4s}, [x9]                          \n" \
	"	sub x9, x9, #16                             \n" \
    "   fmla v27.4s, v11.4s, v5.4s                  \n" \
    "   fmla v28.4s, v11.4s, v6.4s                  \n" \
    "   fmla v29.4s, v11.4s, v7.4s                  \n" \
    \
    "   faddp v22.4s, v22.4s, v27.4s                \n" \
    "   faddp v23.4s, v23.4s, v28.4s                \n" \
    "   faddp v12.4s, v12.4s, v22.4s                \n" \
    "	ldr q22, [x10]                              \n" \
    "   faddp v13.4s, v13.4s, v23.4s                \n" \
    "   fmul v12.4s, v12.4s, v2.s[0]                \n" \
    "   fmul v13.4s, v13.4s, v2.s[0]                \n" \
    "	fadd v12.4s, v12.4s, v18.4s                 \n" \
    "	fadd v13.4s, v13.4s, v19.4s                 \n" \
    "   str q12, [x24]                              \n" \
    "   add x24, x24, #80                           \n" \
    "   str q13, [x25]                              \n" \
    "   add x25, x25, #80                           \n" \
    "   fmax v3.4s, v3.4s, v12.4s                   \n" \
    "   fmax v17.4s, v17.4s, v13.4s                 \n" \
	"	fmaxv s3, v3.4s                             \n" \
	"	fmaxv s17, v17.4s                           \n" \
    "   str s3, [x9]                                \n" \
    "   str s17, [x9, #4]                           \n" \
	\
    "   faddp v24.4s, v24.4s, v29.4s                \n" \
    "   faddp v25.4s, v25.4s, v30.4s                \n" \
    "   faddp v14.4s, v14.4s, v24.4s                \n" \
    "   faddp v15.4s, v15.4s, v25.4s                \n" \
    "   fmul v14.4s, v14.4s, v2.s[0]                \n" \
    "   fmul v15.4s, v15.4s, v2.s[0]                \n" \
    "	fadd v14.4s, v14.4s, v20.4s                 \n" \
    "	fadd v15.4s, v15.4s, v21.4s                 \n" \
    "   str q14, [x26]                              \n" \
    "   add x26, x26, #80                           \n" \
    "   str q15, [x27]                              \n" \
    "   add x27, x27, #80                           \n" \
    "   fmax v4.4s, v4.4s, v14.4s                   \n" \
    "   fmax v0.4s, v0.4s, v15.4s                   \n" \
	"	fmaxv s4, v4.4s                             \n" \
	"	fmaxv s0, v0.4s                             \n" \
    "   str s4, [x9, #8]                            \n" \
    "   str s0, [x9, #12]                           \n" \
	\
    "   faddp v26.4s, v26.4s, v31.4s                \n" \
    "   faddp v16.4s, v16.4s, v26.4s                \n" \
    "   fmul v16.4s, v16.4s, v2.s[0]                \n" \
    "	fadd v16.4s, v16.4s, v22.4s                 \n" \
    "   str q16, [x28]                              \n" \
    "   add x28, x28, #80                           \n" \
    "   fmax v1.4s, v1.4s, v16.4s                   \n" \
	"	fmaxv s1, v1.4s                             \n" \
    "   str s1, [x9, #16]                           \n"

#define gemm_ukernel_5x16_u8_first_k        \
    "   ldp q8, q9, [x20], #32          \n" \
    "   ldr q0, [x15], #16              \n" \
    "   ldr q1, [x16], #16              \n" \
    \
    "   fmul v12.4s, v8.4s, v0.s[0]     \n" \
    "   fmul v17.4s, v9.4s, v0.s[0]     \n" \
    "   ldr q2, [x17], #16              \n" \
    "   fmul v13.4s, v8.4s, v1.s[0]     \n" \
    "   fmul v18.4s, v9.4s, v1.s[0]     \n" \
    "   ldr q3, [x18], #16              \n" \
    "   fmul v14.4s, v8.4s, v2.s[0]     \n" \
    "   fmul v19.4s, v9.4s, v2.s[0]     \n" \
    "   ldr q4, [x19], #16              \n" \
    "   fmul v15.4s, v8.4s, v3.s[0]     \n" \
    "   ldp q10, q11, [x20], #32        \n" \
    "   fmul v20.4s, v9.4s, v3.s[0]     \n" \
    "   fmul v16.4s, v8.4s, v4.s[0]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmul v21.4s, v9.4s, v4.s[0]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmul v22.4s, v10.4s, v0.s[0]    \n" \
    "   fmul v27.4s, v11.4s, v0.s[0]    \n" \
    "   fmul v23.4s, v10.4s, v1.s[0]    \n" \
    "   fmul v28.4s, v11.4s, v1.s[0]    \n" \
    "   ldr q5, [x15], #16              \n" \
    "   fmul v24.4s, v10.4s, v2.s[0]    \n" \
    "   fmul v29.4s, v11.4s, v2.s[0]    \n" \
    "   fmul v25.4s, v10.4s, v3.s[0]    \n" \
    "   fmul v30.4s, v11.4s, v3.s[0]    \n" \
    "   fmul v26.4s, v10.4s, v4.s[0]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmul v31.4s, v11.4s, v4.s[0]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k0             \
    "   fmla v12.4s, v8.4s, v0.s[0]     \n" \
    "   fmla v17.4s, v9.4s, v0.s[0]     \n" \
    "   fmla v13.4s, v8.4s, v1.s[0]     \n" \
    "   fmla v18.4s, v9.4s, v1.s[0]     \n" \
    "   fmla v14.4s, v8.4s, v2.s[0]     \n" \
    "   fmla v19.4s, v9.4s, v2.s[0]     \n" \
    "   fmla v15.4s, v8.4s, v3.s[0]     \n" \
    "   fmla v20.4s, v9.4s, v3.s[0]     \n" \
    "   fmla v16.4s, v8.4s, v4.s[0]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v4.s[0]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v0.s[0]    \n" \
    "   fmla v27.4s, v11.4s, v0.s[0]    \n" \
    "   fmla v23.4s, v10.4s, v1.s[0]    \n" \
    "   fmla v28.4s, v11.4s, v1.s[0]    \n" \
    "   ldr q5, [x15], #16              \n" \
    "   fmla v24.4s, v10.4s, v2.s[0]    \n" \
    "   fmla v29.4s, v11.4s, v2.s[0]    \n" \
    "   fmla v25.4s, v10.4s, v3.s[0]    \n" \
    "   fmla v30.4s, v11.4s, v3.s[0]    \n" \
    "   fmla v26.4s, v10.4s, v4.s[0]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v4.s[0]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k1             \
    "   fmla v12.4s, v8.4s, v0.s[1]     \n" \
    "   fmla v17.4s, v9.4s, v0.s[1]     \n" \
    "   fmla v13.4s, v8.4s, v1.s[1]     \n" \
    "   fmla v18.4s, v9.4s, v1.s[1]     \n" \
    "   fmla v14.4s, v8.4s, v2.s[1]     \n" \
    "   fmla v19.4s, v9.4s, v2.s[1]     \n" \
    "   fmla v15.4s, v8.4s, v3.s[1]     \n" \
    "   fmla v20.4s, v9.4s, v3.s[1]     \n" \
    "   fmla v16.4s, v8.4s, v4.s[1]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v4.s[1]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v0.s[1]    \n" \
    "   fmla v27.4s, v11.4s, v0.s[1]    \n" \
    "   fmla v23.4s, v10.4s, v1.s[1]    \n" \
    "   fmla v28.4s, v11.4s, v1.s[1]    \n" \
    "   ldr q6, [x16], #16              \n" \
    "   fmla v24.4s, v10.4s, v2.s[1]    \n" \
    "   fmla v29.4s, v11.4s, v2.s[1]    \n" \
    "   fmla v25.4s, v10.4s, v3.s[1]    \n" \
    "   fmla v30.4s, v11.4s, v3.s[1]    \n" \
    "   fmla v26.4s, v10.4s, v4.s[1]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v4.s[1]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k2             \
    "   fmla v12.4s, v8.4s, v0.s[2]     \n" \
    "   fmla v17.4s, v9.4s, v0.s[2]     \n" \
    "   fmla v13.4s, v8.4s, v1.s[2]     \n" \
    "   fmla v18.4s, v9.4s, v1.s[2]     \n" \
    "   fmla v14.4s, v8.4s, v2.s[2]     \n" \
    "   fmla v19.4s, v9.4s, v2.s[2]     \n" \
    "   fmla v15.4s, v8.4s, v3.s[2]     \n" \
    "   fmla v20.4s, v9.4s, v3.s[2]     \n" \
    "   fmla v16.4s, v8.4s, v4.s[2]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v4.s[2]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v0.s[2]    \n" \
    "   fmla v27.4s, v11.4s, v0.s[2]    \n" \
    "   fmla v23.4s, v10.4s, v1.s[2]    \n" \
    "   fmla v28.4s, v11.4s, v1.s[2]    \n" \
    "   ldr q7, [x17], #16              \n" \
    "   fmla v24.4s, v10.4s, v2.s[2]    \n" \
    "   fmla v29.4s, v11.4s, v2.s[2]    \n" \
    "   fmla v25.4s, v10.4s, v3.s[2]    \n" \
    "   fmla v30.4s, v11.4s, v3.s[2]    \n" \
    "   fmla v26.4s, v10.4s, v4.s[2]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v4.s[2]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k3             \
    "   fmla v12.4s, v8.4s, v0.s[3]     \n" \
    "   fmla v17.4s, v9.4s, v0.s[3]     \n" \
    "   fmla v13.4s, v8.4s, v1.s[3]     \n" \
    "   fmla v18.4s, v9.4s, v1.s[3]     \n" \
    "   fmla v14.4s, v8.4s, v2.s[3]     \n" \
    "   fmla v19.4s, v9.4s, v2.s[3]     \n" \
    "   fmla v15.4s, v8.4s, v3.s[3]     \n" \
    "   fmla v20.4s, v9.4s, v3.s[3]     \n" \
    "   fmla v16.4s, v8.4s, v4.s[3]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v4.s[3]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v0.s[3]    \n" \
    "   fmla v27.4s, v11.4s, v0.s[3]    \n" \
    "   ldr q0, [x18], #16              \n" \
    "   fmla v23.4s, v10.4s, v1.s[3]    \n" \
    "   fmla v28.4s, v11.4s, v1.s[3]    \n" \
    "   ldr q1, [x19], #16              \n" \
    "   fmla v24.4s, v10.4s, v2.s[3]    \n" \
    "   fmla v29.4s, v11.4s, v2.s[3]    \n" \
    "   fmla v25.4s, v10.4s, v3.s[3]    \n" \
    "   fmla v30.4s, v11.4s, v3.s[3]    \n" \
    "   fmla v26.4s, v10.4s, v4.s[3]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v4.s[3]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k4             \
    "   fmla v12.4s, v8.4s, v5.s[0]     \n" \
    "   fmla v17.4s, v9.4s, v5.s[0]     \n" \
    "   fmla v13.4s, v8.4s, v6.s[0]     \n" \
    "   fmla v18.4s, v9.4s, v6.s[0]     \n" \
    "   fmla v14.4s, v8.4s, v7.s[0]     \n" \
    "   fmla v19.4s, v9.4s, v7.s[0]     \n" \
    "   fmla v15.4s, v8.4s, v0.s[0]     \n" \
    "   fmla v20.4s, v9.4s, v0.s[0]     \n" \
    "   fmla v16.4s, v8.4s, v1.s[0]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v1.s[0]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v5.s[0]    \n" \
    "   fmla v27.4s, v11.4s, v5.s[0]    \n" \
    "   fmla v23.4s, v10.4s, v6.s[0]    \n" \
    "   fmla v28.4s, v11.4s, v6.s[0]    \n" \
    "   ldr q2, [x17], #16              \n" \
    "   fmla v24.4s, v10.4s, v7.s[0]    \n" \
    "   fmla v29.4s, v11.4s, v7.s[0]    \n" \
    "   fmla v25.4s, v10.4s, v0.s[0]    \n" \
    "   fmla v30.4s, v11.4s, v0.s[0]    \n" \
    "   fmla v26.4s, v10.4s, v1.s[0]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v1.s[0]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k5             \
    "   fmla v12.4s, v8.4s, v5.s[1]     \n" \
    "   fmla v17.4s, v9.4s, v5.s[1]     \n" \
    "   fmla v13.4s, v8.4s, v6.s[1]     \n" \
    "   fmla v18.4s, v9.4s, v6.s[1]     \n" \
    "   fmla v14.4s, v8.4s, v7.s[1]     \n" \
    "   fmla v19.4s, v9.4s, v7.s[1]     \n" \
    "   fmla v15.4s, v8.4s, v0.s[1]     \n" \
    "   fmla v20.4s, v9.4s, v0.s[1]     \n" \
    "   fmla v16.4s, v8.4s, v1.s[1]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v1.s[1]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v5.s[1]    \n" \
    "   fmla v27.4s, v11.4s, v5.s[1]    \n" \
    "   fmla v23.4s, v10.4s, v6.s[1]    \n" \
    "   fmla v28.4s, v11.4s, v6.s[1]    \n" \
    "   ldr q3, [x18], #16              \n" \
    "   fmla v24.4s, v10.4s, v7.s[1]    \n" \
    "   fmla v29.4s, v11.4s, v7.s[1]    \n" \
    "   fmla v25.4s, v10.4s, v0.s[1]    \n" \
    "   fmla v30.4s, v11.4s, v0.s[1]    \n" \
    "   fmla v26.4s, v10.4s, v1.s[1]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v1.s[1]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k6             \
    "   fmla v12.4s, v8.4s, v5.s[2]     \n" \
    "   fmla v17.4s, v9.4s, v5.s[2]     \n" \
    "   fmla v13.4s, v8.4s, v6.s[2]     \n" \
    "   fmla v18.4s, v9.4s, v6.s[2]     \n" \
    "   fmla v14.4s, v8.4s, v7.s[2]     \n" \
    "   fmla v19.4s, v9.4s, v7.s[2]     \n" \
    "   fmla v15.4s, v8.4s, v0.s[2]     \n" \
    "   fmla v20.4s, v9.4s, v0.s[2]     \n" \
    "   fmla v16.4s, v8.4s, v1.s[2]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v1.s[2]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v22.4s, v10.4s, v5.s[2]    \n" \
    "   fmla v27.4s, v11.4s, v5.s[2]    \n" \
    "   fmla v23.4s, v10.4s, v6.s[2]    \n" \
    "   fmla v28.4s, v11.4s, v6.s[2]    \n" \
    "   ldr q4, [x19], #16              \n" \
    "   fmla v24.4s, v10.4s, v7.s[2]    \n" \
    "   fmla v29.4s, v11.4s, v7.s[2]    \n" \
    "   fmla v25.4s, v10.4s, v0.s[2]    \n" \
    "   fmla v30.4s, v11.4s, v0.s[2]    \n" \
    "   fmla v26.4s, v10.4s, v1.s[2]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v31.4s, v11.4s, v1.s[2]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_k7             \
    "   fmla v12.4s, v8.4s, v5.s[3]     \n" \
    "   fmla v17.4s, v9.4s, v5.s[3]     \n" \
    "   fmla v13.4s, v8.4s, v6.s[3]     \n" \
    "   fmla v18.4s, v9.4s, v6.s[3]     \n" \
    "   fmla v14.4s, v8.4s, v7.s[3]     \n" \
    "   fmla v19.4s, v9.4s, v7.s[3]     \n" \
    "   fmla v15.4s, v8.4s, v0.s[3]     \n" \
    "   fmla v20.4s, v9.4s, v0.s[3]     \n" \
    "   fmla v16.4s, v8.4s, v1.s[3]     \n" \
    "   ldr q8, [x20], #16              \n" \
    "   fmla v21.4s, v9.4s, v1.s[3]     \n" \
    "   ldr q9, [x20], #16              \n" \
    \
    "   fmla v25.4s, v10.4s, v0.s[3]    \n" \
    "   fmla v30.4s, v11.4s, v0.s[3]    \n" \
    "   ldr q0, [x15], #16              \n" \
    "   fmla v26.4s, v10.4s, v1.s[3]    \n" \
    "   fmla v31.4s, v11.4s, v1.s[3]    \n" \
    "   ldr q1, [x16], #16              \n" \
    "   fmla v22.4s, v10.4s, v5.s[3]    \n" \
    "   fmla v27.4s, v11.4s, v5.s[3]    \n" \
    "   fmla v23.4s, v10.4s, v6.s[3]    \n" \
    "   fmla v28.4s, v11.4s, v6.s[3]    \n" \
    "   fmla v24.4s, v10.4s, v7.s[3]    \n" \
    "   ldr q10, [x20], #16             \n" \
    "   fmla v29.4s, v11.4s, v7.s[3]    \n" \
    "   ldr q11, [x20], #16             \n"

#define gemm_ukernel_5x16_u8_last_k         \
    "	prfm PLDL1STRM, [x9]            \n" \
    \
    "   ld1r {v2.4s}, [x6]              \n" \
    \
    "	fmla v12.4s, v8.4s, v5.s[3]     \n" \
    "	fmla v17.4s, v9.4s, v5.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v22.4s, v10.4s, v5.s[3]    \n" \
    "	fmla v27.4s, v11.4s, v5.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    \
    "	fmul v12.4s, v12.4s, v2.s[0]    \n" \
    "	fmul v17.4s, v17.4s, v2.s[0]    \n" \
    "	fmul v22.4s, v22.4s, v2.s[0]    \n" \
    "	fmul v27.4s, v27.4s, v2.s[0]    \n" \
    \
    "	fadd v12.4s, v12.4s, v3.4s      \n" \
    "	ldr q3, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
    "	fadd v17.4s, v17.4s, v4.4s      \n" \
    "	ldr q4, [x9]                    \n" \
    "	fadd v22.4s, v22.4s, v5.4s      \n" \
    "	fadd v27.4s, v27.4s, v3.4s      \n" \
    "	str q12, [x24]                  \n" \
    "   add x24, x24, #80               \n" \
    "	str q17, [x24]                  \n" \
    "   add x24, x24, #80               \n" \
    \
    "	fmax v12.4s, v12.4s, v17.4s     \n" \
    "	str q22, [x24]                  \n" \
    "   add x24, x24, #80               \n" \
    "	str q27, [x24]                  \n" \
    "	fmax v22.4s, v22.4s, v27.4s     \n" \
    "	fmax v12.4s, v12.4s, v4.4s      \n" \
    "	fmax v12.4s, v12.4s, v22.4s     \n" \
    "	str q12, [x9], #16              \n" \
    \
    "	fmla v13.4s, v8.4s, v6.s[3]     \n" \
    "	fmla v18.4s, v9.4s, v6.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v23.4s, v10.4s, v6.s[3]    \n" \
    "	fmla v28.4s, v11.4s, v6.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
    \
    "	fmul v13.4s, v13.4s, v2.s[0]    \n" \
    "	fmul v18.4s, v18.4s, v2.s[0]    \n" \
    "	fmul v23.4s, v23.4s, v2.s[0]    \n" \
    "	fmul v28.4s, v28.4s, v2.s[0]    \n" \
    \
    "	fadd v13.4s, v13.4s, v3.4s      \n" \
    "	ldr q3, [x9]                    \n" \
    "	fadd v18.4s, v18.4s, v4.4s      \n" \
    "	fadd v23.4s, v23.4s, v5.4s      \n" \
    "	fadd v28.4s, v28.4s, v6.4s      \n" \
    "	str q13, [x25]                  \n" \
    "   add x25, x25, #80               \n" \
    "	str q18, [x25]                  \n" \
    "   add x25, x25, #80               \n" \
    \
    "	fmax v13.4s, v13.4s, v18.4s     \n" \
    "	str q23, [x25]                  \n" \
    "   add x25, x25, #80               \n" \
    "	str q28, [x25]                  \n" \
    "	fmax v23.4s, v23.4s, v28.4s     \n" \
    "	fmax v13.4s, v13.4s, v3.4s      \n" \
    "	fmax v13.4s, v13.4s, v23.4s     \n" \
    "	str q13, [x9], #16              \n" \
    \
    "   fmla v14.4s, v8.4s, v7.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "   fmla v19.4s, v9.4s , v7.s[3]    \n" \
    "   fmla v24.4s, v10.4s, v7.s[3]    \n" \
    "   fmla v29.4s, v11.4s, v7.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
    \
    "	fmul v14.4s, v14.4s, v2.s[0]    \n" \
    "	fmul v19.4s, v19.4s, v2.s[0]    \n" \
    "	fmul v24.4s, v24.4s, v2.s[0]    \n" \
    "	fmul v29.4s, v29.4s, v2.s[0]    \n" \
    \
    "	fadd v14.4s, v14.4s, v3.4s      \n" \
    "	ldr q3, [x9]                    \n" \
    "	fadd v19.4s, v19.4s, v4.4s      \n" \
    "	fadd v24.4s, v24.4s, v5.4s      \n" \
    "	fadd v29.4s, v29.4s, v6.4s      \n" \
    "	str q14, [x26]                  \n" \
    "   add x26, x26, #80               \n" \
    "	str q19, [x26]                  \n" \
    "   add x26, x26, #80               \n" \
    \
    "	fmax v14.4s, v14.4s, v19.4s     \n" \
    "	str q24, [x26]                  \n" \
    "   add x26, x26, #80               \n" \
    "	str q29, [x26]                  \n" \
    "	fmax v24.4s, v24.4s, v29.4s     \n" \
    "	fmax v14.4s, v14.4s, v3.4s      \n" \
    "	fmax v14.4s, v14.4s, v24.4s     \n" \
    "	str q14, [x9], #16              \n" \
    \
    "	fmla v15.4s, v8.4s, v0.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v20.4s, v9.4s , v0.s[3]    \n" \
    "	fmla v25.4s, v10.4s, v0.s[3]    \n" \
    "	fmla v30.4s, v11.4s, v0.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
    \
    "	fmul v15.4s, v15.4s, v2.s[0]    \n" \
    "	fmul v20.4s, v20.4s, v2.s[0]    \n" \
    "	fmul v25.4s, v25.4s, v2.s[0]    \n" \
    "	fmul v30.4s, v30.4s, v2.s[0]    \n" \
    \
    "	fadd v15.4s, v15.4s, v3.4s      \n" \
    "	ldr q3, [x9]                    \n" \
    "	fadd v20.4s, v20.4s, v4.4s      \n" \
    "	fadd v25.4s, v25.4s, v5.4s      \n" \
    "	fadd v30.4s, v30.4s, v6.4s      \n" \
    "	str q15, [x27]                  \n" \
    "   add x27, x27, #80               \n" \
    "	str q20, [x27]                  \n" \
    "   add x27, x27, #80               \n" \
    \
    "	fmax v15.4s, v15.4s, v20.4s     \n" \
    "	str q25, [x27]                  \n" \
    "   add x27, x27, #80               \n" \
    "	str q30, [x27]                  \n" \
    "	fmax v25.4s, v25.4s, v30.4s     \n" \
    "	fmax v15.4s, v15.4s, v3.4s      \n" \
    "	fmax v15.4s, v15.4s, v25.4s     \n" \
    "	str q15, [x9], #16              \n" \
    \
    "	fmla v16.4s, v8.4s, v1.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v21.4s, v9.4s , v1.s[3]    \n" \
    "	fmla v26.4s, v10.4s, v1.s[3]    \n" \
    "	fmla v31.4s, v11.4s, v1.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    \
    "	fmul v16.4s, v16.4s, v2.s[0]    \n" \
    "	fmul v21.4s, v21.4s, v2.s[0]    \n" \
    "	fmul v26.4s, v26.4s, v2.s[0]    \n" \
    "	fmul v31.4s, v31.4s, v2.s[0]    \n" \
    \
    "	fadd v16.4s, v16.4s, v3.4s      \n" \
    "	ldr q3, [x9]                    \n" \
    "	fadd v21.4s, v21.4s, v4.4s      \n" \
    "	fadd v26.4s, v26.4s, v5.4s      \n" \
    "	fadd v31.4s, v31.4s, v6.4s      \n" \
    "	str q16, [x28]                  \n" \
    "   add x28, x28, #80               \n" \
    "	str q21, [x28]                  \n" \
    "   add x28, x28, #80               \n" \
    \
    "	fmax v16.4s, v16.4s, v21.4s     \n" \
    "	str q26, [x28]                  \n" \
    "   add x28, x28, #80               \n" \
    "	str q31, [x28]                  \n" \
    "	fmax v26.4s, v26.4s, v31.4s     \n" \
    "	fmax v16.4s, v16.4s, v3.4s      \n" \
    "	fmax v16.4s, v16.4s, v26.4s     \n" \
    "	str q16, [x9]                   \n"

#define gemm_ukernel_5x16_u8_last_k_with_scale_mask_max \
	/*x6:scale  x9:max  x10:mask*/\
	"	prfm PLDL1STRM, [x9]            \n" \
    "	prfm PLDL1STRM, [x10]           \n" \
    "	prfm PLDL1STRM, [x10, x11]      \n" \
    \
	"	prfm PLDL1STRM, [x24]           \n" \
	"	prfm PLDL1STRM, [x25]           \n" \
	"	prfm PLDL1STRM, [x26]           \n" \
	"	prfm PLDL1STRM, [x27]           \n" \
	"	prfm PLDL1STRM, [x28]           \n" \
	\
    "   ld1r {v2.4s}, [x6]              \n" \
	\
    "	fmla v12.4s, v8.4s, v5.s[3]     \n"/*row 1*/\
    "	fmla v17.4s, v9.4s, v5.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v22.4s, v10.4s, v5.s[3]    \n" \
    "	fmla v27.4s, v11.4s, v5.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    \
    "	fmul v12.4s, v12.4s, v2.s[0]    \n" \
    "	fmul v17.4s, v17.4s, v2.s[0]    \n" \
    "	fmul v22.4s, v22.4s, v2.s[0]    \n" \
    "	fmul v27.4s, v27.4s, v2.s[0]    \n" \
    \
    "	fadd v12.4s, v12.4s, v3.4s      \n" \
    "	ldr q3, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
	"	prfm PLDL1STRM, [x10, x11]      \n" \
    "	fadd v17.4s, v17.4s, v4.4s      \n" \
    "	ld1r {v4.4s}, [x9]              \n" \
    "	fadd v22.4s, v22.4s, v5.4s      \n" \
    "	fadd v27.4s, v27.4s, v3.4s      \n" \
    "	str q12, [x24]                  \n" \
    "   add x24, x24, #80               \n" \
    "	str q17, [x24]                  \n" \
    "   add x24, x24, #80               \n" \
    \
    "	fmax v12.4s, v12.4s, v17.4s     \n" \
    "	str q22, [x24]                  \n" \
    "   add x24, x24, #80               \n" \
    "	str q27, [x24]                  \n" \
    "	fmax v22.4s, v22.4s, v27.4s     \n" \
    "	fmax v12.4s, v12.4s, v4.4s      \n" \
    "	fmax v12.4s, v12.4s, v22.4s     \n" \
	"	fmaxv s12, v12.4s               \n" \
    "	str s12, [x9], #4               \n" \
    \
    "	fmla v13.4s, v8.4s, v6.s[3]     \n"/*row 2*/\
    "	fmla v18.4s, v9.4s, v6.s[3]     \n" \
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v23.4s, v10.4s, v6.s[3]    \n" \
    "	fmla v28.4s, v11.4s, v6.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
	"	prfm PLDL1STRM, [x10, x11]      \n" \
    \
    "	fmul v13.4s, v13.4s, v2.s[0]    \n" \
    "	fmul v18.4s, v18.4s, v2.s[0]    \n" \
    "	fmul v23.4s, v23.4s, v2.s[0]    \n" \
    "	fmul v28.4s, v28.4s, v2.s[0]    \n" \
    \
    "	fadd v13.4s, v13.4s, v3.4s      \n" \
    "	ld1r {v3.4s}, [x9]              \n" \
    "	fadd v18.4s, v18.4s, v4.4s      \n" \
    "	fadd v23.4s, v23.4s, v5.4s      \n" \
    "	fadd v28.4s, v28.4s, v6.4s      \n" \
    "	str q13, [x25]                  \n" \
    "   add x25, x25, #80               \n" \
    "	str q18, [x25]                  \n" \
    "   add x25, x25, #80               \n" \
    \
    "	fmax v13.4s, v13.4s, v18.4s     \n" \
    "	str q23, [x25]                  \n" \
    "   add x25, x25, #80               \n" \
    "	str q28, [x25]                  \n" \
    "	fmax v23.4s, v23.4s, v28.4s     \n" \
    "	fmax v13.4s, v13.4s, v3.4s      \n" \
    "	fmax v13.4s, v13.4s, v23.4s     \n" \
	"	fmaxv s13, v13.4s               \n" \
    "	str s13, [x9], #4               \n" \
    \
    "   fmla v14.4s, v8.4s, v7.s[3]     \n"/*row 3*/\
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "   fmla v19.4s, v9.4s , v7.s[3]    \n" \
    "   fmla v24.4s, v10.4s, v7.s[3]    \n" \
    "   fmla v29.4s, v11.4s, v7.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
	"	prfm PLDL1STRM, [x10, x11]      \n" \
    \
    "	fmul v14.4s, v14.4s, v2.s[0]    \n" \
    "	fmul v19.4s, v19.4s, v2.s[0]    \n" \
    "	fmul v24.4s, v24.4s, v2.s[0]    \n" \
    "	fmul v29.4s, v29.4s, v2.s[0]    \n" \
    \
    "	fadd v14.4s, v14.4s, v3.4s      \n" \
    "	ld1r {v3.4s}, [x9]              \n" \
    "	fadd v19.4s, v19.4s, v4.4s      \n" \
    "	fadd v24.4s, v24.4s, v5.4s      \n" \
    "	fadd v29.4s, v29.4s, v6.4s      \n" \
    "	str q14, [x26]                  \n" \
    "   add x26, x26, #80               \n" \
    "	str q19, [x26]                  \n" \
    "   add x26, x26, #80               \n" \
    \
    "	fmax v14.4s, v14.4s, v19.4s     \n" \
    "	str q24, [x26]                  \n" \
    "   add x26, x26, #80               \n" \
    "	str q29, [x26]                  \n" \
    "	fmax v24.4s, v24.4s, v29.4s     \n" \
    "	fmax v14.4s, v14.4s, v3.4s      \n" \
    "	fmax v14.4s, v14.4s, v24.4s     \n" \
	"	fmaxv s14, v14.4s               \n" \
    "	str s14, [x9], #4               \n" \
    \
    "	fmla v15.4s, v8.4s, v0.s[3]     \n"/*row 4*/\
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v20.4s, v9.4s , v0.s[3]    \n" \
    "	fmla v25.4s, v10.4s, v0.s[3]    \n" \
    "	fmla v30.4s, v11.4s, v0.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    "	add x10, x10, x11               \n" \
    \
    "	fmul v15.4s, v15.4s, v2.s[0]    \n" \
    "	fmul v20.4s, v20.4s, v2.s[0]    \n" \
    "	fmul v25.4s, v25.4s, v2.s[0]    \n" \
    "	fmul v30.4s, v30.4s, v2.s[0]    \n" \
    \
    "	fadd v15.4s, v15.4s, v3.4s      \n" \
    "	ld1r {v3.4s}, [x9]              \n" \
    "	fadd v20.4s, v20.4s, v4.4s      \n" \
    "	fadd v25.4s, v25.4s, v5.4s      \n" \
    "	fadd v30.4s, v30.4s, v6.4s      \n" \
    "	str q15, [x27]                  \n" \
    "   add x27, x27, #80               \n" \
    "	str q20, [x27]                  \n" \
    "   add x27, x27, #80               \n" \
    \
    "	fmax v15.4s, v15.4s, v20.4s     \n" \
    "	str q25, [x27]                  \n" \
    "   add x27, x27, #80               \n" \
    "	str q30, [x27]                  \n" \
    "	fmax v25.4s, v25.4s, v30.4s     \n" \
    "	fmax v15.4s, v15.4s, v3.4s      \n" \
    "	fmax v15.4s, v15.4s, v25.4s     \n" \
	"	fmaxv s15, v15.4s               \n" \
    "	str s15, [x9], #4               \n" \
    \
    "	fmla v16.4s, v8.4s, v1.s[3]     \n"/*row 5*/\
    "	ldr q3, [x10]                   \n" \
    "	ldr q4, [x10, #16]              \n" \
    "	fmla v21.4s, v9.4s , v1.s[3]    \n" \
    "	fmla v26.4s, v10.4s, v1.s[3]    \n" \
    "	fmla v31.4s, v11.4s, v1.s[3]    \n" \
    "	ldr q5, [x10, #32]              \n" \
    "	ldr q6, [x10, #48]              \n" \
    \
    "	fmul v16.4s, v16.4s, v2.s[0]    \n" \
    "	fmul v21.4s, v21.4s, v2.s[0]    \n" \
    "	fmul v26.4s, v26.4s, v2.s[0]    \n" \
    "	fmul v31.4s, v31.4s, v2.s[0]    \n" \
    \
    "	fadd v16.4s, v16.4s, v3.4s      \n" \
    "	ld1r {v3.4s}, [x9]              \n" \
    "	fadd v21.4s, v21.4s, v4.4s      \n" \
    "	fadd v26.4s, v26.4s, v5.4s      \n" \
    "	fadd v31.4s, v31.4s, v6.4s      \n" \
    "	str q16, [x28]                  \n" \
    "   add x28, x28, #80               \n" \
    "	str q21, [x28]                  \n" \
    "   add x28, x28, #80               \n" \
    \
    "	fmax v16.4s, v16.4s, v21.4s     \n" \
    "	str q26, [x28]                  \n" \
    "   add x28, x28, #80               \n" \
    "	str q31, [x28]                  \n" \
    "	fmax v26.4s, v26.4s, v31.4s     \n" \
    "	fmax v16.4s, v16.4s, v3.4s      \n" \
    "	fmax v16.4s, v16.4s, v26.4s     \n" \
	"	fmaxv s16, v16.4s               \n" \
    "	str s16, [x9]                   \n"


// score(mxn, format(5x4)) = scale * q(mxk, row major) * kt(kxn, col major)
// score + mask(mxn, row major) = score
// fuse the per line max value of score
void fused_scalexqxkt_mask_max_kernel_bak(long m, long n, long k/*(head_dim==64)*/,
                    float *scale, float *q, float *kt, float *buffer_kt, float *score, long lds/*todo,careful*/,
                    float *mask, long ldm,
                    float *max_per_line)
{
    int js, is;
    float *addr_q, *addr_kt, *addr_score,
			*addr_mask, *addr_max;
    long ldm_in_bytes=(ldm*sizeof(float));

    for(js=0; js<n; js+=r2) {
        addr_q=q, addr_kt=(kt+js*k), addr_score=(score+(r1*r2)*(js/r2)),
                            addr_mask=(mask+js), addr_max=max_per_line;
        // pack
        {
            asm volatile (
                "ldr x3, %[k]                         \n"
                "ldr x4, %[addr_kt]                   \n"

                "ldr x6, %[scale]                     \n"
                "ldr x9, %[addr_max]                  \n"
                
                "ldr x0, %[addr_mask]                 \n"
                "ldr x11, %[ldm_in_bytes]             \n"

                "ldr x8, %[buffer_kt]                 \n"

                "ldr x24, %[addr_score]               \n" // score
                "add x25, x24, #16                    \n"
                "add x26, x25, #16                    \n"
                "add x27, x26, #16                    \n"
                "add x28, x27, #16                    \n"

                "mov x7, #4                           \n"

                "pack_5x4_start:                      \n"
                
                "   ldr x15, %[addr_q]                \n" // q
                "   add x16, x15, x3, lsl #2          \n"
                "   add x17, x16, x3, lsl #2          \n"
                "   add x18, x17, x3, lsl #2          \n"
                "   add x19, x18, x3, lsl #2          \n"

                "   mov x20, x4                       \n" // kt
                "   add x21, x20, x3, lsl #2          \n"
                "   add x22, x21, x3, lsl #2          \n"
                "   add x23, x22, x3, lsl #2          \n"
                "   add x4, x4, x3, lsl #4            \n"

                "   mov x10, x0                       \n"
                "   add x0, x0, #16                   \n"

                "   mov x12, x8                       \n"
                "   add x8, x8, #16                   \n"

                "   lsr x5, x3, #3                    \n"
                    pack_ukernel_5x4_u8_first_k
                "   subs x5, x5, #1                   \n"
                "   beq pack_5x4_end                  \n"
                
                "pack_5x4_body:                       \n"
                    pack_ukernel_5x4_u8_k1
                    pack_ukernel_5x4_u8_k0
                "   subs x5, x5, #1                   \n"
                "   bne pack_5x4_body                 \n"

                "pack_5x4_end:                        \n"
                    pack_ukernel_5x4_u8_last_k_with_scale_mask_max

                "   subs x7, x7, #1                   \n"
                "   bne pack_5x4_start                \n"

                :
                : [addr_q]       "m" (addr_q),
                  [addr_kt]      "m" (addr_kt),
                  [addr_score]   "m" (addr_score),
                  [k]            "m" (k),
                  [scale]        "m" (scale),
                  [buffer_kt]    "m" (buffer_kt),

                  [addr_mask]    "m" (addr_mask),
                  [ldm_in_bytes] "m" (ldm_in_bytes),

                  [addr_max]     "m" (addr_max)
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

        for(is=r1; is<m; is+=r1) {
            addr_q=(q+is*k), addr_score=(score+(is*lds)+(r1*r2)*(js/r2)),
                            addr_mask=(mask+is*ldm+js), addr_max=(max_per_line+is);
            // kernel
            {
                asm volatile (
                    "ldr x3, %[k]                         \n"

                    "ldr x6, %[scale]                     \n"
                    "ldr x11, %[ldm_in_bytes]             \n"

                    "kernel_5x16_start:                   \n"
                    "   ldr x15, %[addr_q]                \n" // q
                    "   add x16, x15, x3, lsl #2          \n"
                    "   add x17, x16, x3, lsl #2          \n"
                    "   add x18, x17, x3, lsl #2          \n"
                    "   add x19, x18, x3, lsl #2          \n"

                    "   ldr x20, %[buffer_kt]             \n" // kt

                    "   ldr x24, %[addr_score]            \n" // score
                    "   add x25, x24, #16                 \n"
                    "   add x26, x25, #16                 \n"
                    "   add x27, x26, #16                 \n"
                    "   add x28, x27, #16                 \n"

                    "   ldr x9, %[addr_max]               \n"
                    "   ldr x10, %[addr_mask]             \n"

                    "   lsr x5, x3, #3                    \n"
                        gemm_ukernel_5x16_u8_first_k
                    "   b kernel_5x16_body                \n"
                    
                    "kernel_5x16_mid:                     \n"
                        gemm_ukernel_5x16_u8_k0

                    "kernel_5x16_body:                    \n"
                        gemm_ukernel_5x16_u8_k1
                        gemm_ukernel_5x16_u8_k2
                        gemm_ukernel_5x16_u8_k3
                        gemm_ukernel_5x16_u8_k4
                        gemm_ukernel_5x16_u8_k5
                        gemm_ukernel_5x16_u8_k6
                    "   subs x5, x5, #1                   \n"
                    "   beq kernel_5x16_end               \n"
                        gemm_ukernel_5x16_u8_k7
                    "   b kernel_5x16_mid                 \n"

                    "kernel_5x16_end:                     \n"
                        gemm_ukernel_5x16_u8_last_k_with_scale_mask_max

                    :
                    : [addr_q]       "m" (addr_q),
                      [buffer_kt]    "m" (buffer_kt),
                      [addr_score]   "m" (addr_score),
                      [k]            "m" (k),
                      [scale]        "m" (scale),
  
                      [addr_mask]    "m" (addr_mask),
                      [ldm_in_bytes] "m" (ldm_in_bytes),
  
                      [addr_max]     "m" (addr_max)
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


// better performace, edge case
// score(mxn, format(5x4)) = scale * q(mxk, row major) * kt(kxn, col major)
// score + mask(mxn, row major) = score
// fuse the per line max value of score
void fused_scalexqxkt_mask_max_kernel(long m, long n, long k/*(head_dim==64)*/,
                    float *scale, float *q, float *kt, float *buffer_kt, float *score, long lds/*todo,careful*/,
                    float *mask, long ldm,
                    float *max_per_line)
{
    long lds_in_bytes=lds*sizeof(float), ldm_in_bytes=ldm*sizeof(float);

    asm volatile(
// -------------------------------------------------------
        // ping
        ".macro PACK_KERNEL_M5xN4_FIRST_K4              \n"

        "   ldr q8, [x20], #16                          \n" // b0
        "   ldr q0, [x15], #16                          \n"
        "   ldr q1, [x16], #16                          \n"
        "   fmul v12.4s, v8.4s, v0.4s                   \n"
        "   ldr q2, [x17], #16                          \n"
        "   fmul v13.4s, v8.4s, v1.4s                   \n"
        "   ldr q3, [x18], #16                          \n"
        "   fmul v14.4s, v8.4s, v2.4s                   \n"
        "   ldr q4, [x19], #16                          \n"
        "   fmul v15.4s, v8.4s, v3.4s                   \n"
        "   ldr q9, [x21], #16                          \n"
        "   fmul v16.4s, v8.4s, v4.4s                   \n"

        "   ldr q10, [x22], #16                         \n" // b1
        "   fmul v17.4s, v9.4s, v0.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmul v18.4s, v9.4s, v1.4s                   \n"
        "   fmul v19.4s, v9.4s, v2.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #64             			    \n"
        "   fmul v20.4s, v9.4s, v3.4s     				\n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #64             			    \n"
        "   fmul v21.4s, v9.4s, v4.4s     			    \n"

        "   ldr q5, [x15], #16                          \n" // b2
        "   fmul v22.4s, v10.4s, v0.4s                  \n"
        "   fmul v23.4s, v10.4s, v1.4s                  \n"
        "   ldr q6, [x16], #16                          \n"
        "   fmul v24.4s, v10.4s, v2.4s                  \n"
        "   fmul v25.4s, v10.4s, v3.4s                  \n"
        "   ldr q7, [x17], #16                          \n"
        "   fmul v26.4s, v10.4s, v4.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #64                           \n"
        
        "   fmul v27.4s, v11.4s, v0.4s    			    \n" // b3
        "   ldr q0, [x18], #16            			    \n"
        "   fmul v28.4s, v11.4s, v1.4s    			    \n"
        "   ldr q1, [x19], #16            			    \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   ldr q8, [x20], #16                          \n"
        "   fmul v29.4s, v11.4s, v2.4s                  \n"
        "   fmul v30.4s, v11.4s, v3.4s                  \n"
        "   fmul v31.4s, v11.4s, v4.4s                  \n"
        ".endm                                          \n"

        // pong
        ".macro PACK_KERNEL_M5xN4_PONG_K4               \n"
        "   ldr q9, [x21], #16                          \n"
        "   fmla v12.4s, v8.4s, v5.4s                   \n"
        "   fmla v13.4s, v8.4s, v6.4s                   \n"
        "   ldr q10, [x22], #16                         \n"
        "   fmla v14.4s, v8.4s, v7.4s                   \n"
        "   fmla v15.4s, v8.4s, v0.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmla v16.4s, v8.4s, v1.4s                   \n"
        
        "   ldr q2, [x17], #16                          \n"
        "   fmla v17.4s, v9.4s, v5.4s                   \n"
        "   fmla v18.4s, v9.4s, v6.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v19.4s, v9.4s, v7.4s                   \n"
        "   fmla v20.4s, v9.4s, v0.4s                   \n"
        "   fmla v21.4s, v9.4s, v1.4s                   \n"
        
        "   ldr q3, [x18], #16                          \n"
        "   ldr q4, [x19], #16                          \n"
        "   fmla v22.4s, v10.4s, v5.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v23.4s, v10.4s, v6.4s                  \n"
        "   fmla v24.4s, v10.4s, v7.4s                  \n"
        "   fmla v25.4s, v10.4s, v0.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v26.4s, v10.4s, v1.4s                  \n"
        
        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   ldr q8, [x20], #16                          \n"
        "   fmla v30.4s, v11.4s, v0.4s                  \n"
        "   ldr q0, [x15], #16                          \n"
        "   fmla v31.4s, v11.4s, v1.4s                  \n"
        "   ldr q1, [x16], #16                          \n"
        "   fmla v27.4s, v11.4s, v5.4s                  \n" 
        "   fmla v28.4s, v11.4s, v6.4s                  \n"
        "   fmla v29.4s, v11.4s, v7.4s                  \n"
        ".endm                                          \n"

        // ping 
        ".macro PACK_KERNEL_M5xN4_PING_K4               \n"
        "   ldr q9, [x21], #16                          \n"
        "   fmla v12.4s, v8.4s, v0.4s                   \n"
        "   fmla v13.4s, v8.4s, v1.4s                   \n"
        "   ldr q10, [x22], #16                         \n"
        "   fmla v14.4s, v8.4s, v2.4s                   \n"
        "   fmla v15.4s, v8.4s, v3.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmla v16.4s, v8.4s, v4.4s                   \n"
        
        "   ldr q5, [x15], #16                          \n"
        "   fmla v17.4s, v9.4s, v0.4s                   \n"
        "   fmla v18.4s, v9.4s, v1.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v19.4s, v9.4s, v2.4s                   \n"
        "   fmla v20.4s, v9.4s, v3.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v21.4s, v9.4s, v4.4s                   \n"

        "   ldr q6, [x16], #16                          \n"
        "   ldr q7, [x17], #16                          \n"
        "   fmla v22.4s, v10.4s, v0.4s                  \n"
        "   fmla v23.4s, v10.4s, v1.4s                  \n"
        "   fmla v24.4s, v10.4s, v2.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v25.4s, v10.4s, v3.4s                  \n"
        "   fmla v26.4s, v10.4s, v4.4s                  \n"

        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   ldr q8, [x20], #16                          \n"
        "   fmla v27.4s, v11.4s, v0.4s                  \n"
        "   ldr q0, [x18], #16                          \n"
        "   fmla v28.4s, v11.4s, v1.4s                  \n"
        "   ldr q1, [x19], #16                          \n"
        "   fmla v29.4s, v11.4s, v2.4s                  \n"
        "   fmla v30.4s, v11.4s, v3.4s                  \n"
        "   fmla v31.4s, v11.4s, v4.4s                  \n"
        ".endm                                          \n"

        // pong
        ".macro PACK_KERNEL_M5xN4_LAST_K4               \n"
        "   prfm PLDL1STRM, [x9]                        \n"

        "   ldr q9, [x21], #16                          \n" // b0
        "   fmla v12.4s, v8.4s, v5.4s                   \n"
        "   fmla v13.4s, v8.4s, v6.4s                   \n"
        "   ldr q10, [x22], #16                         \n"
        "   fmla v14.4s, v8.4s, v7.4s                   \n"
        "   fmla v15.4s, v8.4s, v0.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmla v16.4s, v8.4s, v1.4s                   \n"

		"   ld1r {v3.4s}, [x9]                          \n"   
        "   add x9, x9, #4                              \n"
        "   fmla v17.4s, v9.4s, v5.4s                   \n" // b1
        "   fmla v18.4s, v9.4s, v6.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v19.4s, v9.4s, v7.4s                   \n"
        "   fmla v20.4s, v9.4s, v0.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v21.4s, v9.4s, v1.4s                   \n"
        
		"   ld1r {v2.4s}, [x6]                          \n"
        "   faddp v12.4s, v12.4s, v17.4s                \n" // reduce b0/b1
        "   ld1r {v17.4s}, [x9]                         \n"
        "   add x9, x9, #4                              \n"
        "   faddp v13.4s, v13.4s, v18.4s                \n"
		"	ldr q18, [x27]                              \n"
		"	add x27, x27, x11                           \n"
        "   faddp v14.4s, v14.4s, v19.4s                \n"
		"	ldr q19, [x27]                              \n"
		"	add x27, x27, x11                           \n"
        "   faddp v15.4s, v15.4s, v20.4s                \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   faddp v16.4s, v16.4s, v21.4s                \n"
        "   ld1r {v4.4s}, [x9]                          \n"
        "   add x9, x9, #4                              \n"
        "   fmla v22.4s, v10.4s, v5.4s                  \n"
		"	ldr q20, [x27]                              \n"
		"	add x27, x27, x11                           \n"
		"	ldr q21, [x27]                              \n"
		"	add x27, x27, x11                           \n"
		
        "   fmla v23.4s, v10.4s, v6.4s                  \n" // b2
        "   fmla v24.4s, v10.4s, v7.4s                  \n"
        "   fmla v25.4s, v10.4s, v0.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #64                           \n"
        "   fmla v26.4s, v10.4s, v1.4s                  \n"
        
        "   fmla v30.4s, v11.4s, v0.4s                  \n" // b3
        "   fmla v31.4s, v11.4s, v1.4s                  \n"
        "   ld1r {v0.4s}, [x9]                          \n"
        "   add x9, x9, #4                              \n"
        "   ld1r {v1.4s}, [x9]                          \n"
        "   sub x9, x9, #16                             \n"
        "   fmla v27.4s, v11.4s, v5.4s                  \n"
        "   fmla v28.4s, v11.4s, v6.4s                  \n"
        "   fmla v29.4s, v11.4s, v7.4s                  \n"
            
        "   faddp v22.4s, v22.4s, v27.4s                \n" // reduce b2/b3; scale; mask; max; str
        "   faddp v23.4s, v23.4s, v28.4s                \n"
        "   faddp v12.4s, v12.4s, v22.4s                \n"
		"	ldr q22, [x27]                              \n"
        "   faddp v13.4s, v13.4s, v23.4s                \n"
        "   fmul v12.4s, v12.4s, v2.s[0]                \n"
        "   fmul v13.4s, v13.4s, v2.s[0]                \n"
		"	fadd v12.4s, v12.4s, v18.4s                 \n"
		"	fadd v13.4s, v13.4s, v19.4s                 \n"
        "   str q12, [x24], #80                         \n"
        "   str q13, [x25], #80                         \n"
        "   fmax v3.4s, v3.4s, v12.4s                   \n"
        "   fmax v17.4s, v17.4s, v13.4s                 \n"
        "   fmaxv s3, v3.4s                             \n"
        "   fmaxv s17, v17.4s                           \n"
        "   faddp v24.4s, v24.4s, v29.4s                \n"
        "   faddp v25.4s, v25.4s, v30.4s                \n"
        "   str s3, [x9]                                \n"
        "   str s17, [x9, #4]                           \n"
        "   faddp v14.4s, v14.4s, v24.4s                \n"
        "   faddp v15.4s, v15.4s, v25.4s                \n"
        "   fmul v14.4s, v14.4s, v2.s[0]                \n"
        "   fmul v15.4s, v15.4s, v2.s[0]                \n"
		"	fadd v14.4s, v14.4s, v20.4s                 \n"
		"	fadd v15.4s, v15.4s, v21.4s                 \n"
        "   str q14, [x26], #16                         \n"
        "   str q15, [x26], #64                         \n"
        "   fmax v4.4s, v4.4s, v14.4s                   \n"
        "   fmax v0.4s, v0.4s, v15.4s                   \n"
        "   fmaxv s4, v4.4s                             \n"
        "   fmaxv s0, v0.4s                             \n"
        "   faddp v26.4s, v26.4s, v31.4s                \n"
        "   faddp v16.4s, v16.4s, v26.4s                \n"
        "   str s4, [x9, #8]                            \n"
        "   str s0, [x9, #12]                           \n"
        "   fmul v16.4s, v16.4s, v2.s[0]                \n"
		"	fadd v16.4s, v16.4s, v22.4s                 \n"
        "   str q16, [x28], #80                         \n"
        "   fmax v1.4s, v1.4s, v16.4s                   \n"
        "   fmaxv s1, v1.4s                             \n"
        "   str s1, [x9, #16]                           \n"
        ".endm                                          \n"

// -------------------------------------------------------
        // ping
        ".macro PACK_KERNEL_EDGE_M5xN4_FIRST_K4         \n"

        "   ldr q8, [x20], #16                          \n" // b0
        "   ldr q0, [x15], #16                          \n"
        "   ldr q1, [x16], #16                          \n"
        "   fmul v12.4s, v8.4s, v0.4s                   \n"
        "   ldr q2, [x17], #16                          \n"
        "   fmul v13.4s, v8.4s, v1.4s                   \n"
        "   ldr q3, [x18], #16                          \n"
        "   fmul v14.4s, v8.4s, v2.4s                   \n"
        "   ldr q4, [x19], #16                          \n"
        "   fmul v15.4s, v8.4s, v3.4s                   \n"
        "   ldr q9, [x21], #16                          \n"
        "   fmul v16.4s, v8.4s, v4.4s                   \n"

        "   ldr q10, [x22], #16                         \n" // b1
        "   fmul v17.4s, v9.4s, v0.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmul v18.4s, v9.4s, v1.4s                   \n"
        "   fmul v19.4s, v9.4s, v2.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #16             			    \n"
        "   fmul v20.4s, v9.4s, v3.4s     				\n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #16             			    \n"
        "   fmul v21.4s, v9.4s, v4.4s     			    \n"

        "   ldr q5, [x15], #16                          \n" // b2
        "   fmul v22.4s, v10.4s, v0.4s                  \n"
        "   fmul v23.4s, v10.4s, v1.4s                  \n"
        "   ldr q6, [x16], #16                          \n"
        "   fmul v24.4s, v10.4s, v2.4s                  \n"
        "   fmul v25.4s, v10.4s, v3.4s                  \n"
        "   ldr q7, [x17], #16                          \n"
        "   fmul v26.4s, v10.4s, v4.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #16                           \n"
        
        "   fmul v27.4s, v11.4s, v0.4s    			    \n" // b3
        "   ldr q0, [x18], #16            			    \n"
        "   fmul v28.4s, v11.4s, v1.4s    			    \n"
        "   ldr q1, [x19], #16            			    \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   ldr q8, [x20], #16                          \n"
        "   fmul v29.4s, v11.4s, v2.4s                  \n"
        "   fmul v30.4s, v11.4s, v3.4s                  \n"
        "   fmul v31.4s, v11.4s, v4.4s                  \n"
        ".endm                                          \n"

        // pong
        ".macro PACK_KERNEL_EDGE_M5xN4_PONG_K4          \n"
        "   ldr q9, [x21], #16                          \n"
        "   fmla v12.4s, v8.4s, v5.4s                   \n"
        "   fmla v13.4s, v8.4s, v6.4s                   \n"
        "   ldr q10, [x22], #16                         \n"
        "   fmla v14.4s, v8.4s, v7.4s                   \n"
        "   fmla v15.4s, v8.4s, v0.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmla v16.4s, v8.4s, v1.4s                   \n"
        
        "   ldr q2, [x17], #16                          \n"
        "   fmla v17.4s, v9.4s, v5.4s                   \n"
        "   fmla v18.4s, v9.4s, v6.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v19.4s, v9.4s, v7.4s                   \n"
        "   fmla v20.4s, v9.4s, v0.4s                   \n"
        "   fmla v21.4s, v9.4s, v1.4s                   \n"
        
        "   ldr q3, [x18], #16                          \n"
        "   ldr q4, [x19], #16                          \n"
        "   fmla v22.4s, v10.4s, v5.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v23.4s, v10.4s, v6.4s                  \n"
        "   fmla v24.4s, v10.4s, v7.4s                  \n"
        "   fmla v25.4s, v10.4s, v0.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v26.4s, v10.4s, v1.4s                  \n"
        
        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   ldr q8, [x20], #16                          \n"
        "   fmla v30.4s, v11.4s, v0.4s                  \n"
        "   ldr q0, [x15], #16                          \n"
        "   fmla v31.4s, v11.4s, v1.4s                  \n"
        "   ldr q1, [x16], #16                          \n"
        "   fmla v27.4s, v11.4s, v5.4s                  \n" 
        "   fmla v28.4s, v11.4s, v6.4s                  \n"
        "   fmla v29.4s, v11.4s, v7.4s                  \n"
        ".endm                                          \n"

        // ping 
        ".macro PACK_KERNEL_EDGE_M5xN4_PING_K4          \n"
        "   ldr q9, [x21], #16                          \n"
        "   fmla v12.4s, v8.4s, v0.4s                   \n"
        "   fmla v13.4s, v8.4s, v1.4s                   \n"
        "   ldr q10, [x22], #16                         \n"
        "   fmla v14.4s, v8.4s, v2.4s                   \n"
        "   fmla v15.4s, v8.4s, v3.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmla v16.4s, v8.4s, v4.4s                   \n"
        
        "   ldr q5, [x15], #16                          \n"
        "   fmla v17.4s, v9.4s, v0.4s                   \n"
        "   fmla v18.4s, v9.4s, v1.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v19.4s, v9.4s, v2.4s                   \n"
        "   fmla v20.4s, v9.4s, v3.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v21.4s, v9.4s, v4.4s                   \n"

        "   ldr q6, [x16], #16                          \n"
        "   ldr q7, [x17], #16                          \n"
        "   fmla v22.4s, v10.4s, v0.4s                  \n"
        "   fmla v23.4s, v10.4s, v1.4s                  \n"
        "   fmla v24.4s, v10.4s, v2.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v25.4s, v10.4s, v3.4s                  \n"
        "   fmla v26.4s, v10.4s, v4.4s                  \n"

        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   ldr q8, [x20], #16                          \n"
        "   fmla v27.4s, v11.4s, v0.4s                  \n"
        "   ldr q0, [x18], #16                          \n"
        "   fmla v28.4s, v11.4s, v1.4s                  \n"
        "   ldr q1, [x19], #16                          \n"
        "   fmla v29.4s, v11.4s, v2.4s                  \n"
        "   fmla v30.4s, v11.4s, v3.4s                  \n"
        "   fmla v31.4s, v11.4s, v4.4s                  \n"
        ".endm                                          \n"

        // pong
        ".macro PACK_KERNEL_EDGE_M5xN4_LAST_K4          \n"
        "   prfm PLDL1STRM, [x9]                        \n"

        "   ldr q9, [x21], #16                          \n" // b0
        "   fmla v12.4s, v8.4s, v5.4s                   \n"
        "   fmla v13.4s, v8.4s, v6.4s                   \n"
        "   ldr q10, [x22], #16                         \n"
        "   fmla v14.4s, v8.4s, v7.4s                   \n"
        "   fmla v15.4s, v8.4s, v0.4s                   \n"
        "   ldr q11, [x23], #16                         \n"
        "   fmla v16.4s, v8.4s, v1.4s                   \n"

		"   ld1r {v3.4s}, [x9]                          \n"
        "   add x9, x9, #4                              \n"
        "   fmla v17.4s, v9.4s, v5.4s                   \n" // b1
        "   fmla v18.4s, v9.4s, v6.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[0], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v19.4s, v9.4s, v7.4s                   \n"
        "   fmla v20.4s, v9.4s, v0.4s                   \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[1], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v21.4s, v9.4s, v1.4s                   \n"
        
		"   ld1r {v2.4s}, [x6]                          \n"
        "   faddp v12.4s, v12.4s, v17.4s                \n" // reduce b0/b1
        "   ld1r {v17.4s}, [x9]                         \n"
        "   add x9, x9, #4                              \n"
        "   faddp v13.4s, v13.4s, v18.4s                \n"
		"	ldr q18, [x27]                              \n"
		"	add x27, x27, x11                           \n"
        "   faddp v14.4s, v14.4s, v19.4s                \n"
		"	ldr q19, [x27]                              \n"
		"	add x27, x27, x11                           \n"
        "   faddp v15.4s, v15.4s, v20.4s                \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[2], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   faddp v16.4s, v16.4s, v21.4s                \n"
        "   ld1r {v4.4s}, [x9]                          \n"
        "   add x9, x9, #4                              \n"
        "   fmla v22.4s, v10.4s, v5.4s                  \n"
		"	ldr q20, [x27]                              \n"
		"	add x27, x27, x11                           \n"
		"	ldr q21, [x27]                              \n"
		"	add x27, x27, x11                           \n"
		
        "   fmla v23.4s, v10.4s, v6.4s                  \n" // b2
        "   fmla v24.4s, v10.4s, v7.4s                  \n"
        "   fmla v25.4s, v10.4s, v0.4s                  \n"
        "   st4 {v8.s, v9.s, v10.s, v11.s}[3], [x12]    \n"
        "   add x12, x12, #16                           \n"
        "   fmla v26.4s, v10.4s, v1.4s                  \n"
        
        "   fmla v30.4s, v11.4s, v0.4s                  \n" // b3
        "   fmla v31.4s, v11.4s, v1.4s                  \n"
        "   ld1r {v0.4s}, [x9]                          \n"
        "   add x9, x9, #4                              \n"
        "   ld1r {v1.4s}, [x9]                          \n"
        "   sub x9, x9, #16                             \n"
        "   fmla v27.4s, v11.4s, v5.4s                  \n"
        "   fmla v28.4s, v11.4s, v6.4s                  \n"
        "   fmla v29.4s, v11.4s, v7.4s                  \n"
            
        "   faddp v22.4s, v22.4s, v27.4s                \n" // reduce b2/b3; scale; mask; max; str
        "   faddp v23.4s, v23.4s, v28.4s                \n"
        "   faddp v12.4s, v12.4s, v22.4s                \n"
		"	ldr q22, [x27]                              \n"
        "   faddp v13.4s, v13.4s, v23.4s                \n"
        "   fmul v12.4s, v12.4s, v2.s[0]                \n"
        "   fmul v13.4s, v13.4s, v2.s[0]                \n"
		"	fadd v12.4s, v12.4s, v18.4s                 \n"
		"	fadd v13.4s, v13.4s, v19.4s                 \n"
        "   str q12, [x24], #80                         \n"
        "   str q13, [x25], #80                         \n"
        "   fmax v3.4s, v3.4s, v12.4s                   \n"
        "   fmax v17.4s, v17.4s, v13.4s                 \n"
        "   fmaxv s3, v3.4s                             \n"
        "   fmaxv s17, v17.4s                           \n"
        "   faddp v24.4s, v24.4s, v29.4s                \n"
        "   faddp v25.4s, v25.4s, v30.4s                \n"
        "   str s3, [x9]                                \n"
        "   str s17, [x9, #4]                           \n"
        "   faddp v14.4s, v14.4s, v24.4s                \n"
        "   faddp v15.4s, v15.4s, v25.4s                \n"
        "   fmul v14.4s, v14.4s, v2.s[0]                \n"
        "   fmul v15.4s, v15.4s, v2.s[0]                \n"
		"	fadd v14.4s, v14.4s, v20.4s                 \n"
		"	fadd v15.4s, v15.4s, v21.4s                 \n"
        "   str q14, [x26], #16                         \n"
        "   str q15, [x26], #64                         \n"
        "   fmax v4.4s, v4.4s, v14.4s                   \n"
        "   fmax v0.4s, v0.4s, v15.4s                   \n"
        "   fmaxv s4, v4.4s                             \n"
        "   fmaxv s0, v0.4s                             \n"
        "   faddp v26.4s, v26.4s, v31.4s                \n"
        "   faddp v16.4s, v16.4s, v26.4s                \n"
        "   str s4, [x9, #8]                            \n"
        "   str s0, [x9, #12]                           \n"
        "   fmul v16.4s, v16.4s, v2.s[0]                \n"
		"	fadd v16.4s, v16.4s, v22.4s                 \n"
        "   str q16, [x28], #80                         \n"
        "   fmax v1.4s, v1.4s, v16.4s                   \n"
        "   fmaxv s1, v1.4s                             \n"
        "   str s1, [x9, #16]                           \n"
        ".endm                                          \n"

// -------------------------------------------------------
        ".macro KERNEL_M5xN16_FIRST_K       \n"
        "   ldp q8, q9, [x20], #32          \n"
        "   ldr q0, [x15], #16              \n"
        "   ldr q1, [x16], #16              \n"

        "   fmul v12.4s, v8.4s, v0.s[0]     \n"
        "   fmul v17.4s, v9.4s, v0.s[0]     \n"
        "   ldr q2, [x17], #16              \n"
        "   fmul v13.4s, v8.4s, v1.s[0]     \n"
        "   fmul v18.4s, v9.4s, v1.s[0]     \n"
        "   ldr q3, [x18], #16              \n"
        "   fmul v14.4s, v8.4s, v2.s[0]     \n"
        "   fmul v19.4s, v9.4s, v2.s[0]     \n"
        "   ldr q4, [x19], #16              \n"
        "   fmul v15.4s, v8.4s, v3.s[0]     \n"
        "   ldp q10, q11, [x20], #32        \n"
        "   fmul v20.4s, v9.4s, v3.s[0]     \n"
        "   fmul v16.4s, v8.4s, v4.s[0]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmul v21.4s, v9.4s, v4.s[0]     \n"
        "   ldr q9, [x20], #16              \n"
        
        "   fmul v22.4s, v10.4s, v0.s[0]    \n"
        "   fmul v27.4s, v11.4s, v0.s[0]    \n"
        "   fmul v23.4s, v10.4s, v1.s[0]    \n"
        "   fmul v28.4s, v11.4s, v1.s[0]    \n"
        "   ldr q5, [x15], #16              \n"
        "   fmul v24.4s, v10.4s, v2.s[0]    \n"
        "   fmul v29.4s, v11.4s, v2.s[0]    \n"
        "   fmul v25.4s, v10.4s, v3.s[0]    \n"
        "   fmul v30.4s, v11.4s, v3.s[0]    \n"
        "   fmul v26.4s, v10.4s, v4.s[0]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmul v31.4s, v11.4s, v4.s[0]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K0            \n"
        "   fmla v12.4s, v8.4s, v0.s[0]     \n"
        "   fmla v17.4s, v9.4s, v0.s[0]     \n"
        "   fmla v13.4s, v8.4s, v1.s[0]     \n"
        "   fmla v18.4s, v9.4s, v1.s[0]     \n"
        "   fmla v14.4s, v8.4s, v2.s[0]     \n"
        "   fmla v19.4s, v9.4s, v2.s[0]     \n"
        "   fmla v15.4s, v8.4s, v3.s[0]     \n"
        "   fmla v20.4s, v9.4s, v3.s[0]     \n"
        "   fmla v16.4s, v8.4s, v4.s[0]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v4.s[0]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v22.4s, v10.4s, v0.s[0]    \n"
        "   fmla v27.4s, v11.4s, v0.s[0]    \n"
        "   fmla v23.4s, v10.4s, v1.s[0]    \n"
        "   fmla v28.4s, v11.4s, v1.s[0]    \n"
        "   ldr q5, [x15], #16              \n"
        "   fmla v24.4s, v10.4s, v2.s[0]    \n"
        "   fmla v29.4s, v11.4s, v2.s[0]    \n"
        "   fmla v25.4s, v10.4s, v3.s[0]    \n"
        "   fmla v30.4s, v11.4s, v3.s[0]    \n"
        "   fmla v26.4s, v10.4s, v4.s[0]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v4.s[0]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K1            \n"
        "   fmla v12.4s, v8.4s, v0.s[1]     \n"
        "   fmla v17.4s, v9.4s, v0.s[1]     \n"
        "   fmla v13.4s, v8.4s, v1.s[1]     \n"
        "   fmla v18.4s, v9.4s, v1.s[1]     \n"
        "   fmla v14.4s, v8.4s, v2.s[1]     \n"
        "   fmla v19.4s, v9.4s, v2.s[1]     \n"
        "   fmla v15.4s, v8.4s, v3.s[1]     \n"
        "   fmla v20.4s, v9.4s, v3.s[1]     \n"
        "   fmla v16.4s, v8.4s, v4.s[1]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v4.s[1]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v22.4s, v10.4s, v0.s[1]    \n"
        "   fmla v27.4s, v11.4s, v0.s[1]    \n"
        "   fmla v23.4s, v10.4s, v1.s[1]    \n"
        "   fmla v28.4s, v11.4s, v1.s[1]    \n"
        "   ldr q6, [x16], #16              \n"
        "   fmla v24.4s, v10.4s, v2.s[1]    \n"
        "   fmla v29.4s, v11.4s, v2.s[1]    \n"
        "   fmla v25.4s, v10.4s, v3.s[1]    \n"
        "   fmla v30.4s, v11.4s, v3.s[1]    \n"
        "   fmla v26.4s, v10.4s, v4.s[1]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v4.s[1]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K2            \n"
        "   fmla v12.4s, v8.4s, v0.s[2]     \n"
        "   fmla v17.4s, v9.4s, v0.s[2]     \n"
        "   fmla v13.4s, v8.4s, v1.s[2]     \n"
        "   fmla v18.4s, v9.4s, v1.s[2]     \n"
        "   fmla v14.4s, v8.4s, v2.s[2]     \n"
        "   fmla v19.4s, v9.4s, v2.s[2]     \n"
        "   fmla v15.4s, v8.4s, v3.s[2]     \n"
        "   fmla v20.4s, v9.4s, v3.s[2]     \n"
        "   fmla v16.4s, v8.4s, v4.s[2]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v4.s[2]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v22.4s, v10.4s, v0.s[2]    \n"
        "   fmla v27.4s, v11.4s, v0.s[2]    \n"
        "   fmla v23.4s, v10.4s, v1.s[2]    \n"
        "   fmla v28.4s, v11.4s, v1.s[2]    \n"
        "   ldr q7, [x17], #16              \n"
        "   fmla v24.4s, v10.4s, v2.s[2]    \n"
        "   fmla v29.4s, v11.4s, v2.s[2]    \n"
        "   fmla v25.4s, v10.4s, v3.s[2]    \n"
        "   fmla v30.4s, v11.4s, v3.s[2]    \n"
        "   fmla v26.4s, v10.4s, v4.s[2]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v4.s[2]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K3            \n"
        "   fmla v12.4s, v8.4s, v0.s[3]     \n"
        "   fmla v17.4s, v9.4s, v0.s[3]     \n"
        "   fmla v13.4s, v8.4s, v1.s[3]     \n"
        "   fmla v18.4s, v9.4s, v1.s[3]     \n"
        "   fmla v14.4s, v8.4s, v2.s[3]     \n"
        "   fmla v19.4s, v9.4s, v2.s[3]     \n"
        "   fmla v15.4s, v8.4s, v3.s[3]     \n"
        "   fmla v20.4s, v9.4s, v3.s[3]     \n"
        "   fmla v16.4s, v8.4s, v4.s[3]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v4.s[3]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v22.4s, v10.4s, v0.s[3]    \n"
        "   fmla v27.4s, v11.4s, v0.s[3]    \n"
        "   ldr q0, [x18], #16              \n"
        "   fmla v23.4s, v10.4s, v1.s[3]    \n"
        "   fmla v28.4s, v11.4s, v1.s[3]    \n"
        "   ldr q1, [x19], #16              \n"
        "   fmla v24.4s, v10.4s, v2.s[3]    \n"
        "   fmla v29.4s, v11.4s, v2.s[3]    \n"
        "   fmla v25.4s, v10.4s, v3.s[3]    \n"
        "   fmla v30.4s, v11.4s, v3.s[3]    \n"
        "   fmla v26.4s, v10.4s, v4.s[3]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v4.s[3]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K4            \n"
        "   fmla v12.4s, v8.4s, v5.s[0]     \n"
        "   fmla v17.4s, v9.4s, v5.s[0]     \n"
        "   fmla v13.4s, v8.4s, v6.s[0]     \n"
        "   fmla v18.4s, v9.4s, v6.s[0]     \n"
        "   fmla v14.4s, v8.4s, v7.s[0]     \n"
        "   fmla v19.4s, v9.4s, v7.s[0]     \n"
        "   fmla v15.4s, v8.4s, v0.s[0]     \n"
        "   fmla v20.4s, v9.4s, v0.s[0]     \n"
        "   fmla v16.4s, v8.4s, v1.s[0]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v1.s[0]     \n"
        "   ldr q9, [x20], #16              \n"
        
        "   fmla v22.4s, v10.4s, v5.s[0]    \n"
        "   fmla v27.4s, v11.4s, v5.s[0]    \n"
        "   fmla v23.4s, v10.4s, v6.s[0]    \n"
        "   fmla v28.4s, v11.4s, v6.s[0]    \n"
        "   ldr q2, [x17], #16              \n"
        "   fmla v24.4s, v10.4s, v7.s[0]    \n"
        "   fmla v29.4s, v11.4s, v7.s[0]    \n"
        "   fmla v25.4s, v10.4s, v0.s[0]    \n"
        "   fmla v30.4s, v11.4s, v0.s[0]    \n"
        "   fmla v26.4s, v10.4s, v1.s[0]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v1.s[0]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K5            \n"
        "   fmla v12.4s, v8.4s, v5.s[1]     \n"
        "   fmla v17.4s, v9.4s, v5.s[1]     \n"
        "   fmla v13.4s, v8.4s, v6.s[1]     \n"
        "   fmla v18.4s, v9.4s, v6.s[1]     \n"
        "   fmla v14.4s, v8.4s, v7.s[1]     \n"
        "   fmla v19.4s, v9.4s, v7.s[1]     \n"
        "   fmla v15.4s, v8.4s, v0.s[1]     \n"
        "   fmla v20.4s, v9.4s, v0.s[1]     \n"
        "   fmla v16.4s, v8.4s, v1.s[1]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v1.s[1]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v22.4s, v10.4s, v5.s[1]    \n"
        "   fmla v27.4s, v11.4s, v5.s[1]    \n"
        "   fmla v23.4s, v10.4s, v6.s[1]    \n"
        "   fmla v28.4s, v11.4s, v6.s[1]    \n"
        "   ldr q3, [x18], #16              \n"
        "   fmla v24.4s, v10.4s, v7.s[1]    \n"
        "   fmla v29.4s, v11.4s, v7.s[1]    \n"
        "   fmla v25.4s, v10.4s, v0.s[1]    \n"
        "   fmla v30.4s, v11.4s, v0.s[1]    \n"
        "   fmla v26.4s, v10.4s, v1.s[1]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v1.s[1]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K6            \n"
        "   fmla v12.4s, v8.4s, v5.s[2]     \n"
        "   fmla v17.4s, v9.4s, v5.s[2]     \n"
        "   fmla v13.4s, v8.4s, v6.s[2]     \n"
        "   fmla v18.4s, v9.4s, v6.s[2]     \n"
        "   fmla v14.4s, v8.4s, v7.s[2]     \n"
        "   fmla v19.4s, v9.4s, v7.s[2]     \n"
        "   fmla v15.4s, v8.4s, v0.s[2]     \n"
        "   fmla v20.4s, v9.4s, v0.s[2]     \n"
        "   fmla v16.4s, v8.4s, v1.s[2]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v1.s[2]     \n"
        "   ldr q9, [x20], #16              \n"
        
        "   fmla v22.4s, v10.4s, v5.s[2]    \n"
        "   fmla v27.4s, v11.4s, v5.s[2]    \n"
        "   fmla v23.4s, v10.4s, v6.s[2]    \n"
        "   fmla v28.4s, v11.4s, v6.s[2]    \n"
        "   ldr q4, [x19], #16              \n"
        "   fmla v24.4s, v10.4s, v7.s[2]    \n"
        "   fmla v29.4s, v11.4s, v7.s[2]    \n"
        "   fmla v25.4s, v10.4s, v0.s[2]    \n"
        "   fmla v30.4s, v11.4s, v0.s[2]    \n"
        "   fmla v26.4s, v10.4s, v1.s[2]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v31.4s, v11.4s, v1.s[2]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_M5xN16_K7            \n"
        "   fmla v12.4s, v8.4s, v5.s[3]     \n"
        "   fmla v17.4s, v9.4s, v5.s[3]     \n"
        "   fmla v13.4s, v8.4s, v6.s[3]     \n"
        "   fmla v18.4s, v9.4s, v6.s[3]     \n"
        "   fmla v14.4s, v8.4s, v7.s[3]     \n"
        "   fmla v19.4s, v9.4s, v7.s[3]     \n"
        "   fmla v15.4s, v8.4s, v0.s[3]     \n"
        "   fmla v20.4s, v9.4s, v0.s[3]     \n"
        "   fmla v16.4s, v8.4s, v1.s[3]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v1.s[3]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v25.4s, v10.4s, v0.s[3]    \n"
        "   fmla v30.4s, v11.4s, v0.s[3]    \n"
        "   ldr q0, [x15], #16              \n"
        "   fmla v26.4s, v10.4s, v1.s[3]    \n"
        "   fmla v31.4s, v11.4s, v1.s[3]    \n"
        "   ldr q1, [x16], #16              \n"
        "   fmla v22.4s, v10.4s, v5.s[3]    \n"
        "   fmla v27.4s, v11.4s, v5.s[3]    \n"
        "   fmla v23.4s, v10.4s, v6.s[3]    \n"
        "   fmla v28.4s, v11.4s, v6.s[3]    \n"
        "   fmla v24.4s, v10.4s, v7.s[3]    \n"
        "   ldr q10, [x20], #16             \n"
        "   fmla v29.4s, v11.4s, v7.s[3]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

		".macro KERNEL_M5xN16_LAST_K        \n"
		"	prfm PLDL1STRM, [x9]            \n"

		"   ld1r {v2.4s}, [x6]              \n"
		
		"	fmla v12.4s, v8.4s, v5.s[3]     \n" // a0
		"	fmla v17.4s, v9.4s, v5.s[3]     \n"
		"	ldr q3, [x27]                   \n"
		"	ldr q4, [x27, #16]              \n"
		"	fmla v22.4s, v10.4s, v5.s[3]    \n"
		"	fmla v27.4s, v11.4s, v5.s[3]    \n"
		"	ldr q5, [x27, #32]              \n"

		"	fmul v12.4s, v12.4s, v2.s[0]    \n"
		"	fmul v17.4s, v17.4s, v2.s[0]    \n"
		"	fmul v22.4s, v22.4s, v2.s[0]    \n"
		"	fmul v27.4s, v27.4s, v2.s[0]    \n"

		"	fadd v12.4s, v12.4s, v3.4s      \n"
		"	ldr q3, [x27, #48]              \n"
		"	add x27, x27, x11               \n"
		"	fadd v17.4s, v17.4s, v4.4s      \n"
		"	ld1r {v4.4s}, [x9]              \n"
		"	fadd v22.4s, v22.4s, v5.4s      \n"
		"	fadd v27.4s, v27.4s, v3.4s      \n"
		"	str q12, [x24]                  \n"
		"	str q17, [x24, #80]             \n"

		"	fmax v12.4s, v12.4s, v17.4s     \n"
		"	str q22, [x24, #160]            \n"
		"	str q27, [x24, #240]            \n"
		"	fmax v22.4s, v22.4s, v27.4s     \n"
		"	fmax v12.4s, v12.4s, v4.4s      \n"
		"	fmax v12.4s, v12.4s, v22.4s     \n"
        "   fmaxv s12, v12.4s               \n"
		"	str s12, [x9], #4               \n"

		"	fmla v13.4s, v8.4s, v6.s[3]     \n" // a1
		"	fmla v18.4s, v9.4s, v6.s[3]     \n"
		"	ldr q3, [x27]                   \n"
		"	ldr q4, [x27, #16]              \n"
		"	fmla v23.4s, v10.4s, v6.s[3]    \n"
		"	fmla v28.4s, v11.4s, v6.s[3]    \n"
		"	ldr q5, [x27, #32]              \n"
		"	ldr q6, [x27, #48]              \n"
		"	add x27, x27, x11               \n"

		"	fmul v13.4s, v13.4s, v2.s[0]    \n"
		"	fmul v18.4s, v18.4s, v2.s[0]    \n"
		"	fmul v23.4s, v23.4s, v2.s[0]    \n"
		"	fmul v28.4s, v28.4s, v2.s[0]    \n"

		"	fadd v13.4s, v13.4s, v3.4s      \n"
		"	ld1r {v3.4s}, [x9]              \n"
		"	fadd v18.4s, v18.4s, v4.4s      \n"
		"	fadd v23.4s, v23.4s, v5.4s      \n"
		"	fadd v28.4s, v28.4s, v6.4s      \n"
		"	str q13, [x25]                  \n"
		"	str q18, [x25, #80]             \n"

		"	fmax v13.4s, v13.4s, v18.4s     \n"
		"	str q23, [x25, #160]            \n"
		"	str q28, [x25, #240]            \n"
		"	fmax v23.4s, v23.4s, v28.4s     \n"
		"	fmax v13.4s, v13.4s, v3.4s      \n"
		"	fmax v13.4s, v13.4s, v23.4s     \n"
        "   fmaxv s13, v13.4s               \n"
		"	str s13, [x9], #4               \n"

		"   fmla v14.4s, v8.4s, v7.s[3]  	\n" // a2
		"	ldr q3, [x27]                   \n"
		"	ldr q4, [x27, #16]              \n"
		"   fmla v19.4s, v9.4s, v7.s[3]     \n"
		"   fmla v24.4s, v10.4s, v7.s[3]    \n"
		"   fmla v29.4s, v11.4s, v7.s[3]    \n"
		"	ldr q5, [x27, #32]              \n"
		"	ldr q6, [x27, #48]              \n"
		"	add x27, x27, x11               \n"

		"	fmul v14.4s, v14.4s, v2.s[0]    \n"
		"	fmul v19.4s, v19.4s, v2.s[0]    \n"
		"	fmul v24.4s, v24.4s, v2.s[0]    \n"
		"	fmul v29.4s, v29.4s, v2.s[0]    \n"

		"	fadd v14.4s, v14.4s, v3.4s      \n"
		"	ld1r {v3.4s}, [x9]              \n"
		"	fadd v19.4s, v19.4s, v4.4s      \n"
		"	fadd v24.4s, v24.4s, v5.4s      \n"
		"	fadd v29.4s, v29.4s, v6.4s      \n"
		"	str q14, [x26]                  \n"
		"	str q19, [x26, #80]             \n"

		"	fmax v14.4s, v14.4s, v19.4s     \n"
		"	str q24, [x26, #160]            \n"
		"	str q29, [x26, #240]            \n"
		"	fmax v24.4s, v24.4s, v29.4s     \n"
		"	fmax v14.4s, v14.4s, v3.4s      \n"
		"	fmax v14.4s, v14.4s, v24.4s     \n"
        "   fmaxv s14, v14.4s               \n"
		"	str s14, [x9], #4               \n"

		"	fmla v15.4s, v8.4s, v0.s[3]     \n" // a3
		"	ldr q3, [x27]                   \n"
		"	ldr q4, [x27, #16]              \n"
		"	fmla v20.4s, v9.4s, v0.s[3]     \n"
		"	fmla v25.4s, v10.4s, v0.s[3]    \n"
		"	fmla v30.4s, v11.4s, v0.s[3]    \n"
		"	ldr q5, [x27, #32]              \n"
		"	ldr q6, [x27, #48]              \n"
		"	add x27, x27, x11               \n"

		"	fmul v15.4s, v15.4s, v2.s[0]    \n"
		"	fmul v20.4s, v20.4s, v2.s[0]    \n"
		"	fmul v25.4s, v25.4s, v2.s[0]    \n"
		"	fmul v30.4s, v30.4s, v2.s[0]    \n"

		"	fadd v15.4s, v15.4s, v3.4s      \n"
		"	ld1r {v3.4s}, [x9]              \n"
		"	fadd v20.4s, v20.4s, v4.4s      \n"
		"	fadd v25.4s, v25.4s, v5.4s      \n"
		"	fadd v30.4s, v30.4s, v6.4s      \n"
		"	str q15, [x21]                  \n"
		"	str q20, [x21, #80]             \n"

		"	fmax v15.4s, v15.4s, v20.4s     \n"
		"	str q25, [x21, #160]            \n"
		"	str q30, [x21, #240]            \n"
		"	fmax v25.4s, v25.4s, v30.4s     \n"
		"	fmax v15.4s, v15.4s, v3.4s      \n"
		"	fmax v15.4s, v15.4s, v25.4s     \n"
        "   fmaxv s15, v15.4s               \n"
		"	str s15, [x9], #4               \n"

		"	fmla v16.4s, v8.4s, v1.s[3]     \n" // a4
		"	ldr q3, [x27]                   \n"
		"	ldr q4, [x27, #16]              \n"
		"	fmla v21.4s, v9.4s, v1.s[3]     \n"
		"	fmla v26.4s, v10.4s, v1.s[3]    \n"
		"	fmla v31.4s, v11.4s, v1.s[3]    \n"
		"	ldr q5, [x27, #32]              \n"
		"	ldr q6, [x27, #48]              \n"

		"	fmul v16.4s, v16.4s, v2.s[0]    \n"
		"	fmul v21.4s, v21.4s, v2.s[0]    \n"
		"	fmul v26.4s, v26.4s, v2.s[0]    \n"
		"	fmul v31.4s, v31.4s, v2.s[0]    \n"

		"	fadd v16.4s, v16.4s, v3.4s      \n"
		"	ld1r {v3.4s}, [x9]              \n"
		"	fadd v21.4s, v21.4s, v4.4s      \n"
		"	fadd v26.4s, v26.4s, v5.4s      \n"
		"	fadd v31.4s, v31.4s, v6.4s      \n"
		"	str q16, [x28]                  \n"
		"	str q21, [x28, #80]             \n"

		"	fmax v16.4s, v16.4s, v21.4s     \n"
		"	str q26, [x28, #160]            \n"
		"	str q31, [x28, #240]            \n"
		"	fmax v26.4s, v26.4s, v31.4s     \n"
		"	fmax v16.4s, v16.4s, v3.4s      \n"
		"	fmax v16.4s, v16.4s, v26.4s     \n"
        "   fmaxv s16, v16.4s               \n"
		"	str s16, [x9]                   \n"
        "   sub x9, x9, #16                 \n"
		
		".endm                              \n"

// -------------------------------------------------------
        ".macro KERNEL_EDGE_M5xN4_FIRST_K   \n"
        "   ldp q8, q9, [x20], #32          \n"
        "   ldr q0, [x15], #16              \n"
        "   ldr q1, [x16], #16              \n"

        "   fmul v12.4s, v8.4s, v0.s[0]     \n"
        "   fmul v13.4s, v8.4s, v1.s[0]     \n"
        "   ldr q2, [x17], #16              \n"
        "   ldr q3, [x18], #16              \n"
        "   fmul v17.4s, v9.4s, v0.s[1]     \n"
        "   fmul v18.4s, v9.4s, v1.s[1]     \n"
        "   ldr q4, [x19], #16              \n"
        "   ldp q10, q11, [x20], #32        \n"
        "   fmul v14.4s, v8.4s, v2.s[0]     \n"
        "   fmul v15.4s, v8.4s, v3.s[0]     \n"
        "   ldr q5, [x15], #16              \n"
        "   ldr q6, [x16], #16              \n"
        "   fmul v19.4s, v9.4s, v2.s[1]     \n"
        "   fmul v20.4s, v9.4s, v3.s[1]     \n"
        "   fmul v16.4s, v8.4s, v4.s[0]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmul v21.4s, v9.4s, v4.s[1]     \n"
        "   ldr q9, [x20], #16              \n"

        "   ldr q7, [x17], #16              \n"
        "   fmul v22.4s, v10.4s, v0.s[2]    \n"
        "   fmul v27.4s, v11.4s, v0.s[3]    \n"
        "   ldr q0, [x18], #16              \n"
        "   fmul v23.4s, v10.4s, v1.s[2]    \n"
        "   fmul v28.4s, v11.4s, v1.s[3]    \n"
        "   ldr q1, [x19], #16              \n"
        "   fmul v24.4s, v10.4s, v2.s[2]    \n"
        "   fmul v25.4s, v10.4s, v3.s[2]    \n"
        "   fmul v26.4s, v10.4s, v4.s[2]    \n"
        "   ldr q10, [x20], #16             \n"

        "   fmul v29.4s, v11.4s, v2.s[3]    \n"
        "   fmul v30.4s, v11.4s, v3.s[3]    \n"
        "   fmul v31.4s, v11.4s, v4.s[3]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_EDGE_M5xN4_PONG_K8   \n"

        "   fmla v12.4s, v8.4s, v5.s[0]     \n"
        "   fmla v13.4s, v8.4s, v6.s[0]     \n"
        "   fmla v17.4s, v9.4s, v5.s[1]     \n"
        "   fmla v18.4s, v9.4s, v6.s[1]     \n"
        "   ldr q2, [x17], #16              \n"
        "   ldr q3, [x18], #16              \n"
        "   fmla v14.4s, v8.4s, v7.s[0]     \n"
        "   fmla v15.4s, v8.4s, v0.s[0]     \n"
        "   ldr q4, [x19], #16              \n"
        "   fmla v19.4s, v9.4s, v7.s[1]     \n"
        "   fmla v20.4s, v9.4s, v0.s[1]     \n"
        "   fmla v16.4s, v8.4s, v1.s[0]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v1.s[1]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v25.4s, v10.4s, v0.s[2]    \n"
        "   fmla v30.4s, v11.4s, v0.s[3]    \n"
        "   ldr q0, [x15], #16              \n"
        "   fmla v26.4s, v10.4s, v1.s[2]    \n"
        "   fmla v31.4s, v11.4s, v1.s[3]    \n"
        "   ldr q1, [x16], #16              \n"
        "   fmla v22.4s, v10.4s, v5.s[2]    \n"
        "   fmla v23.4s, v10.4s, v6.s[2]    \n"
        "   fmla v24.4s, v10.4s, v7.s[2]    \n"
        "   ldr q10, [x20], #16             \n"

        "   fmla v27.4s, v11.4s, v5.s[3]    \n"
        "   fmla v28.4s, v11.4s, v6.s[3]    \n"
        "   fmla v29.4s, v11.4s, v7.s[3]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_EDGE_M5xN4_PING_K8   \n"

        "   fmla v12.4s, v8.4s, v0.s[0]     \n"
        "   fmla v13.4s, v8.4s, v1.s[0]     \n"
        "   fmla v17.4s, v9.4s, v0.s[1]     \n"
        "   fmla v18.4s, v9.4s, v1.s[1]     \n"
        "   ldr q5, [x15], #16              \n"
        "   ldr q6, [x16], #16              \n"
        "   fmla v14.4s, v8.4s, v2.s[0]     \n"
        "   fmla v15.4s, v8.4s, v3.s[0]     \n"
        "   ldr q7, [x17], #16              \n"
        "   fmla v19.4s, v9.4s, v2.s[1]     \n"
        "   fmla v20.4s, v9.4s, v3.s[1]     \n"
        "   fmla v16.4s, v8.4s, v4.s[0]     \n"
        "   ldr q8, [x20], #16              \n"
        "   fmla v21.4s, v9.4s, v4.s[1]     \n"
        "   ldr q9, [x20], #16              \n"

        "   fmla v22.4s, v10.4s, v0.s[2]    \n"
        "   fmla v27.4s, v11.4s, v0.s[3]    \n"
        "   ldr q0, [x18], #16              \n"
        "   fmla v23.4s, v10.4s, v1.s[2]    \n"
        "   fmla v28.4s, v11.4s, v1.s[3]    \n"
        "   ldr q1, [x19], #16              \n"
        "   fmla v24.4s, v10.4s, v2.s[2]    \n"
        "   fmla v25.4s, v10.4s, v3.s[2]    \n"
        "   fmla v26.4s, v10.4s, v4.s[2]    \n"
        "   ldr q10, [x20], #16             \n"

        "   fmla v29.4s, v11.4s, v2.s[3]    \n"
        "   fmla v30.4s, v11.4s, v3.s[3]    \n"
        "   fmla v31.4s, v11.4s, v4.s[3]    \n"
        "   ldr q11, [x20], #16             \n"
        ".endm                              \n"

        ".macro KERNEL_EDGE_M5xN4_LAST_K    \n"
        "   ld1r {v2.4s}, [x6]              \n"

        "   ldr q3, [x27]                   \n"
        "   add x27, x27, x11               \n"
        "   ld1r {v4.4s}, [x9]              \n"
        "   fmla v12.4s, v8.4s, v5.s[0]     \n"
        "   fmla v17.4s, v9.4s, v5.s[1]     \n"
        "   fadd v12.4s, v12.4s, v17.4s     \n"
        "   ldr q17, [x27]                  \n"
        "   add x27, x27, x11               \n"
        "   fmla v22.4s, v10.4s, v5.s[2]    \n"
        "   fmla v27.4s, v11.4s, v5.s[3]    \n"
        "   fadd v22.4s, v22.4s, v27.4s     \n"
        "   fadd v12.4s, v12.4s, v22.4s     \n"
        "   fmul v12.4s, v12.4s, v2.s[0]    \n"
        "   fadd v12.4s, v12.4s, v3.4s      \n"
        "   str q12, [x24]                  \n"
        "   fmax v12.4s, v12.4s, v4.4s      \n"
        "   fmaxv s12, v12.4s               \n"
        "   str s12, [x9], #4               \n"

        "   ld1r {v22.4s}, [x9]             \n"
        "   fmla v13.4s, v8.4s, v6.s[0]     \n"
        "   fmla v18.4s, v9.4s, v6.s[1]     \n"
        "   fadd v13.4s, v13.4s, v18.4s     \n"
        "   ldr q18, [x27]                  \n"
        "   add x27, x27, x11               \n"
        "   fmla v23.4s, v10.4s, v6.s[2]    \n"
        "   fmla v28.4s, v11.4s, v6.s[3]    \n"
        "   fadd v23.4s, v23.4s, v28.4s     \n"
        "   fadd v13.4s, v13.4s, v23.4s     \n"
        "   fmul v13.4s, v13.4s, v2.s[0]    \n"
        "   fadd v13.4s, v13.4s, v17.4s     \n"
        "   str q13, [x25]                  \n"
        "   fmax v13.4s, v13.4s, v22.4s     \n"
        "   fmaxv s13, v13.4s               \n"
        "   str s13, [x9], #4               \n"

        "   ld1r {v23.4s}, [x9]             \n"
        "   fmla v14.4s, v8.4s, v7.s[0]     \n"
        "   fmla v19.4s, v9.4s, v7.s[1]     \n"
        "   fadd v14.4s, v14.4s, v19.4s     \n"
        "   ldr q19, [x27]                  \n"
        "   add x27, x27, x11               \n"
        "   fmla v24.4s, v10.4s, v7.s[2]    \n"
        "   fmla v29.4s, v11.4s, v7.s[3]    \n"
        "   fadd v24.4s, v24.4s, v29.4s     \n"
        "   fadd v14.4s, v14.4s, v24.4s     \n"
        "   fmul v14.4s, v14.4s, v2.s[0]    \n"
        "   fadd v14.4s, v14.4s, v18.4s     \n"
        "   str q14, [x26]                  \n"
        "   fmax v14.4s, v14.4s, v23.4s     \n"
        "   fmaxv s14, v14.4s               \n"
        "   str s14, [x9], #4               \n"

        "   ld1r {v24.4s}, [x9]             \n"
        "   fmla v15.4s, v8.4s, v0.s[0]     \n"
        "   fmla v20.4s, v9.4s, v0.s[1]     \n"
        "   fadd v15.4s, v15.4s, v20.4s     \n"
        "   ldr q20, [x27]                  \n"
        "   fmla v25.4s, v10.4s, v0.s[2]    \n"
        "   fmla v30.4s, v11.4s, v0.s[3]    \n"
        "   fadd v25.4s, v25.4s, v30.4s     \n"
        "   fadd v15.4s, v15.4s, v25.4s     \n"
        "   fmul v15.4s, v15.4s, v2.s[0]    \n"
        "   fadd v15.4s, v15.4s, v19.4s     \n"
        "   str q15, [x21]                  \n"
        "   fmax v15.4s, v15.4s, v24.4s     \n"
        "   fmaxv s15, v15.4s               \n"
        "   str s15, [x9], #4               \n"

        "   ld1r {v25.4s}, [x9]             \n"
        "   fmla v16.4s, v8.4s, v1.s[0]     \n"
        "   fmla v21.4s, v9.4s, v1.s[1]     \n"
        "   fadd v16.4s, v16.4s, v21.4s     \n"
        "   fmla v26.4s, v10.4s, v1.s[2]    \n"
        "   fmla v31.4s, v11.4s, v1.s[3]    \n"
        "   fadd v26.4s, v26.4s, v31.4s     \n"
        "   fadd v16.4s, v16.4s, v26.4s     \n"
        "   fmul v16.4s, v16.4s, v2.s[0]    \n"
        "   fadd v16.4s, v16.4s, v20.4s     \n"
        "   str q16, [x28]                  \n"
        "   fmax v16.4s, v16.4s, v25.4s     \n"
        "   fmaxv s16, v16.4s               \n"
        "   str s16, [x9]                   \n"
        "   sub x9, x9, #16                 \n"
        ".endm                              \n"

// -------------------------------- ENTRY --------------------------------
        "ENTRY:                             \n"
        "   ldr x0,  %[q]                   \n"
        "   ldr x1,  %[kt]                  \n"
        "   ldr x2,  %[score]               \n"
        "   ldr x3,  %[m]                   \n"
        "   ldr x4,  %[n]                   \n"
        "   ldr x5,  %[k]                   \n"
        "   ldr x6,  %[scale]               \n"
        "   ldr x7,  %[buffer_kt]           \n"
        "   ldr x8,  %[lds_in_bytes]        \n"
        "   ldr x9,  %[max_per_line]        \n"
        "   ldr x10, %[mask]                \n"
        "   ldr x11, %[ldm_in_bytes]        \n"

        "   lsr x29, x4, #4                 \n" // n/16
        "   mov x12, #16                    \n"
        "   msub x4, x29, x12, x4           \n" // n%16

// -------------------------------- N16 --------------------------------
        "COMPUTE_N16:                       \n"
        "   cmp x29, #0                     \n"
        "   beq COMPUTE_N4                  \n"

        "   mov x30, #5                     \n" // m/5
        "   udiv x30, x3, x30               \n"

        // -------- x12(pack), x13=4, x14=k/8, x27(mask)
        "PACK_M5xN4_x4:                     \n"
        "   mov x13, #4                     \n"

        "   mov x24, x2                     \n" // score(panel format)
        "   add x25, x24, #16               \n"
        "   add x26, x25, #16               \n"
        "   add x28, x26, #32               \n"
        "   add x2, x2, #320                \n" // r1*r2*sizeof(float)

        "PACK_M5xN4_START:                  \n"
        "   mov x15, x0                     \n" // q(row major)
        "   add x16, x15, x5, lsl #2        \n" 
        "   add x17, x16, x5, lsl #2        \n"
        "   add x18, x17, x5, lsl #2        \n"
        "   add x19, x18, x5, lsl #2        \n"

        "   mov x20, x1                     \n"
        "   add x21, x20, x5, lsl #2        \n" // kt(col major)
        "   add x22, x21, x5, lsl #2        \n"
        "   add x23, x22, x5, lsl #2        \n"
        "   add x1, x1, x5, lsl #4          \n"

        "   mov x12, x7                     \n" // buffer_kt
        "   add x7, x7, #16                 \n"

		"	mov x27, x10                    \n" // mask
		"	add x10, x10, #16               \n"

        "   lsr x14, x5, #3                 \n" // loop
        "   PACK_KERNEL_M5xN4_FIRST_K4      \n"
        "   subs x14, x14, #1               \n"
        "   beq PACK_M5xN4_FINISH           \n"

        "PACK_M5xN4_PONG_PING:              \n"
        "   PACK_KERNEL_M5xN4_PONG_K4       \n"
        "   PACK_KERNEL_M5xN4_PING_K4       \n"
        "   subs x14, x14, #1               \n"
        "   bne PACK_M5xN4_PONG_PING        \n"

        "PACK_M5xN4_FINISH:                 \n"
        "   PACK_KERNEL_M5xN4_LAST_K4       \n"

        "   subs x13, x13, #1               \n"
        "   bne PACK_M5xN4_START            \n"

        "POST_PACK_M5xN4_x4:                \n"
        "   sub x24, x24, #320              \n"
        "   ldr x7, %[buffer_kt]            \n" // buffer_kt
        "	sub x27, x27, #48               \n" // mask

        "   subs x30, x30, #1               \n"
        "   beq POST_COMPUTE_N16            \n"

        // -------- x14=k/8, x27(mask)
        "KERNEL_M5xN16:                     \n"
        "   add x0, x0, x5, lsl #4          \n"
        "   add x0, x0, x5, lsl #2          \n"
        "   mov x15, x0                     \n" // q(row major)
        "   add x16, x15, x5, lsl #2        \n"
        "   add x17, x16, x5, lsl #2        \n"
        "   add x18, x17, x5, lsl #2        \n"
        "   add x19, x18, x5, lsl #2        \n"

        "   add x24, x24, x8, lsl #2        \n"
        "   add x24, x24, x8                \n" // score(panel format)
        "   add x25, x24, #16               \n"
        "   add x26, x25, #16               \n"
        "   add x21, x26, #16               \n"
        "   add x28, x21, #16               \n"

        "   mov x20, x7                     \n" // buffer_kt

		"	add x27, x27, x11               \n" // mask
        "   add x9, x9, #20                 \n" // max

        "KERNEL_M5xN16_START:               \n" // loop
        "   lsr x14, x5, #3                 \n"
        "   KERNEL_M5xN16_FIRST_K           \n"
        "   b KERNEL_M5xN16_LOOP            \n"

        "KERNEL_M5xN16_LOOP_START:          \n"
        "   KERNEL_M5xN16_K0                \n"
        
        "KERNEL_M5xN16_LOOP:                \n"
        "   KERNEL_M5xN16_K1                \n"
        "   KERNEL_M5xN16_K2                \n"
        "   KERNEL_M5xN16_K3                \n"
        "   KERNEL_M5xN16_K4                \n"
        "   KERNEL_M5xN16_K5                \n"
        "   KERNEL_M5xN16_K6                \n"

        "   subs x14, x14, #1               \n"
        "   beq KERNEL_M5xN16_FINISH        \n"
        "   KERNEL_M5xN16_K7                \n"

        "   b KERNEL_M5xN16_LOOP_START      \n"

        "KERNEL_M5xN16_FINISH:              \n"
        "   KERNEL_M5xN16_LAST_K            \n"

        "POST_KERNEL_M5xN16:                \n"
        "   subs x30, x30, #1               \n"
        "   bne KERNEL_M5xN16               \n"

        // -------- POST N16
        "POST_COMPUTE_N16:                  \n"
        "   ldr x0, %[q]                    \n"
        "   ldr x9, %[max_per_line]         \n"

        "   subs x29, x29, #1               \n"
        "   bne COMPUTE_N16                 \n"

// -------------------------------- N4 --------------------------------
        "COMPUTE_N4:                        \n"
        "   cmp x4, #0                      \n"
        "   beq COMPUTE_END                 \n"

        "   mov x30, #5                     \n" // m/5
        "   udiv x30, x3, x30               \n"

        // -------- x12(pack), x14=k/8, x27(mask)
        "PACK_EDGE_M5xN4:                   \n"
        "   mov x24, x2                     \n" // score(panel format)
        "   add x25, x24, #16               \n"
        "   add x26, x25, #16               \n"
        "   add x28, x26, #32               \n"
        "   add x2, x2, #80                 \n"

        "PACK_EDGE_M5xN4_START:             \n"
        "   mov x15, x0                     \n" // q(row major)
        "   add x16, x15, x5, lsl #2        \n" 
        "   add x17, x16, x5, lsl #2        \n"
        "   add x18, x17, x5, lsl #2        \n"
        "   add x19, x18, x5, lsl #2        \n"

        "   mov x20, x1                     \n"
        "   add x21, x20, x5, lsl #2        \n" // kt(col major)
        "   add x22, x21, x5, lsl #2        \n"
        "   add x23, x22, x5, lsl #2        \n"
        "   add x1, x1, x5, lsl #4          \n"

        "   mov x12, x7                     \n" // buffer_kt

		"	mov x27, x10                    \n" // mask
        "   add x10, x10, #16               \n"

        "   lsr x14, x5, #3                 \n" // loop
        "   PACK_KERNEL_EDGE_M5xN4_FIRST_K4 \n"
        "   subs x14, x14, #1               \n"
        "   beq PACK_EDGE_M5xN4_FINISH      \n"

        "PACK_EDGE_M5xN4_PONG_PING:         \n"
        "   PACK_KERNEL_EDGE_M5xN4_PONG_K4  \n"
        "   PACK_KERNEL_EDGE_M5xN4_PING_K4  \n"

        "   subs x14, x14, #1               \n"
        "   bne PACK_EDGE_M5xN4_PONG_PING   \n"

        "PACK_EDGE_M5xN4_FINISH:            \n"
        "   PACK_KERNEL_EDGE_M5xN4_LAST_K4  \n"

        "POST_PACK_EDGE_M5xN4:              \n"
        "   sub x24, x24, #80               \n"

        "   subs x30, x30, #1               \n"
        "   beq POST_COMPUTE_N4             \n"

        // -------- x14=k/8, x27(mask)
        "KERNEL_EDGE_M5xN4:                 \n"
        "   add x0, x0, x5, lsl #4          \n"
        "   add x0, x0, x5, lsl #2          \n"
        "   mov x15, x0                     \n" // q(row major)
        "   add x16, x15, x5, lsl #2        \n"
        "   add x17, x16, x5, lsl #2        \n"
        "   add x18, x17, x5, lsl #2        \n"
        "   add x19, x18, x5, lsl #2        \n"

        "   add x24, x24, x8, lsl #2        \n"
        "   add x24, x24, x8                \n" // score(panel format)
        "   add x25, x24, #16               \n"
        "   add x26, x25, #16               \n"
        "   add x21, x26, #16               \n"
        "   add x28, x21, #16               \n"

        "   mov x20, x7                     \n" // buffer_kt

		"	add x27, x27, x11               \n" // mask
        "   add x9, x9, #20                 \n" // max

        "KERNEL_EDGE_M5xN4_START:           \n" // loop
        "   lsr x14, x5, #3                 \n"
        "   KERNEL_EDGE_M5xN4_FIRST_K       \n"

        "   subs x14, x14, #1               \n"
        "   beq KERNEL_EDGE_M5xN4_FINISH    \n"

        "KERNEL_EDGE_M5xN4_PONG_PING:       \n"
        "   KERNEL_EDGE_M5xN4_PONG_K8       \n"
        "   KERNEL_EDGE_M5xN4_PING_K8       \n"

        "   subs x14, x14, #1               \n"
        "   bne KERNEL_EDGE_M5xN4_PONG_PING \n"

        "KERNEL_EDGE_M5xN4_FINISH:          \n"
        "   KERNEL_EDGE_M5xN4_LAST_K        \n"

        "POST_KERNEL_EDGE_M5xN4:            \n"
        "   subs x30, x30, #1               \n"
        "   bne KERNEL_EDGE_M5xN4           \n"

        //  POST N4
        "POST_COMPUTE_N4:                   \n"
        "   ldr x0, %[q]                    \n"
        "   ldr x9, %[max_per_line]         \n"

        "   subs x4, x4, #4                 \n"
        "   bne COMPUTE_N4                  \n"

// -------------------------------- END ---------------------------------
        "COMPUTE_END:                       \n"
        "   nop                             \n"

        :
        : [q]              "m" (q),
          [kt]             "m" (kt),
          [score]          "m" (score),
          [m]              "m" (m),
          [n]              "m" (n),
          [k]              "m" (k),
          [scale]          "m" (scale),
          [buffer_kt]      "m" (buffer_kt),
          [lds_in_bytes]   "m" (lds_in_bytes),
          [max_per_line]   "m" (max_per_line),
          [mask]           "m" (mask),
          [ldm_in_bytes]   "m" (ldm_in_bytes)
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

