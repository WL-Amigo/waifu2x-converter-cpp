/*
* The MIT License (MIT)
* Copyright (c) 2015 amigo(white luckers), tanakamura, DeadSix27, YukihoAA and contributors
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <thread>
#include <atomic>
#include <arm_neon.h>
#include "filters.hpp"
#include "sec.hpp"

struct v256_t
{
	float32x4_t v0, v1;
};


static inline ALWAYS_INLINE v256_t madd256(v256_t const &v0, v256_t const &v1, v256_t const &v2)
{
	v256_t ret;
	ret.v0 = vmlaq_f32(v2.v0, v0.v0, v1.v0);
	ret.v1 = vmlaq_f32(v2.v1, v0.v1, v1.v1);
	return ret;
}

static inline v256_t load_broadcast(const float *p)
{
	v256_t ret;
	ret.v0 = vdupq_n_f32(p[0]);
	ret.v1 = vdupq_n_f32(p[0]);
	return ret;
}

static inline v256_t load256(const float *p)
{
	v256_t ret;
	ret.v0 = *(float32x4_t*)(p);
	ret.v1 = *(float32x4_t*)(p+4);
	return ret;
}


static inline void store256(float *p, v256_t const &v)
{
	*(float32x4_t*)(p+0) = v.v0;
	*(float32x4_t*)(p+4) = v.v1;
}

static inline v256_t zero()
{
	v256_t ret;
	ret.v0 = vdupq_n_f32(0);
	ret.v1 = vdupq_n_f32(0);
	return ret;
}

static inline v256_t set1(float a)
{
	v256_t ret;
	ret.v0 = vdupq_n_f32(a);
	ret.v1 = vdupq_n_f32(a);
	return ret;
}

static inline float hadd8(v256_t const &v)
{
	float32x4_t sum4 = vaddq_f32(v.v0, v.v1);
	float32x2_t hi = vget_high_f32(sum4);
	float32x2_t lo = vget_low_f32(sum4);
	float32x2_t a = vadd_f32(hi, lo);
	return vget_lane_f32(a,0) + vget_lane_f32(a,1);
}

#define NEON_GEN_BINARY(func_name, intrin_name) \
static inline v256_t	                        \
func_name(v256_t const &a, v256_t const &b)     \
{                                               \
	v256_t ret;                                 \
	ret.v0 = intrin_name(a.v0, b.v0);           \
	ret.v1 = intrin_name(a.v1, b.v1);           \
	return ret;                                 \
}

NEON_GEN_BINARY(add256, vaddq_f32)
NEON_GEN_BINARY(mul256, vmulq_f32)
NEON_GEN_BINARY(max256, vmaxq_f32)
NEON_GEN_BINARY(min256, vminq_f32)
#include "modelHandler_avx_func.hpp"

#undef UNROLL
#define UNROLL 4

/* arm neon */
typedef float32x4_t vreg_t;
#define VEC_NELEM 4
#define store_vreg(ptr,val) (*(float32x4_t*)(ptr))=(val)
#define load_vreg(ptr) (*(float32x4_t*)(ptr))
#define load_vreg_broadcast(ptr) vdupq_n_f32(*(float*)ptr)
#define madd_vreg(a,b,c) vmlaq_f32(c,a,b)

#define add_vreg vaddq_f32
#define zero_vreg() vdupq_n_f32(0)
#define min_vreg vminq_f32
#define max_vreg vmaxq_f32
#define set1_vreg vdupq_n_f32

#define SIMD_OPLANE

#include "modelHandler_simd.hpp"

namespace w2xc
{
	void filter_NEON_impl
	(
		ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *fbiases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	)
	{
		if (simd_available(nInputPlanes, nOutputPlanes))
		{
			filter_simd_impl0
			(
				env,
				packed_input,
				packed_output,
				nInputPlanes,
				nOutputPlanes,
				fbiases,
				weight,
				ip_width,
				ip_height,
				nJob
			);
		}
		else
		{
			filter_AVX_impl0
			(
				env,
				packed_input,
				packed_output,
				nInputPlanes,
				nOutputPlanes,
				fbiases,
				weight,
				ip_width,
				ip_height,
				nJob
			);
		}
	}
}
