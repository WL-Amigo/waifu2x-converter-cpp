/*
* The MIT License (MIT)
* This file is part of waifu2x-converter-cpp
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
#include <immintrin.h>
#include <atomic>
#include "filters.hpp"
#include "sec.hpp"

struct v256_t
{
	__m128 v0, v1;
};


static inline ALWAYS_INLINE v256_t madd256(v256_t const &v0, v256_t const &v1, v256_t const &v2)
{
	v256_t ret;
	ret.v0 = _mm_add_ps(_mm_mul_ps(v0.v0,v1.v0), v2.v0);
	ret.v1 = _mm_add_ps(_mm_mul_ps(v0.v1,v1.v1), v2.v1);
	return ret;
}

static inline v256_t load_broadcast(const float *p)
{
	v256_t ret;
	ret.v0 = _mm_set1_ps(p[0]);
	ret.v1 = _mm_set1_ps(p[0]);
	return ret;
}

static inline v256_t load256(const float *p)
{
	v256_t ret;
	ret.v0 = _mm_load_ps(p);
	ret.v1 = _mm_load_ps(p+4);
	return ret;
}


static inline void store256(float *p, v256_t const &v)
{
	_mm_storeu_ps(p, v.v0);
	_mm_storeu_ps(p+4, v.v1);
}

static inline v256_t zero()
{
	v256_t ret;
	ret.v0 = _mm_setzero_ps();
	ret.v1 = _mm_setzero_ps();
	return ret;
}

static inline v256_t set1(float a)
{
	v256_t ret;
	ret.v0 = _mm_set1_ps(a);
	ret.v1 = _mm_set1_ps(a);
	return ret;
}

static inline float hadd8(v256_t const &v)
{
	__m128 sum4 = _mm_add_ps(v.v0, v.v1);
	sum4 = _mm_hadd_ps(sum4, sum4);
	sum4 = _mm_hadd_ps(sum4, sum4);
	return _mm_cvtss_f32(sum4);
}

#define SSE_GEN_BINARY(func_name, intrin_name) \
static inline v256_t                           \
func_name(v256_t const &a, v256_t const &b)    \
{                                              \
	v256_t ret;                                \
	ret.v0 = intrin_name(a.v0, b.v0);          \
	ret.v1 = intrin_name(a.v1, b.v1);          \
	return ret;                                \
}

SSE_GEN_BINARY(add256, _mm_add_ps)
SSE_GEN_BINARY(mul256, _mm_mul_ps)
SSE_GEN_BINARY(max256, _mm_max_ps)
SSE_GEN_BINARY(min256, _mm_min_ps)

#include "modelHandler_avx_func.hpp"

#undef UNROLL

#define USE_SSE3
#ifdef __x86_64
#define UNROLL 4
#else
#define UNROLL 2
#endif

/* x86 SSE */
typedef __m128 vreg_t;
#define VEC_NELEM 4
#define store_vreg(ptr,val) _mm_store_ps((float*)(ptr), val)
#define load_vreg(ptr) _mm_load_ps((float*)(ptr))
    
static inline __m128 load_vreg_broadcast(const unsigned char *ptr)
{
    return _mm_set1_ps(*(float*)ptr);
}

static inline __m128 madd_vreg(__m128 a, __m128 b, __m128 c)
{
    return _mm_add_ps(_mm_mul_ps(a,b), c);
}

#define add_vreg _mm_add_ps
#define zero_vreg _mm_setzero_ps
#define min_vreg _mm_min_ps
#define max_vreg _mm_max_ps
#define set1_vreg _mm_set1_ps

#define SIMD_OPLANE

#include "modelHandler_simd.hpp"


namespace w2xc
{
	void filter_SSE_impl
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
