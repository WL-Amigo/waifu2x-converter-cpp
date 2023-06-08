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

#define HAVE_FMA
#define HAVE_AVX

typedef __m256 v256_t;

static inline __m256 madd256(__m256 v0, __m256 v1, __m256 v2)
{
	return _mm256_fmadd_ps(v0, v1, v2);
}

#define load_broadcast _mm256_broadcast_ss
#define load256 _mm256_load_ps
#define store256 _mm256_store_ps
#define add256 _mm256_add_ps
#define max256 _mm256_max_ps
#define min256 _mm256_min_ps
#define zero _mm256_setzero_ps
#define set1 _mm256_set1_ps
#define mul256 _mm256_mul_ps


static inline float hadd8(__m256 v)
{
	v = _mm256_hadd_ps(v, v);
	v = _mm256_hadd_ps(v, v);

	float v0 = _mm_cvtss_f32(_mm256_extractf128_ps(v,0));
	float v1 = _mm_cvtss_f32(_mm256_extractf128_ps(v,1));

	return v0 + v1;
}



#include "modelHandler_avx_func.hpp"


#undef UNROLL
typedef __m256 vreg_t;
#define VEC_NELEM 8
#ifdef __x86_64
#define UNROLL 5
#else
#define UNROLL 2
#endif

#define store_vreg(ptr,val) _mm256_store_ps((float*)(ptr), val)
#define load_vreg(ptr) _mm256_load_ps((float*)(ptr))
#define load_vreg_broadcast(ptr) _mm256_broadcast_ss((float*)(ptr))
#define madd_vreg _mm256_fmadd_ps // a*b + c
#define add_vreg _mm256_add_ps
#define zero_vreg _mm256_setzero_ps
#define min_vreg _mm256_min_ps
#define max_vreg _mm256_max_ps
#define set1_vreg _mm256_set1_ps

#define SIMD_OPLANE

#include "modelHandler_simd.hpp"

namespace w2xc
{
	void filter_FMA_impl
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
			filter_AVX_impl0(
			env,
			packed_input,
			packed_output,
			nInputPlanes,
			nOutputPlanes,
			fbiases,
			weight,
			ip_width,
			ip_height,
			nJob);
		}
	}
}
