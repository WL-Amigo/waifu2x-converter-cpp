#include <thread>
#include <immintrin.h>
#include <atomic>
#include "filters.hpp"
#include "sec.hpp"

#define HAVE_AVX

typedef __m256 v256_t;
static inline __m256
madd256(__m256 v0, __m256 v1, __m256 v2)
{
	return _mm256_add_ps(_mm256_mul_ps(v0, v1), v2);
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

static inline float
hadd8(__m256 v)
{
	v = _mm256_hadd_ps(v, v);
	v = _mm256_hadd_ps(v, v);

	float v0 = _mm_cvtss_f32(_mm256_extractf128_ps(v,0));
	float v1 = _mm_cvtss_f32(_mm256_extractf128_ps(v,1));

	return v0 + v1;
}

#include "modelHandler_avx_func.hpp"

namespace w2xc {
void
filter_AVX_impl(ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *fbiases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob)
{
	filter_AVX_impl0(env,
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
