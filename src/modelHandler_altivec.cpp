#include <thread>
#include <atomic>
#include <altivec.h>
#include "filters.hpp"
#include "sec.hpp"

struct v256_t {
	vector float v0, v1;
};

static inline v256_t
madd256(v256_t const &v0, v256_t const &v1, v256_t const &v2)
{
	v256_t ret;
	ret.v0 = vec_madd(v0.v0, v1.v0, v2.v0);
	ret.v1 = vec_madd(v0.v1, v1.v1, v2.v1);
	return ret;
}

static inline v256_t
load_broadcast(const float *p)
{
	v256_t ret;
	ret.v0 = vec_splats(p[0]);
	ret.v1 = vec_splats(p[0]);
	return ret;
}

static inline v256_t
load256(const float *p)
{
	v256_t ret;
	ret.v0 = vec_ld(0, p);
	ret.v1 = vec_ld(16, p);
	return ret;
}

static inline void
store256(float *p, v256_t const &v)
{
	vec_st(v.v0, 0, p);
	vec_st(v.v1, 16, p);
}

static inline v256_t
zero()
{
	v256_t ret;
	ret.v0 = vec_splats(0.f);
	ret.v1 = vec_splats(0.f);
	return ret;
}

static inline v256_t
set1(float a)
{
	v256_t ret;
	ret.v0 = vec_splats(a);
	ret.v1 = vec_splats(a);
	return ret;
}

static inline float
hadd8(v256_t const &v)
{
	vector float sum2, sum4, sum8;
	sum2 = vec_add(v.v0, v.v1);
	sum4 = vec_add(sum2, vec_sld(sum2, sum2, 4));
	sum8 = vec_add(sum4, vec_sld(sum4, sum4, 8));

	return vec_extract(sum8, 0);
}

static inline v256_t
mul256(v256_t const &a, v256_t const &b)
{
	v256_t ret;
	ret.v0 = vec_madd(a.v0, b.v0, vec_splats(0.f));
	ret.v1 = vec_madd(a.v1, b.v1, vec_splats(0.f));
	return ret;
}

static inline v256_t
add256(v256_t const &a, v256_t const &b)
{
	v256_t ret;
	ret.v0 = vec_add(a.v0, b.v0);
	ret.v1 = vec_add(a.v1, b.v1);
	return ret;
}

static inline v256_t
max256(v256_t const &a, v256_t const &b)
{
	v256_t ret;
	ret.v0 = vec_max(a.v0, b.v0);
	ret.v1 = vec_max(a.v1, b.v1);
	return ret;
}

static inline v256_t
min256(v256_t const &a, v256_t const &b)
{
	v256_t ret;
	ret.v0 = vec_min(a.v0, b.v0);
	ret.v1 = vec_min(a.v1, b.v1);
	return ret;
}

#include "modelHandler_avx_func.hpp"

#undef UNROLL
#define UNROLL 4

/* PowerPC AltiVec */
#define VPACK 1
#define VEC_NELEM 8

/* Use v256_t as vreg_t so that the compiler will utilize the large amount of
 * registers available.
 *
 * Note: GCC8 seems to need the struct/array wrapping to generate well-performing code,
 *       so I'm keeping it. */
typedef struct {
	v256_t v[VPACK];
} vreg_t;

static inline vreg_t
madd_vreg(vreg_t const &a, vreg_t const &b, vreg_t const &c)
{
	vreg_t ret;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = madd256(a.v[i], b.v[i], c.v[i]);
	}
	return ret;
}

static inline vreg_t
add_vreg(vreg_t const &a, vreg_t const &b)
{
	vreg_t ret;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = add256(a.v[i], b.v[i]);
	}
	return ret;
}

static inline vreg_t
zero_vreg()
{
	vreg_t ret;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = zero();
	}
	return ret;
}

static inline vreg_t
min_vreg(vreg_t const &a, vreg_t const &b)
{
	vreg_t ret;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = min256(a.v[i], b.v[i]);
	}
	return ret;
}

static inline vreg_t
max_vreg(vreg_t const &a, vreg_t const &b)
{
	vreg_t ret;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = max256(a.v[i], b.v[i]);
	}
	return ret;
}

static inline vreg_t
set1_vreg(float a)
{
	vreg_t ret;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = set1(a);
	}
	return ret;
}

static inline void
store_vreg(const unsigned char *ptr, vreg_t const &val)
{
	float *p = (float*) ptr;
	for(int i = 0; i < VPACK; i++) {
		store256(p+i*8, val.v[i]);
	}
}

static inline vreg_t
load_vreg(const unsigned char *ptr)
{
	vreg_t ret;
	float *p = (float*) ptr;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = load256(p+i*8);
	}
	return ret;
}

static inline vreg_t
load_vreg_broadcast(const unsigned char *ptr)
{
	vreg_t ret;
	float *p = (float*) ptr;
	for(int i = 0; i < VPACK; i++) {
		ret.v[i] = load_broadcast(p);
	}
	return ret;
}

#undef VPACK

#define SIMD_OPLANE

#include "modelHandler_simd.hpp"


namespace w2xc {
void
filter_AltiVec_impl(ComputeEnv *env,
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
	if (simd_available(nInputPlanes, nOutputPlanes)) {
		filter_simd_impl0(env,
				packed_input,
				packed_output,
				nInputPlanes,
				nOutputPlanes,
				fbiases,
				weight,
				ip_width,
				ip_height,
				nJob);
	} else {
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
}
