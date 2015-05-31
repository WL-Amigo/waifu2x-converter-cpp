#include <thread>
#include <immintrin.h>
#include "modelHandler.hpp"
#include "common.hpp"
#include "sec.hpp"

template <bool border> inline
float
get_data(const float *p, int hsz, int wsz, int step, int yi, int xi, int num_plane, int plane)
{
	if (border) {
		yi = (std::min)(hsz-1, yi);
		yi = (std::max)(0, yi);

		xi = (std::min)(wsz-1, xi);
		xi = (std::max)(0, xi);

		char *p1 = (char*)p;
		return ((float*)(p1 + yi*step))[xi*num_plane + plane];
	} else {
		char *p1 = (char*)p;
		return ((float*)(p1 + yi*step))[xi*num_plane + plane];
	}
}

template <bool border> void
filter_1elem(const float *packed_input,
	     int nInputPlanes,
	     float *packed_output,
	     int nOutputPlanes,
	     const float *biases,
	     unsigned long hsz,
	     unsigned long wsz,
	     unsigned long yi,
	     unsigned long xi,
	     const float *weight,
	     float *intermediate)
{
	const float *in = packed_input;
	size_t in_step = wsz * sizeof(float) * nInputPlanes;

	for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {

#if 0
		float i00 = get_data<border>(in, hsz, wsz, in_step, yi-1, xi-1, ipIndex);
		float i01 = get_data<border>(in, hsz, wsz, in_step, yi-1, xi  , ipIndex);
		float i02 = get_data<border>(in, hsz, wsz, in_step, yi-1, xi+1, ipIndex);

		float i10 = get_data<border>(in, hsz, wsz, in_step, yi  , xi-1, ipIndex);
		float i11 = get_data<border>(in, hsz, wsz, in_step, yi  , xi  , ipIndex);
		float i12 = get_data<border>(in, hsz, wsz, in_step, yi  , xi+1, ipIndex);

		float i20 = get_data<border>(in, hsz, wsz, in_step, yi+1, xi-1, ipIndex);
		float i21 = get_data<border>(in, hsz, wsz, in_step, yi+1, xi  , ipIndex);
		float i22 = get_data<border>(in, hsz, wsz, in_step, yi+1, xi+1, ipIndex);

		float *w_base = weight + (ipIndex * nOutputPlanes) * 9;

		for (unsigned int opIndex = 0;
		     opIndex < (unsigned int)nOutputPlanes;
		     opIndex ++)
		{
			int oi_0 = opIndex % VEC_WIDTH;
			int oi_1 = (opIndex / VEC_WIDTH) * VEC_WIDTH;

			float *w = w_base + oi_1*9 + oi_0;
			float v = 0;

			v += w[0*VEC_WIDTH] * i00;
			v += w[1*VEC_WIDTH] * i01;
			v += w[2*VEC_WIDTH] * i02;

			v += w[3*VEC_WIDTH] * i10;
			v += w[4*VEC_WIDTH] * i11;
			v += w[5*VEC_WIDTH] * i12;

			v += w[6*VEC_WIDTH] * i20;
			v += w[7*VEC_WIDTH] * i21;
			v += w[8*VEC_WIDTH] * i22;

			if (ipIndex == 0) {
				intermediate[opIndex] = v;
			} else {
				intermediate[opIndex] += v;
			}
		}

#else
		__m256 i00 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi-1, xi-1, nInputPlanes, ipIndex));
		__m256 i01 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi-1, xi  , nInputPlanes, ipIndex));
		__m256 i02 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi-1, xi+1, nInputPlanes, ipIndex));

		__m256 i10 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi  , xi-1, nInputPlanes, ipIndex));
		__m256 i11 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi  , xi  , nInputPlanes, ipIndex));
		__m256 i12 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi  , xi+1, nInputPlanes, ipIndex));

		__m256 i20 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi+1, xi-1, nInputPlanes, ipIndex));
		__m256 i21 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi+1, xi  , nInputPlanes, ipIndex));
		__m256 i22 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi+1, xi+1, nInputPlanes, ipIndex));

		const float *w = weight + (ipIndex * nOutputPlanes) * 9;

		for (unsigned int opIndex = 0;
		     opIndex < (unsigned int)nOutputPlanes;
		     opIndex += VEC_WIDTH*UNROLL)
		{
#define APPLY_FILTER(e,off)						\
			__m256 v##e;					\
			v##e = _mm256_setzero_ps();		\
									\
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i00, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i01, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i02, v##e); \
									\
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i10, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i11, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i12, v##e); \
									\
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i20, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i21, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i22, v##e); \
									\
			w += 9 * VEC_WIDTH;				\
									\

			APPLY_FILTER(0, 0);
			APPLY_FILTER(1, 8);

			if (ipIndex == 0) {
				_mm256_storeu_ps(&intermediate[opIndex+0], v0);
				_mm256_storeu_ps(&intermediate[opIndex+8], v1);
			} else {					\
				__m256 prev0 = _mm256_loadu_ps(&intermediate[opIndex+0]);
				__m256 prev1 = _mm256_loadu_ps(&intermediate[opIndex+8]);
				_mm256_storeu_ps(&intermediate[opIndex+0], _mm256_add_ps(prev0,v0));
				_mm256_storeu_ps(&intermediate[opIndex+8], _mm256_add_ps(prev1,v1));
			}
		}
#endif
	}

	float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
	for (int opIndex = 0; opIndex < nOutputPlanes; opIndex+=VEC_WIDTH) {
		__m256 bv = _mm256_loadu_ps(&biases[opIndex]);
		__m256 v = _mm256_loadu_ps(&intermediate[opIndex]);
		v = _mm256_add_ps(v, bv);
		__m256 mtz = _mm256_max_ps(v, _mm256_setzero_ps());
		__m256 ltz = _mm256_min_ps(v, _mm256_setzero_ps());

		v = _mm256_add_ps(_mm256_mul_ps(ltz, _mm256_set1_ps(0.1f)), mtz);

		_mm256_storeu_ps(&out[opIndex], v);
	}

}

namespace w2xc {
void
filter_AVX_impl(const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *fbiases,
		const float *weight,
		cv::Size ipSize,
		int nJob)
{
	int wsz = ipSize.width;
	int hsz = ipSize.height;

	// filter processing
	// input : inputPlanes
	// kernel : weightMatrices


#if 1
	int per_job = hsz / nJob;

	std::vector<std::thread> workerThreads;

	for (int ji=0; ji<nJob; ji++) {
		auto t = std::thread([&](int ji) {
				float *intermediate = (float*)malloc(sizeof(float)*nOutputPlanes);

				int start = per_job * ji, end;

				if (ji == nJob-1) {
					end = hsz;
				} else {
					end = per_job * (ji+1);
				}

				for (int yi=start; yi<end; yi++) {
					for (int xi=0; xi<wsz; xi++) {
						if (yi == 0 || xi ==0 || yi == (hsz-1) || xi == (wsz-1)) {
							filter_1elem<true>(packed_input, nInputPlanes,
									   packed_output, nOutputPlanes,
									   fbiases, hsz, wsz, yi, xi, weight, intermediate);
						} else {
							filter_1elem<false>(packed_input, nInputPlanes,
									    packed_output, nOutputPlanes,
									    fbiases, hsz, wsz, yi, xi, weight, intermediate);
						}
					}
				}

				free(intermediate);
			}, ji
			);

		workerThreads.push_back(std::move(t));
	}

	for (auto& th : workerThreads) {
		th.join();
	}

#else

#pragma omp parallel for
	for (int yi=0; yi<hsz; yi++) {
		float *intermediate = (float*)malloc(sizeof(float)*nOutputPlanes);

		for (int xi=0; xi<wsz; xi++) {
			if (yi == 0 || xi ==0 || yi == (hsz-1) || xi == (wsz-1)) {
				filter_1elem<true>(packed_input, nInputPlanes,
						   packed_output, nOutputPlanes,
						   fbiases, hsz, wsz, yi, xi, weight, intermediate);
			} else {
				filter_1elem<false>(packed_input, nInputPlanes,
						    packed_output, nOutputPlanes,
						    fbiases, hsz, wsz, yi, xi, weight, intermediate);
			}
		}

		free(intermediate);
	}

#endif
}
}
