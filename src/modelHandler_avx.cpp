#include <thread>
#include <immintrin.h>
#include "modelHandler.hpp"
#include "common.hpp"
#include "sec.hpp"

template <bool border> inline
float
get_data(const float *p, int wsz, int xi, int num_plane, int plane)
{
	if (border) {
		xi = (std::min)(wsz-1, xi);
		xi = (std::max)(0, xi);

		return p[xi * num_plane + plane];
	} else {
		return p[xi * num_plane + plane];
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
	     float *intermediate0)
{
	size_t in_step = wsz * sizeof(float) * nInputPlanes;
	char *inp = (char*)packed_input;

	inp += in_step*yi;
	char *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}

	char *in1p = inp;
	char *in2p = inp + in_step;

	if (yi == hsz-1) {
		in2p = inp;
	}

	float *in01 = (float*)in0p;
	float *in11 = (float*)in1p;
	float *in21 = (float*)in2p;

	in01 += xi * nInputPlanes;
	in11 += xi * nInputPlanes;
	in21 += xi * nInputPlanes;

	float *intermediate1 = intermediate0 + nOutputPlanes;

	for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
		__m256 i01 = _mm256_broadcast_ss(in01);
		__m256 i11 = _mm256_broadcast_ss(in11);
		__m256 i21 = _mm256_broadcast_ss(in21);

		__m256 i00, i10, i20;
		__m256 i02, i12, i22;
		__m256 i03, i13, i23;

		if (border && xi == 0) {
			i00 = i01;
			i10 = i11;
			i20 = i21;
		} else {
			i00 = _mm256_broadcast_ss(in01-nInputPlanes);
			i10 = _mm256_broadcast_ss(in11-nInputPlanes);
			i20 = _mm256_broadcast_ss(in21-nInputPlanes);
		}

		i02 = _mm256_broadcast_ss(in01+nInputPlanes);
		i12 = _mm256_broadcast_ss(in11+nInputPlanes);
		i22 = _mm256_broadcast_ss(in21+nInputPlanes);

		if (border && xi+1 == wsz-1) {
			i03 = i02;
			i13 = i12;
			i23 = i22;
		} else {
			i03 = _mm256_broadcast_ss(in01+nInputPlanes*2);
			i13 = _mm256_broadcast_ss(in11+nInputPlanes*2);
			i23 = _mm256_broadcast_ss(in21+nInputPlanes*2);
		}

		in01++;
		in11++;
		in21++;

		const float *w = weight + (ipIndex * nOutputPlanes) * 9;

		for (unsigned int opIndex = 0;
		     opIndex < (unsigned int)nOutputPlanes;
		     opIndex += VEC_WIDTH)
		{
			__m256 v0, v1;
			v0 = _mm256_setzero_ps();
			v1 = _mm256_setzero_ps();

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i00, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i01, v1);

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i01, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i02, v1);

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i02, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i03, v1);



			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i10, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i11, v1);

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i11, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i12, v1);

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i12, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i13, v1);


			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i20, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i21, v1);

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i21, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i22, v1);

			v0 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i22, v0);
			v1 = _mm256_fmadd_ps(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i23, v1);

			w += 9 * VEC_WIDTH;

			if (ipIndex == 0) {
				_mm256_storeu_ps(&intermediate0[opIndex+0], v0);
				_mm256_storeu_ps(&intermediate1[opIndex+0], v1);
			} else {					\
				__m256 prev0 = _mm256_loadu_ps(&intermediate0[opIndex+0]);
				__m256 prev1 = _mm256_loadu_ps(&intermediate1[opIndex+0]);

				_mm256_storeu_ps(&intermediate0[opIndex+0], _mm256_add_ps(prev0,v0));
				_mm256_storeu_ps(&intermediate1[opIndex+0], _mm256_add_ps(prev1,v1));
			}
		}
	}

	float *out0 = packed_output + (yi*wsz + xi)*nOutputPlanes;
	float *out1 = packed_output + (yi*wsz + (xi+1))*nOutputPlanes;

	for (int opIndex = 0; opIndex < nOutputPlanes; opIndex+=VEC_WIDTH) {
		__m256 bv = _mm256_loadu_ps(&biases[opIndex]);
		__m256 v, mtz, ltz;

		v = _mm256_loadu_ps(&intermediate0[opIndex]);
		v = _mm256_add_ps(v, bv);
		mtz = _mm256_max_ps(v, _mm256_setzero_ps());
		ltz = _mm256_min_ps(v, _mm256_setzero_ps());
		v = _mm256_add_ps(_mm256_mul_ps(ltz, _mm256_set1_ps(0.1f)), mtz);
		_mm256_storeu_ps(&out0[opIndex], v);

		v = _mm256_loadu_ps(&intermediate1[opIndex]);
		v = _mm256_add_ps(v, bv);
		mtz = _mm256_max_ps(v, _mm256_setzero_ps());
		ltz = _mm256_min_ps(v, _mm256_setzero_ps());
		v = _mm256_add_ps(_mm256_mul_ps(ltz, _mm256_set1_ps(0.1f)), mtz);
		_mm256_storeu_ps(&out1[opIndex], v);
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

	int per_job = hsz / nJob;

	std::vector<std::thread> workerThreads;

	for (int ji=0; ji<nJob; ji++) {
		auto t = std::thread([&](int ji) {
				float *intermediate = (float*)malloc(sizeof(float)*nOutputPlanes*2);

				int start = per_job * ji, end;

				if (ji == nJob-1) {
					end = hsz;
				} else {
					end = per_job * (ji+1);
				}

				for (int yi=start; yi<end; yi++) {
					for (int xi=0; xi<wsz; xi+=2) {
						if (xi ==0 || xi+1 == (wsz-1)) {
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

}
}
