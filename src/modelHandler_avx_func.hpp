#include "threadPool.hpp"
#include "params.h"

#define BLOCK_SIZE_HOR 256
#define BLOCK_SIZE_VER 16

template <bool have_fma> __m256 MADD(__m256 v0, __m256 v1, __m256 v2);

#ifdef HAVE_FMA
template <> inline __m256
MADD<true>(__m256 v0, __m256 v1, __m256 v2)
{
	return _mm256_fmadd_ps(v0, v1, v2);
}
#endif

template <> inline __m256
MADD<false>(__m256 v0, __m256 v1, __m256 v2)
{
	return _mm256_add_ps(_mm256_mul_ps(v0, v1), v2);
}


template <bool border, bool ip0, bool have_fma>
static void
apply_filter(unsigned long xi, unsigned long wsz,
	     const float *in01,
	     const float *in11,
	     const float *in21,
	     const float *w,
	     float *intermediate0,
	     int ipIndex,
	     int nInputPlanes,
	     int nOutputPlanes)
{
	float *intermediate1 = intermediate0 + nOutputPlanes;

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

	for (unsigned int opIndex = 0;
	     opIndex < (unsigned int)nOutputPlanes;
	     opIndex += VEC_WIDTH*UNROLL)
	{
		__m256 v00, v01, v10, v11;
		v00 = _mm256_setzero_ps();
		v01 = _mm256_setzero_ps();

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i00, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i01, v01);

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i01, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i02, v01);

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i02, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i03, v01);


		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i10, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i11, v01);

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i11, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i12, v01);

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i12, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i13, v01);


		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i20, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i21, v01);

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i21, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i22, v01);

		v00 = MADD<have_fma>(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i22, v00);
		v01 = MADD<have_fma>(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i23, v01);

		w += 9 * VEC_WIDTH;

		if (ip0) {
			_mm256_storeu_ps(&intermediate0[opIndex+0], v00);
			_mm256_storeu_ps(&intermediate1[opIndex+0], v01);
		} else {					\
			__m256 prev00 = _mm256_loadu_ps(&intermediate0[opIndex+0]);
			__m256 prev01 = _mm256_loadu_ps(&intermediate1[opIndex+0]);

			_mm256_storeu_ps(&intermediate0[opIndex+0], _mm256_add_ps(prev00,v00));
			_mm256_storeu_ps(&intermediate1[opIndex+0], _mm256_add_ps(prev01,v01));
		}

		v10 = _mm256_setzero_ps();
		v11 = _mm256_setzero_ps();

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i00, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i01, v11);

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i01, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i02, v11);

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i02, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i03, v11);


		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i10, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i11, v11);

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i11, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i12, v11);

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i12, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i13, v11);


		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i20, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i21, v11);

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i21, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i22, v11);

		v10 = MADD<have_fma>(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i22, v10);
		v11 = MADD<have_fma>(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i23, v11);

		w += 9 * VEC_WIDTH;

		if (ip0) {
			_mm256_storeu_ps(&intermediate0[opIndex+8], v10);
			_mm256_storeu_ps(&intermediate1[opIndex+8], v11);
		} else {
			__m256 prev10 = _mm256_loadu_ps(&intermediate0[opIndex+8]);
			__m256 prev11 = _mm256_loadu_ps(&intermediate1[opIndex+8]);

			_mm256_storeu_ps(&intermediate0[opIndex+8], _mm256_add_ps(prev10,v10));
			_mm256_storeu_ps(&intermediate1[opIndex+8], _mm256_add_ps(prev11,v11));
		}
	}

}

template <bool border, bool have_fma> void
filter_2elem(const float *packed_input,
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

	for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
		const float *w = weight + (ipIndex * nOutputPlanes) * 9;

		if (ipIndex == 0) {
			apply_filter<border, true, have_fma>(xi, wsz, in01, in11, in21, w, intermediate0,
							     ipIndex,
							     nInputPlanes, nOutputPlanes);
		} else {
			apply_filter<border, false, have_fma>(xi, wsz, in01, in11, in21, w, intermediate0,
							      ipIndex,
							      nInputPlanes, nOutputPlanes);
		}

		in01++;
		in11++;
		in21++;
	}

	float *out0 = packed_output + (yi*wsz + xi)*nOutputPlanes;
	float *out1 = packed_output + (yi*wsz + (xi+1))*nOutputPlanes;
	float *intermediate1 = intermediate0 + nOutputPlanes;

	for (int opIndex = 0; opIndex < nOutputPlanes; opIndex+=VEC_WIDTH) {
		__m256 bv = _mm256_loadu_ps(&biases[opIndex]);
		__m256 v, mtz, ltz;

		v = _mm256_loadu_ps(&intermediate0[opIndex]);
		v = _mm256_add_ps(v, bv);
		mtz = _mm256_max_ps(v, _mm256_setzero_ps());
		ltz = _mm256_min_ps(v, _mm256_setzero_ps());
		v = MADD<have_fma>(ltz, _mm256_set1_ps(0.1f), mtz);
		_mm256_storeu_ps(&out0[opIndex], v);

		v = _mm256_loadu_ps(&intermediate1[opIndex]);
		v = _mm256_add_ps(v, bv);
		mtz = _mm256_max_ps(v, _mm256_setzero_ps());
		ltz = _mm256_min_ps(v, _mm256_setzero_ps());
		v = MADD<have_fma>(ltz, _mm256_set1_ps(0.1f), mtz);
		_mm256_storeu_ps(&out1[opIndex], v);
	}
}

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

template <bool border, bool have_fma> void
filter_1elem_output1(const float *packed_input,
		     int nInputPlanes,
		     float *packed_output,
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

	__m256 sum = _mm256_setzero_ps();
	const float *w = weight;

	for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex+=VEC_WIDTH) {
		__m256 i00, i01, i02;
		__m256 i10, i11, i12;
		__m256 i20, i21, i22;

		i01 = _mm256_loadu_ps(&in01[0]);
		i11 = _mm256_loadu_ps(&in11[0]);
		i21 = _mm256_loadu_ps(&in21[0]);

		if (border && xi == 0) {
			i00 = i01;
			i10 = i11;
			i20 = i21;
		} else {
			i00 = _mm256_loadu_ps(&in01[-nInputPlanes]);
			i10 = _mm256_loadu_ps(&in11[-nInputPlanes]);
			i20 = _mm256_loadu_ps(&in21[-nInputPlanes]);
		}

		if (border && xi == wsz-1) {
			i02 = i01;
			i12 = i11;
			i22 = i21;
		} else {
			i02 = _mm256_loadu_ps(&in01[+nInputPlanes]);
			i12 = _mm256_loadu_ps(&in11[+nInputPlanes]);
			i22 = _mm256_loadu_ps(&in21[+nInputPlanes]);
		}

		in01+=VEC_WIDTH;
		in11+=VEC_WIDTH;
		in21+=VEC_WIDTH;

		__m256 v;

		v = _mm256_mul_ps(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i00);
		v = MADD<have_fma>(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i01, v);
		v = MADD<have_fma>(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i02, v);

		v = MADD<have_fma>(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i10, v);
		v = MADD<have_fma>(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i11, v);
		v = MADD<have_fma>(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i12, v);

		v = MADD<have_fma>(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i20, v);
		v = MADD<have_fma>(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i21, v);
		v = MADD<have_fma>(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i22, v);

		sum = _mm256_add_ps(v, sum);

		w += 9 * VEC_WIDTH;
	}

	sum = _mm256_hadd_ps(sum, sum);
	sum = _mm256_hadd_ps(sum, sum);

	float v0 = _mm_cvtss_f32(_mm256_extractf128_ps(sum,0));
	float v1 = _mm_cvtss_f32(_mm256_extractf128_ps(sum,1));

	float v = v0 + v1;

	float *out0 = packed_output + (yi*wsz + xi);

	float bv = biases[0];
	v += bv;
	float mtz = (std::max)(v, 0.0f);
	float ltz = (std::min)(v, 0.0f);

	v = ltz * 0.1f + mtz;

	*out0 = v;
}


#define CEIL_DIV(a,b) (((a)+(b-1))/(b))

template <bool have_fma>
static void
filter_AVX_impl0(const float *packed_input,
		 float *packed_output,
		 int nInputPlanes,
		 int nOutputPlanes,
		 const float *fbiases,
		 const float *weight,
		 int ip_width,
		 int ip_height,
		 int nJob)
{
	unsigned int wsz = ip_width;
	unsigned int hsz = ip_height;

	// filter processing
	// input : inputPlanes
	// kernel : weightMatrices

	unsigned int num_block_hor = CEIL_DIV(wsz, BLOCK_SIZE_HOR);
	unsigned int num_block_ver = CEIL_DIV(hsz, BLOCK_SIZE_VER);

	unsigned int total_block = num_block_hor * num_block_ver;

	std::atomic<unsigned int> block_counter(0U);

	startFunc([&]() {
			float *intermediate = (float*)_mm_malloc(sizeof(float)*nOutputPlanes*2, 64);

			while (1) {
				unsigned int bi = block_counter++;

				if (bi >= total_block) {
					_mm_free(intermediate);
					return;
				}

				unsigned int block_x = bi % num_block_hor;
				unsigned int block_y = bi / num_block_hor;

				unsigned int y_start = block_y * BLOCK_SIZE_VER;
				unsigned int y_end = (std::min)(y_start + BLOCK_SIZE_VER, hsz);

				unsigned int x_start = block_x * BLOCK_SIZE_HOR;
				unsigned int x_end = (std::min)(x_start + BLOCK_SIZE_HOR, wsz);

				if (nOutputPlanes == 1) {
					for (unsigned int yi=y_start; yi<y_end; yi++) {
						for (unsigned int xi=x_start; xi<x_end; xi++) {
							if (xi ==0 || xi == (wsz-1)) {
								filter_1elem_output1<true,have_fma>(packed_input, nInputPlanes,
												    packed_output,
												    fbiases, hsz, wsz, yi, xi, weight, intermediate);
							} else {
								filter_1elem_output1<false,have_fma>(packed_input, nInputPlanes,
												     packed_output,
												     fbiases, hsz, wsz, yi, xi, weight, intermediate);
							}
						}
					}
				} else {
					for (unsigned int yi=y_start; yi<y_end; yi++) {
						for (unsigned int xi=x_start; xi<x_end; xi+=2) {
							if (xi ==0 || xi+1 == (wsz-1)) {
								filter_2elem<true,have_fma>(packed_input, nInputPlanes,
											    packed_output, nOutputPlanes,
											    fbiases, hsz, wsz, yi, xi, weight, intermediate);
							} else {
								filter_2elem<false, have_fma>(packed_input, nInputPlanes,
											      packed_output, nOutputPlanes,
											      fbiases, hsz, wsz, yi, xi, weight, intermediate);
							}
						}
					}
				}
			}
		}
		);
}
