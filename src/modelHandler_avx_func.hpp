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

#include "threadPool.hpp"
#include "params.h"
#include <algorithm>
#include <vector>

namespace
{
#define BLOCK_SIZE_HOR 256
#define BLOCK_SIZE_VER 16
	template <bool border, bool ip0>
	static void apply_filter
	(
		unsigned long xi,
		unsigned long wsz,
		const float *in01,
		const float *in11,
		const float *in21,
		const float *w,
		float *intermediate0,
		int ipIndex,
		int nInputPlanes,
		int nOutputPlanes
	)
	{
		float *intermediate1 = intermediate0 + nOutputPlanes;

		v256_t i01 = load_broadcast(in01);
		v256_t i11 = load_broadcast(in11);
		v256_t i21 = load_broadcast(in21);

		v256_t i00, i10, i20;
		v256_t i02, i12, i22;
		v256_t i03, i13, i23;

		if (border && xi == 0) {
			i00 = i01;
			i10 = i11;
			i20 = i21;
		}
		else
		{
			i00 = load_broadcast(in01-nInputPlanes);
			i10 = load_broadcast(in11-nInputPlanes);
			i20 = load_broadcast(in21-nInputPlanes);
		}

		i02 = load_broadcast(in01+nInputPlanes);
		i12 = load_broadcast(in11+nInputPlanes);
		i22 = load_broadcast(in21+nInputPlanes);

		if (border && xi+1 == wsz-1)
		{
			i03 = i02;
			i13 = i12;
			i23 = i22;
		}
		else
		{
			i03 = load_broadcast(in01+nInputPlanes*2);
			i13 = load_broadcast(in11+nInputPlanes*2);
			i23 = load_broadcast(in21+nInputPlanes*2);
		}

		for (unsigned int opIndex = 0; opIndex < (unsigned int)nOutputPlanes; opIndex += VEC_WIDTH*UNROLL)
		{
			v256_t v00, v01, v10, v11;
			v00 = zero();
			v01 = zero();

			v00 = madd256(load256(&w[0*VEC_WIDTH]), i00, v00);
			v01 = madd256(load256(&w[0*VEC_WIDTH]), i01, v01);

			v00 = madd256(load256(&w[1*VEC_WIDTH]), i01, v00);
			v01 = madd256(load256(&w[1*VEC_WIDTH]), i02, v01);

			v00 = madd256(load256(&w[2*VEC_WIDTH]), i02, v00);
			v01 = madd256(load256(&w[2*VEC_WIDTH]), i03, v01);


			v00 = madd256(load256(&w[3*VEC_WIDTH]), i10, v00);
			v01 = madd256(load256(&w[3*VEC_WIDTH]), i11, v01);

			v00 = madd256(load256(&w[4*VEC_WIDTH]), i11, v00);
			v01 = madd256(load256(&w[4*VEC_WIDTH]), i12, v01);

			v00 = madd256(load256(&w[5*VEC_WIDTH]), i12, v00);
			v01 = madd256(load256(&w[5*VEC_WIDTH]), i13, v01);


			v00 = madd256(load256(&w[6*VEC_WIDTH]), i20, v00);
			v01 = madd256(load256(&w[6*VEC_WIDTH]), i21, v01);

			v00 = madd256(load256(&w[7*VEC_WIDTH]), i21, v00);
			v01 = madd256(load256(&w[7*VEC_WIDTH]), i22, v01);

			v00 = madd256(load256(&w[8*VEC_WIDTH]), i22, v00);
			v01 = madd256(load256(&w[8*VEC_WIDTH]), i23, v01);

			w += 9 * VEC_WIDTH;

			if (ip0)
			{
				store256(&intermediate0[opIndex+0], v00);
				store256(&intermediate1[opIndex+0], v01);
			}
			else
			{
				v256_t prev00 = load256(&intermediate0[opIndex+0]);
				v256_t prev01 = load256(&intermediate1[opIndex+0]);

				store256(&intermediate0[opIndex+0], add256(prev00,v00));
				store256(&intermediate1[opIndex+0], add256(prev01,v01));
			}

			v10 = zero();
			v11 = zero();

			v10 = madd256(load256(&w[0*VEC_WIDTH]), i00, v10);
			v11 = madd256(load256(&w[0*VEC_WIDTH]), i01, v11);

			v10 = madd256(load256(&w[1*VEC_WIDTH]), i01, v10);
			v11 = madd256(load256(&w[1*VEC_WIDTH]), i02, v11);

			v10 = madd256(load256(&w[2*VEC_WIDTH]), i02, v10);
			v11 = madd256(load256(&w[2*VEC_WIDTH]), i03, v11);


			v10 = madd256(load256(&w[3*VEC_WIDTH]), i10, v10);
			v11 = madd256(load256(&w[3*VEC_WIDTH]), i11, v11);

			v10 = madd256(load256(&w[4*VEC_WIDTH]), i11, v10);
			v11 = madd256(load256(&w[4*VEC_WIDTH]), i12, v11);

			v10 = madd256(load256(&w[5*VEC_WIDTH]), i12, v10);
			v11 = madd256(load256(&w[5*VEC_WIDTH]), i13, v11);


			v10 = madd256(load256(&w[6*VEC_WIDTH]), i20, v10);
			v11 = madd256(load256(&w[6*VEC_WIDTH]), i21, v11);

			v10 = madd256(load256(&w[7*VEC_WIDTH]), i21, v10);
			v11 = madd256(load256(&w[7*VEC_WIDTH]), i22, v11);

			v10 = madd256(load256(&w[8*VEC_WIDTH]), i22, v10);
			v11 = madd256(load256(&w[8*VEC_WIDTH]), i23, v11);

			w += 9 * VEC_WIDTH;

			if (ip0)
			{
				store256(&intermediate0[opIndex+8], v10);
				store256(&intermediate1[opIndex+8], v11);
			}
			else
			{
				v256_t prev10 = load256(&intermediate0[opIndex+8]);
				v256_t prev11 = load256(&intermediate1[opIndex+8]);

				store256(&intermediate0[opIndex+8], add256(prev10,v10));
				store256(&intermediate1[opIndex+8], add256(prev11,v11));
			}
		}

	}

	template <bool border>
	static inline void filter_2elem
	(
		const float *packed_input,
		int nInputPlanes,
		float *packed_output,
		int nOutputPlanes,
		const float *biases,
		unsigned long hsz,
		unsigned long wsz,
		unsigned long yi,
		unsigned long xi,
		const float *weight,
		float *intermediate0
	)
	{
		size_t in_step = wsz * sizeof(float) * nInputPlanes;
		char *inp = (char*)packed_input;

		inp += in_step*yi;
		char *in0p = inp - in_step;

		if (yi == 0)
		{
			in0p = inp;
		}

		char *in1p = inp;
		char *in2p = inp + in_step;

		if (yi == hsz-1)
		{
			in2p = inp;
		}

		float *in01 = (float*)in0p;
		float *in11 = (float*)in1p;
		float *in21 = (float*)in2p;

		in01 += xi * nInputPlanes;
		in11 += xi * nInputPlanes;
		in21 += xi * nInputPlanes;

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++)
		{
			const float *w = weight + (ipIndex * nOutputPlanes) * 9;

			if (ipIndex == 0)
			{
				apply_filter<border, true>
				(
					xi,
					wsz,
					in01,
					in11,
					in21,
					w,
					intermediate0,
					ipIndex,
					nInputPlanes,
					nOutputPlanes
				);
			}
			else
			{
				apply_filter<border, false>
				(
					xi,
					wsz,
					in01,
					in11,
					in21,
					w,
					intermediate0,
					ipIndex,
					nInputPlanes,
					nOutputPlanes
				);
			}

			in01++;
			in11++;
			in21++;
		}

		float *out0 = packed_output + (yi*wsz + xi)*nOutputPlanes;
		float *out1 = packed_output + (yi*wsz + (xi+1))*nOutputPlanes;
		float *intermediate1 = intermediate0 + nOutputPlanes;

		for (int opIndex = 0; opIndex < nOutputPlanes; opIndex+=VEC_WIDTH)
		{
			v256_t bv = load256(&biases[opIndex]);
			v256_t v, mtz, ltz;

			v = load256(&intermediate0[opIndex]);
			v = add256(v, bv);
			mtz = max256(v, zero());
			ltz = min256(v, zero());
			v = madd256(ltz, set1(0.1f), mtz);
			store256(&out0[opIndex], v);

			v = load256(&intermediate1[opIndex]);
			v = add256(v, bv);
			mtz = max256(v, zero());
			ltz = min256(v, zero());
			v = madd256(ltz, set1(0.1f), mtz);
			store256(&out1[opIndex], v);
		}
	}

	template <bool border>
	static inline float get_data(const float *p, int wsz, int xi, int num_plane, int plane)
	{
		if (border)
		{
			xi = (std::min)(wsz-1, xi);
			xi = (std::max)(0, xi);

			return p[xi * num_plane + plane];
		}
		else
		{
			return p[xi * num_plane + plane];
		}
	}

	template <bool border>
	static void filter_1elem_output1
	(
		const float *packed_input,
		int nInputPlanes,
		float *packed_output,
		const float *biases,
		unsigned long hsz,
		unsigned long wsz,
		unsigned long yi,
		unsigned long xi,
		const float *weight,
		float *intermediate0
	)
	{
		size_t in_step = wsz * sizeof(float) * nInputPlanes;
		char *inp = (char*)packed_input;

		inp += in_step*yi;
		char *in0p = inp - in_step;

		if (yi == 0)
		{
			in0p = inp;
		}

		char *in1p = inp;
		char *in2p = inp + in_step;

		if (yi == hsz-1)
		{
			in2p = inp;
		}

		float *in01 = (float*)in0p;
		float *in11 = (float*)in1p;
		float *in21 = (float*)in2p;

		in01 += xi * nInputPlanes;
		in11 += xi * nInputPlanes;
		in21 += xi * nInputPlanes;

		v256_t sum = zero();
		const float *w = weight;

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex+=VEC_WIDTH)
		{
			v256_t i00, i01, i02;
			v256_t i10, i11, i12;
			v256_t i20, i21, i22;

			i01 = load256(&in01[0]);
			i11 = load256(&in11[0]);
			i21 = load256(&in21[0]);

			if (border && xi == 0)
			{
				i00 = i01;
				i10 = i11;
				i20 = i21;
			}
			else
			{
				i00 = load256(&in01[-nInputPlanes]);
				i10 = load256(&in11[-nInputPlanes]);
				i20 = load256(&in21[-nInputPlanes]);
			}

			if (border && xi == wsz-1)
			{
				i02 = i01;
				i12 = i11;
				i22 = i21;
			}
			else
			{
				i02 = load256(&in01[+nInputPlanes]);
				i12 = load256(&in11[+nInputPlanes]);
				i22 = load256(&in21[+nInputPlanes]);
			}

			in01+=VEC_WIDTH;
			in11+=VEC_WIDTH;
			in21+=VEC_WIDTH;

			v256_t v;

			v = mul256(load256(&w[0*VEC_WIDTH]), i00);
			v = madd256(load256(&w[1*VEC_WIDTH]), i01, v);
			v = madd256(load256(&w[2*VEC_WIDTH]), i02, v);

			v = madd256(load256(&w[3*VEC_WIDTH]), i10, v);
			v = madd256(load256(&w[4*VEC_WIDTH]), i11, v);
			v = madd256(load256(&w[5*VEC_WIDTH]), i12, v);

			v = madd256(load256(&w[6*VEC_WIDTH]), i20, v);
			v = madd256(load256(&w[7*VEC_WIDTH]), i21, v);
			v = madd256(load256(&w[8*VEC_WIDTH]), i22, v);

			sum = add256(v, sum);

			w += 9 * VEC_WIDTH;
		}

		float v = hadd8(sum);

		float *out0 = packed_output + (yi*wsz + xi);

		float bv = biases[0];
		v += bv;
		float mtz = (std::max)(v, 0.0f);
		float ltz = (std::min)(v, 0.0f);

		v = ltz * 0.1f + mtz;

		*out0 = v;
	}

	template <bool border>
	static void filter_1elem_output3
	(
		const float *packed_input,
		int nInputPlanes,
		float *packed_output,
		const float *biases,
		unsigned long hsz,
		unsigned long wsz,
		unsigned long yi,
		unsigned long xi,
		const float *weight,
		float *intermediate0
	)
	{
		size_t in_step = wsz * sizeof(float) * nInputPlanes;
		char *inp = (char*)packed_input;

		inp += in_step*yi;
		char *in0p = inp - in_step;

		if (yi == 0)
		{
			in0p = inp;
		}

		char *in1p = inp;
		char *in2p = inp + in_step;

		if (yi == hsz-1)
		{
			in2p = inp;
		}

		float *in01 = (float*)in0p;
		float *in11 = (float*)in1p;
		float *in21 = (float*)in2p;

		in01 += xi * nInputPlanes;
		in11 += xi * nInputPlanes;
		in21 += xi * nInputPlanes;

		v256_t sum0 = zero();
		v256_t sum1 = zero();
		v256_t sum2 = zero();
		const float *w0 = weight + 9 * nInputPlanes * 0;
		const float *w1 = weight + 9 * nInputPlanes * 1;
		const float *w2 = weight + 9 * nInputPlanes * 2;

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex+=VEC_WIDTH)
		{
			v256_t i00, i01, i02;
			v256_t i10, i11, i12;
			v256_t i20, i21, i22;

			i01 = load256(&in01[0]);
			i11 = load256(&in11[0]);
			i21 = load256(&in21[0]);

			if (border && xi == 0)
			{
				i00 = i01;
				i10 = i11;
				i20 = i21;
			}
			else
			{
				i00 = load256(&in01[-nInputPlanes]);
				i10 = load256(&in11[-nInputPlanes]);
				i20 = load256(&in21[-nInputPlanes]);
			}

			if (border && xi == wsz-1)
			{
				i02 = i01;
				i12 = i11;
				i22 = i21;
			}
			else
			{
				i02 = load256(&in01[+nInputPlanes]);
				i12 = load256(&in11[+nInputPlanes]);
				i22 = load256(&in21[+nInputPlanes]);
			}

			in01+=VEC_WIDTH;
			in11+=VEC_WIDTH;
			in21+=VEC_WIDTH;

			v256_t v0, v1, v2;

#define OUT3_CONV9(I)							                   \
		v##I = mul256(load256(&w##I[0*nInputPlanes]), i00);        \
		v##I = madd256(load256(&w##I[1*nInputPlanes]), i01, v##I); \
		v##I = madd256(load256(&w##I[2*nInputPlanes]), i02, v##I); \
                                                                   \
		v##I = madd256(load256(&w##I[3*nInputPlanes]), i10, v##I); \
		v##I = madd256(load256(&w##I[4*nInputPlanes]), i11, v##I); \
		v##I = madd256(load256(&w##I[5*nInputPlanes]), i12, v##I); \
                                                                   \
		v##I = madd256(load256(&w##I[6*nInputPlanes]), i20, v##I); \
		v##I = madd256(load256(&w##I[7*nInputPlanes]), i21, v##I); \
		v##I = madd256(load256(&w##I[8*nInputPlanes]), i22, v##I); \
                                                                   \
		sum##I = add256(v##I, sum##I);

			OUT3_CONV9(0);
			OUT3_CONV9(1);
			OUT3_CONV9(2);

			w0 += VEC_WIDTH;
			w1 += VEC_WIDTH;
			w2 += VEC_WIDTH;
		}

		float *out0 = packed_output + (yi*wsz + xi) * 3;

#define OUT3_RELU(I)                         \
		{                                    \
			float v = hadd8(sum##I);;        \
											 \
			float bv = biases[I];            \
			v += bv;                         \
			float mtz = (std::max)(v, 0.0f); \
			float ltz = (std::min)(v, 0.0f); \
											 \
			v = ltz * 0.1f + mtz;            \
											 \
			out0[I] = v;                     \
		}

		OUT3_RELU(0);
		OUT3_RELU(1);
		OUT3_RELU(2);
	}


#define CEIL_DIV(a,b) (((a)+(b-1))/(b))

	static void filter_AVX_impl0
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
		unsigned int wsz = ip_width;
		unsigned int hsz = ip_height;

		// filter processing
		// input : inputPlanes
		// kernel : weightMatrices

		unsigned int num_block_hor = CEIL_DIV(wsz, BLOCK_SIZE_HOR);
		unsigned int num_block_ver = CEIL_DIV(hsz, BLOCK_SIZE_VER);

		unsigned int total_block = num_block_hor * num_block_ver;

		std::atomic<unsigned int> block_counter(0U);

		auto func = [&]()
		{
			float *intermediate = (float*)w2xc_aligned_malloc(sizeof(float)*nOutputPlanes*2, 64);

			while (true)
			{
				unsigned int bi = block_counter++;

				if (bi >= total_block)
				{
					w2xc_aligned_free(intermediate);
					return;
				}

				unsigned int block_x = bi % num_block_hor;
				unsigned int block_y = bi / num_block_hor;

				unsigned int y_start = block_y * BLOCK_SIZE_VER;
				unsigned int y_end = (std::min)(y_start + BLOCK_SIZE_VER, hsz);

				unsigned int x_start = block_x * BLOCK_SIZE_HOR;
				unsigned int x_end = (std::min)(x_start + BLOCK_SIZE_HOR, wsz);

				if (nOutputPlanes == 1)
				{
					for (unsigned int yi=y_start; yi<y_end; yi++)
					{
						for (unsigned int xi=x_start; xi<x_end; xi++)
						{
							if (xi ==0 || xi == (wsz-1))
							{
								filter_1elem_output1<true>
								(
									packed_input,
									nInputPlanes,
									packed_output,
									fbiases,
									hsz,
									wsz,
									yi,
									xi,
									weight,
									intermediate
								);
							}
							else
							{
								filter_1elem_output1<false>(
									packed_input,
									nInputPlanes,
									packed_output,
									fbiases,
									hsz,
									wsz,
									yi,
									xi,
									weight,
									intermediate
								);
							}
						}
					}
				}
				else if (nOutputPlanes == 3)
				{
					for (unsigned int yi=y_start; yi<y_end; yi++)
					{
						for (unsigned int xi=x_start; xi<x_end; xi++)
						{
							if (xi ==0 || xi == (wsz-1))
							{
								filter_1elem_output3<true>
								(
									packed_input,
									nInputPlanes,
									packed_output,
									fbiases,
									hsz,
									wsz, 
									yi, 
									xi, 
									weight,
									intermediate
								);
							}
							else
							{
								filter_1elem_output3<false>
								(
									packed_input,
									nInputPlanes,
									packed_output,
									fbiases,
									hsz,
									wsz,
									yi,
									xi,
									weight,
									intermediate
								);
							}
						}
					}
				}
				else
				{
					for (unsigned int yi = y_start; yi < y_end; yi++)
					{
						for (unsigned int xi = x_start; xi < x_end; xi+=2)
						{
							if (xi ==0 || xi+1 == (wsz-1))
							{
								filter_2elem<true>
								(
									packed_input,
									nInputPlanes,
									packed_output,
									nOutputPlanes,
									fbiases,
									hsz,
									wsz,
									yi,
									xi,
									weight,
									intermediate
								);
							}
							else
							{
								filter_2elem<false>
								(
									packed_input,
									nInputPlanes,
									packed_output,
									nOutputPlanes,
									fbiases,
									hsz,
									wsz,
									yi,
									xi,
									weight,
									intermediate
								);
							}
						}
					}
				}
			}
		};

#if !defined(_WIN32) && !defined(__linux)
		std::vector<std::thread> workerThreads;
		for (int ji=0; ji<nJob; ji++)
		{
			workerThreads.emplace_back(std::thread(func));
		}
		for (auto& th : workerThreads)
		{
			th.join();
		}
#else
		startFunc(env->tpool, func);
#endif
	}
}
