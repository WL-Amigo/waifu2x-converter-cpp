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

#include <string.h>
#include "threadPool.hpp"

namespace
{

	static inline bool simd_available(int nInputPlanes, int nOutputPlanes)
	{
		return ((nInputPlanes % (VEC_NELEM*4)) == 0 && (nOutputPlanes % (VEC_NELEM*2)) == 0);
	}

	static void apply_filter_line
	(
		unsigned long width,
		unsigned long height,
		const float *fin,
		const float *fw,
		float *foutput,
		const float *fbiases,
		int nInputPlanes,
		int nOutputPlanes,
		int xb_start,
		int xb_end,
		int yi
	)
	{
		const unsigned char *in = (unsigned char*)fin;
		const unsigned char *w = (unsigned char*)fw;
		const unsigned char *biases = (unsigned char*)fbiases;
		unsigned char *output = (unsigned char*)foutput;

#ifdef SIMD_OPLANE
		int OP_BLOCK_SIZE = VEC_NELEM * 2;
		int IP_BLOCK_SIZE = VEC_NELEM * 4;
#elif defined SIMD_IPLANE
		int OP_BLOCK_SIZE = VEC_NELEM * 4;
		int IP_BLOCK_SIZE = VEC_NELEM * 2;
#endif

		int nOutputPlane_block = nOutputPlanes / OP_BLOCK_SIZE;
		int nInputPlane_block = nInputPlanes / IP_BLOCK_SIZE;

		for (int dposy=0; dposy<3; dposy++)
		{
			bool dposy_last = dposy==2;
			bool dopsy_first = dposy==0;

			int dposy2 = dposy - 1;

			if (yi == 0 && dposy == 0)
			{
				dposy2 = 0;
			}

			if (yi == height-1 && dposy == 2)
			{
				dposy2 = 0;
			}

			for(int ii0=0; ii0<nInputPlane_block; ii0++)
			{
				bool iplane_last = (ii0 == nInputPlane_block-1);
				bool iplane_first = (ii0 == 0);

				for (int oi0=0; oi0<nOutputPlane_block; oi0++)
				{
					int w_chunk_index = (((dposy * nInputPlane_block) + ii0) * nOutputPlane_block) + oi0;
					size_t w_chunk_size = OP_BLOCK_SIZE * IP_BLOCK_SIZE * 3 * sizeof(float);
					const unsigned char *w_chunk_base = w + (w_chunk_size * w_chunk_index);

					bool last_iter = dposy_last && iplane_last;

					int x0 = xb_start;

					if (UNROLL == 5)
					{
						for (; x0<xb_end-4; x0+=5)
						{
#							include "modelHandler_simd_unroll5.hpp"
						}
					}


					if (UNROLL == 2)
					{
						for (; x0<xb_end-1; x0+=2)
							{
#								include "modelHandler_simd_unroll2.hpp"
							}
					}


					if (UNROLL == 4)
					{
						for (; x0<xb_end-3; x0+=4)
						{
#							include "modelHandler_simd_unroll4.hpp"
						}
					}



#if 0
					if (UNROLL == 6)
					{
						for (; x0<xb_end-5; x0+=6)
						{
#							include "body-unroll6.hpp"
						}
					}
#endif
					for (;x0<xb_end; x0++)
					{
#						include "modelHandler_simd_unroll1.hpp"
					}
#if 0
					// simple test routine
					for (; x0<xb_end; x0++)
					{
						const unsigned char *w_cur = w_chunk_base;
						unsigned char *output_base0 = output + ((x0+0)*nOutputPlanes + oi0*OP_BLOCK_SIZE)*sizeof(float);

						float otmp[OP_BLOCK_SIZE];
						memset(otmp, 0, sizeof(otmp));

						for (int dposx=0; dposx<3; dposx++)
						{
							int dposx2 = dposx-1;

							if ((x0 == 0) && (dposx == 0))
							{
								dposx2 = 0;
							}

							if ((x0 == width-1) && (dposx == 2))
							{
								dposx2 = 0;
							}

							const unsigned char *input_cur = in + ((dposy2*width + x0 + dposx2)*nInputPlanes+ii0*IP_BLOCK_SIZE)*sizeof(float);

							for (int ii1=0; ii1<IP_BLOCK_SIZE; ii1++)
							{
								float ireg = *(float*)input_cur;
								input_cur += sizeof(float);

								for (int oi1=0; oi1<OP_BLOCK_SIZE; oi1++)
								{
									float wreg = *(float*)w_cur;
									w_cur += sizeof(float);

									otmp[oi1] += ireg*wreg;
								}
							}
						}

						if (dposy == 0 && ii0 == 0)
						{
							for (int oi1=0; oi1<OP_BLOCK_SIZE; oi1++)
							{
								((float*)output_base0)[oi1] = otmp[oi1];
							}
						}
						else if (last_iter)
						{
							for (int oi1=0; oi1<OP_BLOCK_SIZE; oi1++)
							{
								float bv = ((float*)fbiases)[oi0*OP_BLOCK_SIZE+oi1];
								float v = ((float*)output_base0)[oi1];
								v += otmp[oi1] + bv;
								float mtz = (std::max)(v, 0.0f);
								float ltz = (std::min)(v, 0.0f);
								v = ltz * 0.1f + mtz;
								((float*)output_base0)[oi1] = v;
							}
						}
						else
						{
							for (int oi1=0; oi1<OP_BLOCK_SIZE; oi1++)
							{
								((float*)output_base0)[oi1] += otmp[oi1];
							}
						}
					}
#endif
				}
			}
		}
	}

	static inline void filter_simd_impl0
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

		int block_size_hor = 128;

		if (UNROLL == 5)
		{
			block_size_hor = 125;
		}

		int block_size_ver = 16;

		// filter processing
		// input : inputPlanes
		// kernel : weightMatrices

		unsigned int num_block_hor = CEIL_DIV(wsz, block_size_hor);
		unsigned int num_block_ver = CEIL_DIV(hsz, block_size_ver);
		unsigned int total_block = num_block_hor * num_block_ver;

		std::atomic<unsigned int> block_counter(0U);

		auto func = [&]()
		{
			while (true)
			{
				unsigned int bi = block_counter++;

				if (bi >= total_block)
				{
					return;
				}

				unsigned int block_x = bi % num_block_hor;
				unsigned int block_y = bi / num_block_hor;

				unsigned int y_start = block_y * block_size_ver;
				unsigned int y_end = (std::min)(y_start + block_size_ver, hsz);

				unsigned int x_start = block_x * block_size_hor;
				unsigned int x_end = (std::min)(x_start + block_size_hor, wsz);
				
				for (unsigned int yi=y_start; yi<y_end; yi++)
				{
					const float *input_block = packed_input + (nInputPlanes*yi*wsz);
					float *output_block = packed_output + (nOutputPlanes*yi*wsz);

					apply_filter_line
					(
						wsz, 
						hsz,
						input_block,
						weight,
						output_block,
						fbiases,
						nInputPlanes,
						nOutputPlanes,
						x_start,
						x_end,
						yi
					);
				}
			}
		};
#if !defined(_WIN32) && !defined(__linux)
		std::vector<std::thread> workerThreads;
		for (int ji=0; ji<nJob; ji++) {
			workerThreads.emplace_back(std::thread(func));
		}
		for (auto& th : workerThreads) {
			th.join();
		}
#else
		w2xc::startFunc(env->tpool, func);
#endif
	}
}
