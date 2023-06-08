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

#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "w2xconv.h"
#include "Buffer.hpp"
#include <vector>

namespace w2xc
{
	void initOpenCLGlobal(std::vector<W2XConvProcessor> *proc_list);
	void initCUDAGlobal(std::vector<W2XConvProcessor> *proc_list);

	bool initOpenCL(W2XConv *c, ComputeEnv *env, W2XConvProcessor *proc);
	void finiOpenCL(ComputeEnv *env);
	bool initCUDA(ComputeEnv *env, int dev_id);
	void finiCUDA(ComputeEnv *env);

	extern void filter_SSE_impl
	(
		ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

	extern void filter_AVX_impl
	(
		ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

	extern void filter_FMA_impl
	(
		ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

	extern void filter_NEON_impl
	(
		ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

	extern void filter_OpenCL_impl
	(
		ComputeEnv *env,
		Buffer *packed_input,
		Buffer *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

	extern void filter_AltiVec_impl
	(
		ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

	extern void filter_CUDA_impl
	(
		ComputeEnv *env,
		Buffer *packed_input,
		Buffer *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *biases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob
	);

}

#endif
