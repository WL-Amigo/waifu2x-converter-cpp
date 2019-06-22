/*
* The MIT License (MIT)
* Copyright (c) 2015 amigo(white luckers), tanakamura, DeadSix27, YukihoAA and contributors
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

/* -*- mode: c++ -*- */

#define UNROLL9(F)				\
	F(0);					\
	F(1);					\
	F(2);					\
	F(3);					\
	F(4);					\
	F(5);					\
	F(6);					\
	F(7);					\
	F(8);					\


#define UNROLL8x3x3(F)				\
	F(0,0,0);				\
	F(0,0,1);				\
	F(0,0,2);				\
	F(0,1,0);				\
	F(0,1,1);				\
	F(0,1,2);				\
	F(0,2,0);				\
	F(0,2,1);				\
	F(0,2,2);				\
						\
	F(1,0,0);				\
	F(1,0,1);				\
	F(1,0,2);				\
	F(1,1,0);				\
	F(1,1,1);				\
	F(1,1,2);				\
	F(1,2,0);				\
	F(1,2,1);				\
	F(1,2,2);				\
						\
	F(2,0,0);				\
	F(2,0,1);				\
	F(2,0,2);				\
	F(2,1,0);				\
	F(2,1,1);				\
	F(2,1,2);				\
	F(2,2,0);				\
	F(2,2,1);				\
	F(2,2,2);				\
						\
	F(3,0,0);				\
	F(3,0,1);				\
	F(3,0,2);				\
	F(3,1,0);				\
	F(3,1,1);				\
	F(3,1,2);				\
	F(3,2,0);				\
	F(3,2,1);				\
	F(3,2,2);				\
						\
	F(4,0,0);				\
	F(4,0,1);				\
	F(4,0,2);				\
	F(4,1,0);				\
	F(4,1,1);				\
	F(4,1,2);				\
	F(4,2,0);				\
	F(4,2,1);				\
	F(4,2,2);				\
						\
	F(5,0,0);				\
	F(5,0,1);				\
	F(5,0,2);				\
	F(5,1,0);				\
	F(5,1,1);				\
	F(5,1,2);				\
	F(5,2,0);				\
	F(5,2,1);				\
	F(5,2,2);				\
						\
	F(6,0,0);				\
	F(6,0,1);				\
	F(6,0,2);				\
	F(6,1,0);				\
	F(6,1,1);				\
	F(6,1,2);				\
	F(6,2,0);				\
	F(6,2,1);				\
	F(6,2,2);				\
						\
	F(7,0,0);				\
	F(7,0,1);				\
	F(7,0,2);				\
	F(7,1,0);				\
	F(7,1,1);				\
	F(7,1,2);				\
	F(7,2,0);				\
	F(7,2,1);				\
	F(7,2,2);				\

#define UNROLL8(F)				\
	F(0);					\
	F(1);					\
	F(2);					\
	F(3);					\
	F(4);					\
	F(5);					\
	F(6);					\
	F(7);					\


#define UNROLL8x3(F)				\
	F(0,0);					\
	F(0,1);					\
	F(0,2);					\
	F(0,3);					\
	F(0,4);					\
	F(0,5);					\
	F(0,6);					\
	F(0,7);					\
						\
	F(1,0);					\
	F(1,1);					\
	F(1,2);					\
	F(1,3);					\
	F(1,4);					\
	F(1,5);					\
	F(1,6);					\
	F(1,7);					\
						\
	F(2,0);					\
	F(2,1);					\
	F(2,2);					\
	F(2,3);					\
	F(2,4);					\
	F(2,5);					\
	F(2,6);					\
	F(2,7);					\


#define UNROLL10x3(F)				\
	F(0,0);					\
	F(0,1);					\
	F(0,2);					\
	F(0,3);					\
	F(0,4);					\
	F(0,5);					\
	F(0,6);					\
	F(0,7);					\
	F(0,8);					\
	F(0,9);					\
						\
	F(1,0);					\
	F(1,1);					\
	F(1,2);					\
	F(1,3);					\
	F(1,4);					\
	F(1,5);					\
	F(1,6);					\
	F(1,7);					\
	F(1,8);					\
	F(1,9);					\
						\
	F(2,0);					\
	F(2,1);					\
	F(2,2);					\
	F(2,3);					\
	F(2,4);					\
	F(2,5);					\
	F(2,6);					\
	F(2,7);					\
	F(2,8);					\
	F(2,9);					\


#define BLOCK_SIZE 8

extern __shared__ float shared_buf[];

template <int nInputPlanes>
__device__ void
filter(const float * __restrict__ packed_input,
       float * __restrict__ packed_output,
       int nOutputPlanes,
       const float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       const float * __restrict__ weight)
{
	unsigned int yi = blockIdx.x;

	size_t in_step = wsz * nInputPlanes;
	const float *inp = packed_input;
	inp += yi * in_step;

	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	float *shared_ptr = shared_buf;
	float *in_block0_base = shared_ptr;
	shared_ptr += nInputPlanes*(BLOCK_SIZE+2);
	float *in_block1_base = shared_ptr;
	shared_ptr += nInputPlanes*(BLOCK_SIZE+2);
	float *in_block2_base = shared_ptr;
	shared_ptr += nInputPlanes*(BLOCK_SIZE+2);

	float *in_block0 = in_block0_base + nInputPlanes;
	float *in_block1 = in_block1_base + nInputPlanes;
	float *in_block2 = in_block2_base + nInputPlanes;
	int lid = threadIdx.x;
	float bv = biases[lid];

	for (int xi0=0; xi0<wsz; xi0+=BLOCK_SIZE) {

		/*for (unsigned int op=0; op<nOutputPlanes; op++) thread */
		{
			int op = lid;
			int rem = wsz - xi0;
			__syncthreads();
			if (lid < nInputPlanes/2) {
				int bi;
				int lid2 = lid*2;
				for (bi=0; bi<BLOCK_SIZE; bi++) {
					int xi = xi0 + bi;
					if (xi == wsz) {
						break;
					}

					/* load to shared */
					*(float2*)&in_block0[bi*nInputPlanes + lid2] = *(float2*)&in01[xi*nInputPlanes + lid2];
					*(float2*)&in_block1[bi*nInputPlanes + lid2] = *(float2*)&in11[xi*nInputPlanes + lid2];
					*(float2*)&in_block2[bi*nInputPlanes + lid2] = *(float2*)&in21[xi*nInputPlanes + lid2];
				}

				{
					int xi = xi0 + bi;
					if (xi == wsz) {
						*(float2*)&in_block0[bi*(int)nInputPlanes + lid2] = *(float2*)&in01[(xi-1)*(int)nInputPlanes + lid2];
						*(float2*)&in_block1[bi*(int)nInputPlanes + lid2] = *(float2*)&in11[(xi-1)*(int)nInputPlanes + lid2];
						*(float2*)&in_block2[bi*(int)nInputPlanes + lid2] = *(float2*)&in21[(xi-1)*(int)nInputPlanes + lid2];
					} else {
						*(float2*)&in_block0[bi*(int)nInputPlanes + lid2] = *(float2*)&in01[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block1[bi*(int)nInputPlanes + lid2] = *(float2*)&in11[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block2[bi*(int)nInputPlanes + lid2] = *(float2*)&in21[xi*(int)nInputPlanes + lid2];
					}
				}

				{
					int xi = xi0-1;
					if (xi == -1) {
						*(float2*)&in_block0[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in01[lid2];
						*(float2*)&in_block1[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in11[lid2];
						*(float2*)&in_block2[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in21[lid2];
					} else {
						*(float2*)&in_block0[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in01[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block1[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in11[xi*(int)nInputPlanes + lid2];
						*(float2*)&in_block2[-1*(int)nInputPlanes + (int)lid2] = *(float2*)&in21[xi*(int)nInputPlanes + lid2];
					}
				}
			}
			__syncthreads();

			if (rem >= BLOCK_SIZE) {
#define DECL_PTR(y,x)		float *p##y##x = &in_block##y[nInputPlanes * (x-1)];

				UNROLL10x3(DECL_PTR);

				float sum0 = 0;
				float sum1 = 0;
				float sum2 = 0;
				float sum3 = 0;

				float sum4 = 0;
				float sum5 = 0;
				float sum6 = 0;
				float sum7 = 0;

				{
					const float *w0 = weight + lid;

					for (int ip = 0; ip < nInputPlanes; ip++) {
#define LOAD_INPUT2(y,x)			float2 i##y##x##_2 = *(float2*)&p##y##x[ip];

						UNROLL10x3(LOAD_INPUT2);

#define LOAD_COEF(X)				float w_##X = w[X * 128];

#define CALC(IDX,Y,I0,I1,I2,I3,I4,I5,I6,I7)				\
						sum0 += w_##IDX * i##Y##I0; \
						sum1 += w_##IDX * i##Y##I1; \
						sum2 += w_##IDX * i##Y##I2; \
						sum3 += w_##IDX * i##Y##I3; \
						sum4 += w_##IDX * i##Y##I4; \
						sum5 += w_##IDX * i##Y##I5; \
						sum6 += w_##IDX * i##Y##I6; \
						sum7 += w_##IDX * i##Y##I7;


						{
#define LOAD_INPUT1X(Y,X)				float i##Y##X = i##Y##X##_2.x;

							UNROLL10x3(LOAD_INPUT1X);

							const float *w = (w0 + (ip * 128) * 9);
							UNROLL9(LOAD_COEF);

							{
								CALC(0,0,0,1,2,3,4,5,6,7);
								CALC(1,0,1,2,3,4,5,6,7,8);
								CALC(2,0,2,3,4,5,6,7,8,9);

								CALC(3,1,0,1,2,3,4,5,6,7);
								CALC(4,1,1,2,3,4,5,6,7,8);
								CALC(5,1,2,3,4,5,6,7,8,9);

								CALC(6,2,0,1,2,3,4,5,6,7);
								CALC(7,2,1,2,3,4,5,6,7,8);
								CALC(8,2,2,3,4,5,6,7,8,9);
							}
						}

						ip++;
						{
#define LOAD_INPUT1Y(Y,X)				float i##Y##X = i##Y##X##_2.y;

							UNROLL10x3(LOAD_INPUT1Y);

							const float *w = (w0 + (ip * 128) * 9);
							UNROLL9(LOAD_COEF);

							{
								CALC(0,0,0,1,2,3,4,5,6,7);
								CALC(1,0,1,2,3,4,5,6,7,8);
								CALC(2,0,2,3,4,5,6,7,8,9);

								CALC(3,1,0,1,2,3,4,5,6,7);
								CALC(4,1,1,2,3,4,5,6,7,8);
								CALC(5,1,2,3,4,5,6,7,8,9);

								CALC(6,2,0,1,2,3,4,5,6,7);
								CALC(7,2,1,2,3,4,5,6,7,8);
								CALC(8,2,2,3,4,5,6,7,8,9);
							}
						}

					}

#define RELU(BI)							\
					{				\
						float *out = packed_output + (yi*wsz + (xi0+BI))*nOutputPlanes; \
									\
						{			\
							int opIndex = lid; \
							float v = sum##BI; \
							v += bv;	\
									\
							float mtz = max(v, 0.0f); \
							float ltz = min(v, 0.0f); \
									\
							v = ltz * 0.1f + mtz; \
									\
							out[opIndex] = v; \
						}			\
					}

					UNROLL8(RELU);

#undef DECL_PTR
#undef LOAD_COEF
#undef CALC
#undef LOAD_INPUT2
#undef LOAD_INPUT1X
#undef LOAD_INPUT1Y
#undef RELU
				}
			} else {
				for (int bi=0; bi<BLOCK_SIZE; bi++) {
					int xi = xi0+bi;
					if (xi == wsz) {
						break;
					}

					const float *w0 = weight + lid;
					float sum = 0;

					for (int ip=0; ip<nInputPlanes; ip++) {
						float i00, i01, i02;
						float i10, i11, i12;
						float i20, i21, i22;

						i00 = in_block0[(bi-1)*nInputPlanes+ip];
						i10 = in_block1[(bi-1)*nInputPlanes+ip];
						i20 = in_block2[(bi-1)*nInputPlanes+ip];

						i01 = in_block0[bi*nInputPlanes+ip];
						i11 = in_block1[bi*nInputPlanes+ip];
						i21 = in_block2[bi*nInputPlanes+ip];

						i02 = in_block0[(bi+1)*nInputPlanes+ip];
						i12 = in_block1[(bi+1)*nInputPlanes+ip];
						i22 = in_block2[(bi+1)*nInputPlanes+ip];

						const float *w = w0;
						sum += w[(9*ip+0) * 128]*i00;
						sum += w[(9*ip+1) * 128]*i01;
						sum += w[(9*ip+2) * 128]*i02;

						sum += w[(9*ip+3) * 128]*i10;
						sum += w[(9*ip+4) * 128]*i11;
						sum += w[(9*ip+5) * 128]*i12;

						sum += w[(9*ip+6) * 128]*i20;
						sum += w[(9*ip+7) * 128]*i21;
						sum += w[(9*ip+8) * 128]*i22;
					}

					float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
					{
						float v = sum;
						v += bv;

						float mtz = max(v, 0.0f);
						float ltz = min(v, 0.0f);

						v = ltz * 0.1f + mtz;
						out[op] = v;
					}
				}
			}
		}
	}
}

extern "C" __global__ void
filter_i32(const float * __restrict__ packed_input,
	   float * __restrict__ packed_output,
	   int nOutputPlanes,
	   const float * __restrict__ biases,
	   unsigned int hsz,
	   unsigned int wsz,
	   const float * __restrict__ weight)
{
	filter<32>(packed_input, packed_output, nOutputPlanes, biases, hsz, wsz, weight);
}

extern "C" __global__ void
filter_i64(const float * __restrict__ packed_input,
	   float * __restrict__ packed_output,
	   int nOutputPlanes,
	   const float * __restrict__ biases,
	   unsigned int hsz,
	   unsigned int wsz,
	   const float * __restrict__ weight)
{
	filter<64>(packed_input, packed_output, nOutputPlanes, biases, hsz, wsz, weight);
}

extern "C" __global__ void
filter_i128(const float * __restrict__ packed_input,
	    float * __restrict__ packed_output,
	    int nOutputPlanes,
	    const float * __restrict__ biases,
	    unsigned int hsz,
	    unsigned int wsz,
	    const float * __restrict__ weight)
{
	filter<128>(packed_input, packed_output, nOutputPlanes, biases, hsz, wsz, weight);
}

#if __CUDA_ARCH__ >= 300
static inline __device__ float
warp_sum(float v) {
	v += __shfl_down_sync(0xFFFFFFFF, v, 1);
	v += __shfl_down_sync(0xFFFFFFFF, v, 2);
	v += __shfl_down_sync(0xFFFFFFFF, v, 4);
	v += __shfl_down_sync(0xFFFFFFFF, v, 8);
	v += __shfl_down_sync(0xFFFFFFFF, v, 16);
	return v;
}
#endif

template <int nInputPlanes,
	  int nOutputPlanes>
void __device__
filter_weight_blocking(const float * __restrict__ packed_input,
		       float * __restrict__ packed_output,
		       const float * __restrict__ biases,
		       unsigned int hsz,
		       unsigned int wsz,
		       const float * __restrict__ weight,
		       int ib0,
		       int ob0)
{
#define INPUT_BLOCK_SIZE 32
#define OUTPUT_BLOCK_SIZE 64 // == blockDim.x
#define X_BLOCK_SIZE 8
	unsigned int yi = blockIdx.x;

	size_t in_step = wsz * nInputPlanes;
	const float *inp = packed_input;
	inp += yi * in_step;

	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	__shared__ float shared_buf_base[INPUT_BLOCK_SIZE * (X_BLOCK_SIZE+2) * 3];
	float *in_block0_base = shared_buf_base + INPUT_BLOCK_SIZE * (BLOCK_SIZE+2) * 0;
	float *in_block1_base = shared_buf_base + INPUT_BLOCK_SIZE * (BLOCK_SIZE+2) * 1;
	float *in_block2_base = shared_buf_base + INPUT_BLOCK_SIZE * (BLOCK_SIZE+2) * 2;

	float *in_block0 = in_block0_base + INPUT_BLOCK_SIZE;
	float *in_block1 = in_block1_base + INPUT_BLOCK_SIZE;
	float *in_block2 = in_block2_base + INPUT_BLOCK_SIZE;
	int lid = threadIdx.x;

	{ // ib0
		{ // ob0
			int op = lid + ob0;

			float bv = biases[op];

			for (int xi0=0; xi0<wsz; xi0+=BLOCK_SIZE) {
				float *out_base = packed_output + (yi*wsz + xi0)*nOutputPlanes + op;

				float *linp0 = in_block0 + lid;
				float *linp1 = in_block1 + lid;
				float *linp2 = in_block2 + lid;

				__syncthreads();
				int rem = wsz - xi0;
				const float *inb0 = in01 + ib0+lid;
				const float *inb1 = in11 + ib0+lid;
				const float *inb2 = in21 + ib0+lid;

				if (rem > 8 && xi0 != 0) {
					if (lid < INPUT_BLOCK_SIZE) {
						linp0[-1*INPUT_BLOCK_SIZE] = linp0[7*INPUT_BLOCK_SIZE];
						linp1[-1*INPUT_BLOCK_SIZE] = linp1[7*INPUT_BLOCK_SIZE];
						linp2[-1*INPUT_BLOCK_SIZE] = linp2[7*INPUT_BLOCK_SIZE];

						linp0[0*INPUT_BLOCK_SIZE] = linp0[8*INPUT_BLOCK_SIZE];
						linp1[0*INPUT_BLOCK_SIZE] = linp1[8*INPUT_BLOCK_SIZE];
						linp2[0*INPUT_BLOCK_SIZE] = linp2[8*INPUT_BLOCK_SIZE];
					}
					__syncthreads();

					if (lid < INPUT_BLOCK_SIZE) {
						int bi;
#pragma unroll
						for (bi=1; bi<X_BLOCK_SIZE+1; bi++) {
							int xi = xi0 + bi;
							/* load to shared */
							linp0[bi*INPUT_BLOCK_SIZE] = inb0[xi*nInputPlanes];
							linp1[bi*INPUT_BLOCK_SIZE] = inb1[xi*nInputPlanes];
							linp2[bi*INPUT_BLOCK_SIZE] = inb2[xi*nInputPlanes];
						}
					}

				} else {
					if (lid < INPUT_BLOCK_SIZE) {
						int bi;
						for (bi=0; bi<X_BLOCK_SIZE; bi++) {
							int xi = xi0 + bi;
							if (xi == wsz) {
								break;
							}

							/* load to shared */
							linp0[bi*INPUT_BLOCK_SIZE] = inb0[xi*nInputPlanes];
							linp1[bi*INPUT_BLOCK_SIZE] = inb1[xi*nInputPlanes];
							linp2[bi*INPUT_BLOCK_SIZE] = inb2[xi*nInputPlanes];
						}

						{
							int xi = xi0 + bi;
							if (xi == wsz) {
								linp0[bi*(int)INPUT_BLOCK_SIZE] = inb0[(xi-1)*(int)nInputPlanes];
								linp1[bi*(int)INPUT_BLOCK_SIZE] = inb1[(xi-1)*(int)nInputPlanes];
								linp2[bi*(int)INPUT_BLOCK_SIZE] = inb2[(xi-1)*(int)nInputPlanes];
							} else {
								linp0[bi*(int)INPUT_BLOCK_SIZE] = inb0[xi*(int)nInputPlanes];
								linp1[bi*(int)INPUT_BLOCK_SIZE] = inb1[xi*(int)nInputPlanes];
								linp2[bi*(int)INPUT_BLOCK_SIZE] = inb2[xi*(int)nInputPlanes];
							}
						}

						{
							int xi = xi0-1;
							if (xi == -1) {
								linp0[-1*(int)INPUT_BLOCK_SIZE] = inb0[0];
								linp1[-1*(int)INPUT_BLOCK_SIZE] = inb1[0];
								linp2[-1*(int)INPUT_BLOCK_SIZE] = inb2[0];
							} else {
								linp0[-1*(int)INPUT_BLOCK_SIZE] = inb0[xi*(int)nInputPlanes];
								linp1[-1*(int)INPUT_BLOCK_SIZE] = inb1[xi*(int)nInputPlanes];
								linp2[-1*(int)INPUT_BLOCK_SIZE] = inb2[xi*(int)nInputPlanes];
							}
						}
					}
				}
				__syncthreads();

				const float *w0 = weight + op;

				if (rem >= BLOCK_SIZE) {
#define DECL_PTR(y,x)			float *p##y##x = &in_block##y[INPUT_BLOCK_SIZE * (x-1)];
					UNROLL10x3(DECL_PTR);

					float sum0 = 0;
					float sum1 = 0;
					float sum2 = 0;
					float sum3 = 0;

					float sum4 = 0;
					float sum5 = 0;
					float sum6 = 0;
					float sum7 = 0;

					for (int ip1 = 0; ip1 < INPUT_BLOCK_SIZE; ip1+=2) {
						int ip = ip1 + ib0;

#define LOAD_INPUT2(y,x)			float2 i##y##x##_2 = *(float2*)&p##y##x[ip1];

						UNROLL10x3(LOAD_INPUT2);

#define LOAD_COEF(X)				float w_##X = w[X * 128];

#define CALC(SYM,IDX,Y,I0,I1,I2,I3,I4,I5,I6,I7)				\
						sum0 += w_##IDX * i##Y##I0##_2.SYM; \
						sum1 += w_##IDX * i##Y##I1##_2.SYM; \
						sum2 += w_##IDX * i##Y##I2##_2.SYM; \
						sum3 += w_##IDX * i##Y##I3##_2.SYM; \
						sum4 += w_##IDX * i##Y##I4##_2.SYM; \
						sum5 += w_##IDX * i##Y##I5##_2.SYM; \
						sum6 += w_##IDX * i##Y##I6##_2.SYM; \
						sum7 += w_##IDX * i##Y##I7##_2.SYM;


						{
							const float *w = (w0 + (ip * 128) * 9);
							UNROLL9(LOAD_COEF);

							{
								CALC(x, 0,0,0,1,2,3,4,5,6,7);
								CALC(x, 1,0,1,2,3,4,5,6,7,8);
								CALC(x, 2,0,2,3,4,5,6,7,8,9);

								CALC(x, 3,1,0,1,2,3,4,5,6,7);
								CALC(x, 4,1,1,2,3,4,5,6,7,8);
								CALC(x, 5,1,2,3,4,5,6,7,8,9);

								CALC(x, 6,2,0,1,2,3,4,5,6,7);
								CALC(x, 7,2,1,2,3,4,5,6,7,8);
								CALC(x, 8,2,2,3,4,5,6,7,8,9);
							}
						}

						ip++;
						{
							const float *w = (w0 + (ip * 128) * 9);
							UNROLL9(LOAD_COEF);

							{
								CALC(y, 0,0,0,1,2,3,4,5,6,7);
								CALC(y, 1,0,1,2,3,4,5,6,7,8);
								CALC(y, 2,0,2,3,4,5,6,7,8,9);

								CALC(y, 3,1,0,1,2,3,4,5,6,7);
								CALC(y, 4,1,1,2,3,4,5,6,7,8);
								CALC(y, 5,1,2,3,4,5,6,7,8,9);

								CALC(y, 6,2,0,1,2,3,4,5,6,7);
								CALC(y, 7,2,1,2,3,4,5,6,7,8);
								CALC(y, 8,2,2,3,4,5,6,7,8,9);
							}
						}
					}

#define RELU(BI)							\
					{				\
									\
						{			\
							float v = sum##BI + out_base[BI*nOutputPlanes]; \
							v += bv; \
									\
							float mtz = max(v, 0.0f); \
							float ltz = min(v, 0.0f); \
									\
							v = ltz * 0.1f + mtz; \
									\
							out_base[BI*nOutputPlanes] = v;	\
						}			\
					}

					if ((ib0+INPUT_BLOCK_SIZE) == nInputPlanes) {
						UNROLL8(RELU);
					} else if (ib0 == 0) {
						out_base[nOutputPlanes*0] = sum0;
						out_base[nOutputPlanes*1] = sum1;
						out_base[nOutputPlanes*2] = sum2;
						out_base[nOutputPlanes*3] = sum3;
						out_base[nOutputPlanes*4] = sum4;
						out_base[nOutputPlanes*5] = sum5;
						out_base[nOutputPlanes*6] = sum6;
						out_base[nOutputPlanes*7] = sum7;
					} else {
						out_base[nOutputPlanes*0] += sum0;
						out_base[nOutputPlanes*1] += sum1;
						out_base[nOutputPlanes*2] += sum2;
						out_base[nOutputPlanes*3] += sum3;
						out_base[nOutputPlanes*4] += sum4;
						out_base[nOutputPlanes*5] += sum5;
						out_base[nOutputPlanes*6] += sum6;
						out_base[nOutputPlanes*7] += sum7;
					}
				} else {
					for (int bi=0; bi<X_BLOCK_SIZE; bi++) {
						int xi = xi0+bi;
						if (xi == wsz) {
							break;
						}

						float sum = 0;

						for (int ip1=0; ip1<INPUT_BLOCK_SIZE; ip1++) {
							int ip = ib0 + ip1;

							float i00, i01, i02;
							float i10, i11, i12;
							float i20, i21, i22;

							i00 = in_block0[(bi-1)*INPUT_BLOCK_SIZE+ip1];
							i10 = in_block1[(bi-1)*INPUT_BLOCK_SIZE+ip1];
							i20 = in_block2[(bi-1)*INPUT_BLOCK_SIZE+ip1];

							i01 = in_block0[bi*INPUT_BLOCK_SIZE+ip1];
							i11 = in_block1[bi*INPUT_BLOCK_SIZE+ip1];
							i21 = in_block2[bi*INPUT_BLOCK_SIZE+ip1];

							i02 = in_block0[(bi+1)*INPUT_BLOCK_SIZE+ip1];
							i12 = in_block1[(bi+1)*INPUT_BLOCK_SIZE+ip1];
							i22 = in_block2[(bi+1)*INPUT_BLOCK_SIZE+ip1];

							sum += w0[(9*ip+0) * 128]*i00;
							sum += w0[(9*ip+1) * 128]*i01;
							sum += w0[(9*ip+2) * 128]*i02;

							sum += w0[(9*ip+3) * 128]*i10;
							sum += w0[(9*ip+4) * 128]*i11;
							sum += w0[(9*ip+5) * 128]*i12;

							sum += w0[(9*ip+6) * 128]*i20;
							sum += w0[(9*ip+7) * 128]*i21;
							sum += w0[(9*ip+8) * 128]*i22;
						}

						float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
						if ((ib0+INPUT_BLOCK_SIZE) == nInputPlanes) {
							/* last */
							float v = sum + out[op];
							v += bv;

							float mtz = max(v, 0.0f);
							float ltz = min(v, 0.0f);

							v = ltz * 0.1f + mtz;
							out[op] = v;
						} else if (ib0 == 0) {
							out[op] = sum;
						} else {
							out[op] += sum;
						}
					}
				}

			}

		}
	}

}

extern "C" __global__
void
filter_i128_o128(const float * __restrict__ packed_input,
		 float * __restrict__ packed_output,
		 const float * __restrict__ biases,
		 unsigned int hsz,
		 unsigned int wsz,
		 const float * __restrict__ weight,
		 int ib0,
		 int ob0)
{
	filter_weight_blocking<128,128>(packed_input,
					packed_output,
					biases,
					hsz,
					wsz,
					weight,
					ib0,
					ob0);
}

extern "C" __global__
void
filter_i64_o128(const float * __restrict__ packed_input,
		float * __restrict__ packed_output,
		const float * __restrict__ biases,
		unsigned int hsz,
		unsigned int wsz,
		const float * __restrict__ weight,
		int ib0,
		int ob0)
{
	filter_weight_blocking<64,128>(packed_input,
				       packed_output,
				       biases,
				       hsz,
				       wsz,
				       weight,
				       ib0,
				       ob0);
}


extern "C" __global__
void
filter_i64_o64(const float * __restrict__ packed_input,
	       float * __restrict__ packed_output,
	       const float * __restrict__ biases,
	       unsigned int hsz,
	       unsigned int wsz,
	       const float * __restrict__ weight,
	       int ib0,
	       int ob0)
{
	filter_weight_blocking<64,64>(packed_input,
				      packed_output,
				      biases,
				      hsz,
				      wsz,
				      weight,
				      ib0,
				      ob0);
}



extern "C" __global__ void
filter_i128_o1(const float * __restrict__ packed_input,
	       float * __restrict__ packed_output,
	       float * __restrict__ biases,
	       unsigned int hsz,
	       unsigned int wsz,
	       float * __restrict__ weight)
{
	int nInputPlanes = 128;
	int nOutputPlanes = 1;
	{
		unsigned int yi = blockIdx.x;

		size_t in_step = wsz * nInputPlanes;
		const float *inp = packed_input;
		inp += yi * in_step;

		const float *in0p = inp - in_step;
		if (yi == 0) {
			in0p = inp;
		}
		const float *in1p = inp;

		const float *in2p = inp + in_step;
		if (yi == hsz-1) {
			in2p = in1p;
		}

		const float *in01 = in0p;
		const float *in11 = in1p;
		const float *in21 = in2p;

		unsigned int lid = threadIdx.x;

		float bv0 = biases[0];

		/* 128 item */
		/* x      : (1width/group) */
		/* y      : (2height/group) */
		/* iplane : 1plane / 1item * 128plane */

		__shared__ float shared_buf[128 * 10];

		float lin00;
		float lin01;
		float lin02;

		float lin10;
		float lin11;
		float lin12;

		float lin20;
		float lin21;
		float lin22;

		float *sum_buffer = shared_buf + 128*9;

#define OUT1_LOAD_WEIGHT(I,Y,X) float w##I##Y##X = weight[(I*16 + lid)*9 + Y*3 + X];
		float w00 = weight[lid*9 + 0];
		float w01 = weight[lid*9 + 1];
		float w02 = weight[lid*9 + 2];
		float w10 = weight[lid*9 + 3];
		float w11 = weight[lid*9 + 4];
		float w12 = weight[lid*9 + 5];
		float w20 = weight[lid*9 + 6];
		float w21 = weight[lid*9 + 7];
		float w22 = weight[lid*9 + 8];

		const float *pin01 = in01 + lid;
		const float *pin02 = in01 + nInputPlanes + lid;

		const float *pin11 = in11 + lid;
		const float *pin12 = in11 + nInputPlanes + lid;

		const float *pin21 = in21 + lid;
		const float *pin22 = in21 + nInputPlanes + lid;

		lin01 = pin01[0];
		lin02 = pin01[0];

		lin11 = pin11[0];
		lin12 = pin11[0];

		lin21 = pin21[0];
		lin22 = pin21[0];

#define OUT1_BODY(LEDGE,REDGE,SUM_RELU)						\
		{							\
			float sum = 0;					\
			{						\
				lin00 = lin01;		\
				lin01 = lin02;		\
				     	     		\
				lin10 = lin11;		\
				lin11 = lin12;		\
				     	     		\
				lin20 = lin21;		\
				lin21 = lin22;		\
									\
				if (REDGE) {				\
					lin02 = lin01;			\
					lin12 = lin11;		\
					lin22 = lin21;		\
				} else {     			\
					lin02 = pin02[xi*128];	\
					lin12 = pin12[xi*128];	\
					lin22 = pin22[xi*128];	\
				}					\
									\
				sum += w00 * lin00;		\
				sum += w10 * lin10;		\
				sum += w20 * lin20;		\
						  			\
				sum += w01 * lin01;		\
				sum += w11 * lin11;		\
				sum += w21 * lin21;		\
						  			\
				sum += w02 * lin02;		\
				sum += w12 * lin12;		\
				sum += w22 * lin22;		\
									\
			}						\
			__syncthreads();				\
			sum_buffer[lid] = sum;				\
			__syncthreads();				\
			if (lid < 64) {					\
				float v2 = sum_buffer[lid+64];		\
				sum_buffer[lid] += v2;			\
			}						\
			__syncthreads();				\
			SUM_RELU(0);					\
		}

#if __CUDA_ARCH__ >= 300
#define SUM_RELU(OI)							\
		if (lid < 32) {						\
			float v0 = sum_buffer[lid] + sum_buffer[lid+32];			\
			float sum = warp_sum(v0);			\
									\
			if (lid == 0) {					\
				float v = sum;				\
				float *out = packed_output + (yi*wsz + xi)*nOutputPlanes; \
				v += bv##OI;				\
				float mtz = max(v, 0.0f);		\
				float ltz = min(v, 0.0f);		\
				v = ltz * 0.1f + mtz;			\
				out[OI] = v;				\
			}						\
		}							\

#else

#define SUM_RELU(OI)							\
		if (lid < 32) {						\
			sum_buffer[lid] += sum_buffer[lid+32];		\
		}							\
		__syncthreads();					\
		if (lid < 16) {						\
			sum_buffer[lid] += sum_buffer[lid+16];		\
		}							\
		__syncthreads();					\
		if (lid < 8) {						\
			sum_buffer[lid] += sum_buffer[lid+8];		\
		}							\
		__syncthreads();					\
		if (lid < 4) {						\
			sum_buffer[lid] += sum_buffer[lid+4];		\
		}							\
		__syncthreads();					\
		if (lid < 2) {						\
			sum_buffer[lid] += sum_buffer[lid+2];		\
		}							\
		__syncthreads();					\
		if (lid == 0) {						\
			float sum = sum_buffer[0] + sum_buffer[1];	\
			float v = sum;					\
			float *out = packed_output + (yi*wsz + xi)*nOutputPlanes; \
			v += bv##OI;					\
			float mtz = max(v, 0.0f);			\
			float ltz = min(v, 0.0f);			\
			v = ltz * 0.1f + mtz;				\
			out[OI] = v;					\
		}

#endif





		for (int xi=0; xi<wsz-1; xi++) {
			OUT1_BODY(0,0,SUM_RELU);
		}
		{
			int xi = wsz-1;
			OUT1_BODY(0,1,SUM_RELU);
		}
	}
}




extern "C" __global__ void
filter_i1_o32(const float * __restrict__ packed_input,
	      float * __restrict__ packed_output,
	      float * __restrict__ biases,
	      unsigned int hsz,
	      unsigned int wsz,
	      float * __restrict__ weight)
{
	//int nInputPlanes = 1;
	int nOutputPlanes = 32;

	unsigned int yi = blockIdx.x;
	unsigned int lid = threadIdx.x;

	size_t in_step = wsz;

	const float *inp = packed_input;
	inp += in_step * yi;
	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	__shared__ float in_block0_base[256+2];
	__shared__ float in_block1_base[256+2];
	__shared__ float in_block2_base[256+2];

	float *in_block0 = in_block0_base + 1;
	float *in_block1 = in_block1_base + 1;
	float *in_block2 = in_block2_base + 1;

	/* 256 item / group */
	/* x         : (64width/group) */
	/* 32 oplane : (8weight/item * 4item)*/
	unsigned int xoff = lid / 4U;
	unsigned int ooff = (lid % 4U) * 8;

#define IN1_LOAD_COEF(O,Y,X)				\
	float w##O##Y##X = weight[9 * (O + ooff) + (Y*3) + X];

	UNROLL8x3x3(IN1_LOAD_COEF);

	for (int xi0=0; xi0<wsz; xi0+=256) {
		/* load */
		__syncthreads();
		{
			int xi = xi0 + lid;

			if (xi < wsz) {
				in_block0[lid] = in01[xi0 + lid];
				in_block1[lid] = in11[xi0 + lid];
				in_block2[lid] = in21[xi0 + lid];

			}

			if (lid == 0) {
				if (xi == 0) {
					in_block0[-1] = in01[0];
					in_block1[-1] = in11[0];
					in_block2[-1] = in21[0];
				}  else {
					in_block0[-1] = in01[xi-1];
					in_block1[-1] = in11[xi-1];
					in_block2[-1] = in21[xi-1];
				}
			}

			if (xi == wsz-1) {
				in_block0[lid+1] = in01[xi];
				in_block1[lid+1] = in11[xi];
				in_block2[lid+1] = in21[xi];
			}

			if ((lid == 255) && (xi < wsz-1)) {
				in_block0[256] = in01[xi+1];
				in_block1[256] = in11[xi+1];
				in_block2[256] = in21[xi+1];
			}
		}
		__syncthreads();

		for (int xi1_base=0; xi1_base<4; xi1_base++) {
			{
				int xi1 = xi1_base*64 + xoff;

				int xi = xi0 + xi1;
				if (xi < wsz) {

#define IN1_DECLSUM(O)			float sum##O = 0;
#define IN1_CALC(O,Y,X)			sum##O += in_block##Y[xi1+X-1] * w##O##Y##X;
#define IN1_RELU(O)			{				\
						float v = sum##O;	\
						int opIndex = ooff + O;	\
						float bv = biases[opIndex]; \
						v += bv;		\
						float mtz = max(v, 0.0f); \
						float ltz = min(v, 0.0f); \
						v = ltz * 0.1f + mtz;	\
						out[opIndex] = v;	\
					}

					UNROLL8(IN1_DECLSUM);
					UNROLL8x3x3(IN1_CALC);
					float *out = packed_output + (yi*wsz + xi) * nOutputPlanes;
					UNROLL8(IN1_RELU);
				}
			}

		}

	}
}



/* blockDim.x == 192 */
extern "C" __global__ void
filter_i3_o32(const float * __restrict__ packed_input,
	      float * __restrict__ packed_output,
	      float * __restrict__ biases,
	      unsigned int hsz,
	      unsigned int wsz,
	      float * __restrict__ weight)
{
	int nInputPlanes = 3;
	int nOutputPlanes = 32;

	unsigned int yi = blockIdx.x;
	unsigned int lid = threadIdx.x;

	size_t in_step = wsz * nInputPlanes;

	const float *inp = packed_input;
	inp += in_step * yi;
	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	__shared__ float in_block0_base[(64+2)*3];
	__shared__ float in_block1_base[(64+2)*3];
	__shared__ float in_block2_base[(64+2)*3];
	__shared__ float sum_buffer[192];

	float *in_block0 = in_block0_base + 3;
	float *in_block1 = in_block1_base + 3;
	float *in_block2 = in_block2_base + 3;

	/* 192 item / group */
	/* load 192 item */

	/* 3 iplane  : */
	/* x         : (64width/group) */
	/* 32 oplane : (8weight/item * 4item)*/
	unsigned int ioff = lid / 32U;
	unsigned int ooff = lid % 32U;

#define I3_O32_LOAD_COEF(I)						\
	float w##I = weight[9*nOutputPlanes*ioff+ooff+I*nOutputPlanes];

	UNROLL9(I3_O32_LOAD_COEF);

	for (int xi0=0; xi0<wsz; xi0+=64) {
		/* load */
		int nelem = min(wsz - xi0, 64);
		int nload = nelem * 3;

		if (lid < nload) {
			int xi = xi0*3 + lid;

			in_block0[lid] = in01[xi];
			in_block1[lid] = in11[xi];
			in_block2[lid] = in21[xi];

			if (lid < 3) {
				if (xi <= 2) {
					/* left edge */
					in_block0[-3+(int)lid] = in01[lid];
					in_block1[-3+(int)lid] = in11[lid];
					in_block2[-3+(int)lid] = in21[lid];
				}  else {
					/* 0, 1, 2 */
					in_block0[-3+(int)lid] = in01[-3+(int)xi];
					in_block1[-3+(int)lid] = in11[-3+(int)xi];
					in_block2[-3+(int)lid] = in21[-3+(int)xi];
				}
			}

			if (xi >= wsz*3-3) {
				/* right edge */
				in_block0[lid+3] = in01[xi];
				in_block1[lid+3] = in11[xi];
				in_block2[lid+3] = in21[xi];
			} else if (lid >= 189) {
				/* 189, 190, 191 */
				in_block0[lid+3] = in01[xi+3];
				in_block1[lid+3] = in11[xi+3];
				in_block2[lid+3] = in21[xi+3];
			}
		}
		__syncthreads();

		for (int xi1=0; xi1<nelem; xi1++) {
			int xi = xi0 + xi1;

			if (lid < 96) { // 3input x 32output
				float sum = 0;

				sum += w0 * in_block0[(xi1 - 1)*3+(int)ioff];
				sum += w1 * in_block0[(xi1    )*3+(int)ioff];
				sum += w2 * in_block0[(xi1 + 1)*3+(int)ioff];

				sum += w3 * in_block1[(xi1 - 1)*3+(int)ioff];
				sum += w4 * in_block1[(xi1    )*3+(int)ioff];
				sum += w5 * in_block1[(xi1 + 1)*3+(int)ioff];

				sum += w6 * in_block2[(xi1 - 1)*3+(int)ioff];
				sum += w7 * in_block2[(xi1    )*3+(int)ioff];
				sum += w8 * in_block2[(xi1 + 1)*3+(int)ioff];

				sum_buffer[lid] = sum;
			}

			__syncthreads();

			if (lid < 32) {
				int oi = lid;
				float v = 0;
				float *out = packed_output + (yi*wsz + xi) * nOutputPlanes;

				/* 96 to 32 reduction */
				v += sum_buffer[32 * 0 + lid];
				v += sum_buffer[32 * 1 + lid];
				v += sum_buffer[32 * 2 + lid];

				float bv = biases[oi];
				v += bv;
				float mtz = max(v, 0.0f);
				float ltz = min(v, 0.0f);
				v = ltz * 0.1f + mtz;

				out[oi] = v;
			}

			__syncthreads();
		}
	}
}


/* blockDim.x == 128 */
extern "C" __global__ void
filter_i128_o3(const float * __restrict__ packed_input,
	       float * __restrict__ packed_output,
	       float * __restrict__ biases,
	       unsigned int hsz,
	       unsigned int wsz,
	       float * __restrict__ weight)
{
	int nInputPlanes = 128;
	int nOutputPlanes = 3;

	unsigned int yi = blockIdx.x;
	unsigned int lid = threadIdx.x;

	size_t in_step = wsz * nInputPlanes;

	const float *inp = packed_input;
	inp += in_step * yi;
	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	float lin00, lin01, lin02;
	float lin10, lin11, lin12;
	float lin20, lin21, lin22;

	__shared__ float sum_buffer[128];

	/* 128 item / group */
	/* load 128 item (load 3elem/item) */

	/* 128  iplane
	 * 1    input
	 * 3    output  (27coeff)
	 */

	int ioff = lid;
	float bv0 = biases[0];
	float bv1 = biases[1];
	float bv2 = biases[2];

#define I128_O3_LOAD_COEF(I)						\
	float w0##I = weight[9*0*nInputPlanes + I*nInputPlanes + ioff]; \
	float w1##I = weight[9*1*nInputPlanes + I*nInputPlanes + ioff]; \
	float w2##I = weight[9*2*nInputPlanes + I*nInputPlanes + ioff];

	UNROLL9(I128_O3_LOAD_COEF);

	lin01 = lin02 = in01[lid];
	lin11 = lin12 = in11[lid];
	lin21 = lin22 = in21[lid];

	int addroff = 0;
	char *p0 = (char*)(in01 + lid + nInputPlanes);
	char *p1 = (char*)(in11 + lid + nInputPlanes);
	char *p2 = (char*)(in21 + lid + nInputPlanes);

	for (int xi=0; xi<wsz; xi++) {
		lin00 = lin01;
		lin01 = lin02;

		lin10 = lin11;
		lin11 = lin12;

		lin20 = lin21;
		lin21 = lin22;

		if (xi == wsz-1) {
			/* nop */
		} else {
			lin02 = *(float *)(p0 + addroff);
			lin12 = *(float *)(p1 + addroff);
			lin22 = *(float *)(p2 + addroff);
		}
		addroff += nInputPlanes * sizeof(float);

#define I128_O3(OI)							\
		{							\
			float sum = 0;					\
			sum += w##OI##0 * lin00; \
			sum += w##OI##1 * lin01;		\
			sum += w##OI##2 * lin02; \
									\
			sum += w##OI##3 * lin10; \
			sum += w##OI##4 * lin11;		\
			sum += w##OI##5 * lin12; \
									\
			sum += w##OI##6 * lin20; \
			sum += w##OI##7 * lin21;		\
			sum += w##OI##8 * lin22; \
									\
			__syncthreads();				\
			sum_buffer[lid] = sum;				\
									\
			/* 128 to 1 */					\
			__syncthreads();				\
			if (lid < 64) {					\
				sum_buffer[lid] += sum_buffer[lid + 64]; \
			}						\
			__syncthreads();				\
									\
			SUM_RELU(OI);					\
		}

		I128_O3(0);
		I128_O3(1);
		I128_O3(2);
	}
}
