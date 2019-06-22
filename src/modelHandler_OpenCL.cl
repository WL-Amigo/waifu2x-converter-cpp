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

/* -*- mode: c -*- */

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


#define SUM_RELU(OI)							\
	if (lid < 32) {							\
		sum_buffer[lid] += sum_buffer[lid+32];			\
	}								\
	barrier(CLK_LOCAL_MEM_FENCE);					\
	if (lid < 16) {							\
		sum_buffer[lid] += sum_buffer[lid+16];			\
	}								\
	barrier(CLK_LOCAL_MEM_FENCE);					\
	if (lid < 8) {							\
		sum_buffer[lid] += sum_buffer[lid+8];			\
	}								\
	barrier(CLK_LOCAL_MEM_FENCE);					\
	if (lid < 4) {							\
		sum_buffer[lid] += sum_buffer[lid+4];			\
	}								\
	barrier(CLK_LOCAL_MEM_FENCE);					\
	if (lid < 2) {							\
		sum_buffer[lid] += sum_buffer[lid+2];			\
	}								\
	barrier(CLK_LOCAL_MEM_FENCE);					\
	if (lid == 0) {							\
		float sum = sum_buffer[0] + sum_buffer[1];		\
		float v = sum;						\
		__global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes; \
		v += bv##OI;						\
		float mtz = max(v, 0.0f);			\
		float ltz = min(v, 0.0f);			\
		v = ltz * 0.1f + mtz;				\
		out[OI] = v;					\
	}

/* a += b*c */
#define CUM(a,b,c) a = mad(b,c,a)

#define BLOCK_SIZE 8

__kernel void
filter_in1_out32(__global const float * __restrict__ packed_input,
		 int nInputPlanes,
		 __global float * __restrict__ packed_output,
		 int nOutputPlanes,
		 __global float * __restrict__ biases,
		 unsigned int hsz,
		 unsigned int wsz,
		 __global float * __restrict__ weight)
{
	unsigned int yi = get_group_id(0);
	unsigned int lid = get_local_id(0);

	size_t in_step = wsz * sizeof(float) * nInputPlanes;

	__global char *inp = (__global char*)packed_input;

	inp += in_step*yi;
	__global char *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}

	__global char *in1p = inp;
	__global char *in2p = inp + in_step;

	if (yi == hsz-1) {
		in2p = inp;
	}

	__global float *in01 = (__global float*)in0p;
	__global float *in11 = (__global float*)in1p;
	__global float *in21 = (__global float*)in2p;

	__local float in_block0_base[256+2];
	__local float in_block1_base[256+2];
	__local float in_block2_base[256+2];

	__local float *in_block0 = in_block0_base + 1;
	__local float *in_block1 = in_block1_base + 1;
	__local float *in_block2 = in_block2_base + 1;

	unsigned int vec_width = min((int)128, (int)nOutputPlanes);

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
		barrier(CLK_LOCAL_MEM_FENCE);
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
		barrier(CLK_LOCAL_MEM_FENCE);

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
					__global float *out = packed_output + (yi*wsz + xi) * 32;
					UNROLL8(IN1_RELU);
				}
			}

		}

	}
}

__kernel void
filter_in128_out1(__global const float * __restrict__ packed_input,
		  int nInputPlanes,
		  __global float * __restrict__ packed_output,
		  int nOutputPlanes,
		  __global float * __restrict__ biases,
		  unsigned int hsz,
		  unsigned int wsz,
		  __global float * __restrict__ weight)
{
	unsigned int yi_base = get_group_id(0)*1;
	__local float sum_buffer[128];

	for (int yi0=0; yi0<1; yi0++) {
		int yi = yi_base + yi0;
		unsigned int lid = get_local_id(0);

		size_t in_step = wsz * sizeof(float) * nInputPlanes;

		__global char *inp = (__global char*)packed_input;

		inp += in_step*yi;
		__global char *in0p = inp - in_step;
		if (yi == 0) {
			in0p = inp;
		}

		__global char *in1p = inp;
		__global char *in2p = inp + in_step;

		if (yi == hsz-1) {
			in2p = in1p;
		}

		__global float *in01 = (__global float*)in0p;
		__global float *in11 = (__global float*)in1p;
		__global float *in21 = (__global float*)in2p;

		float bv = biases[0];

		/* 128 item */
		/* x      : (1width/group) */
		/* y      : (2height/group) */
		/* iplane : 1plane / 1item * 128plane */

		float lin00;
		float lin01;
		float lin02;

		float lin10;
		float lin11;
		float lin12;

		float lin20;
		float lin21;
		float lin22;

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

		__global float *pin00 = in01 - 128 + lid;
		__global float *pin01 = in01 + lid;
		__global float *pin02 = in01 + 128 + lid;

		__global float *pin10 = in11 - 128 + lid;
		__global float *pin11 = in11 + lid;
		__global float *pin12 = in11 + 128 + lid;

		__global float *pin20 = in21 - 128 + lid;
		__global float *pin21 = in21 + lid;
		__global float *pin22 = in21 + 128 + lid;

		lin01 = pin01[0];
		lin02 = pin01[0];

		lin11 = pin11[0];
		lin12 = pin11[0];

		lin21 = pin21[0];
		lin22 = pin21[0];

#define OUT1_BODY(LEDGE,REDGE)						\
		{							\
			float sum = 0;					\
			{						\
				int i = lid;				\
				float tmp0 = lin00;			\
				float tmp1 = lin10;			\
				float tmp2 = lin20;			\
									\
				lin00 = lin01; lin01 = lin02; lin02 = tmp0; \
				lin10 = lin11; lin11 = lin12; lin12 = tmp1; \
				lin20 = lin21; lin21 = lin22; lin22 = tmp2; \
									\
				if (REDGE) {				\
					lin02 = lin01;			\
					lin12 = lin11;			\
					lin22 = lin21;			\
				} else {				\
					lin02 = pin02[xi*128];		\
					lin12 = pin12[xi*128];		\
					lin22 = pin22[xi*128];		\
				}					\
									\
				CUM(sum, w00, lin00);			\
				CUM(sum, w10, lin10);			\
				CUM(sum, w20, lin20);			\
									\
				CUM(sum, w01, lin01);			\
				CUM(sum, w11, lin11);			\
				CUM(sum, w21, lin21);			\
									\
				CUM(sum, w02, lin02);			\
				CUM(sum, w12, lin12);			\
				CUM(sum, w22, lin22);			\
									\
			}						\
			barrier(CLK_LOCAL_MEM_FENCE);			\
			sum_buffer[lid] = sum;				\
			barrier(CLK_LOCAL_MEM_FENCE);			\
			if (lid < 64) {					\
				sum_buffer[lid] += sum_buffer[lid+64];	\
			}						\
			barrier(CLK_LOCAL_MEM_FENCE);			\
			if (lid < 32) {					\
				sum_buffer[lid] += sum_buffer[lid+32];	\
			}						\
			barrier(CLK_LOCAL_MEM_FENCE);			\
									\
									\
			if (lid == 0) {					\
				float sum = 0;				\
				for (int i=0; i<32; i++) {		\
					sum += sum_buffer[i];		\
				}					\
									\
				float v = sum;				\
				__global float *out = packed_output + (yi*wsz + xi); \
				v += bv;				\
				float mtz = max(v, 0.0f);		\
				float ltz = min(v, 0.0f);		\
				v = ltz * 0.1f + mtz;			\
				out[0] = v;				\
			}						\
		}

		for (int xi=0; xi<wsz-1; xi++) {
			OUT1_BODY(0,0);
		}
		{
			int xi = wsz-1;
			OUT1_BODY(0,1);
		}

	}
}


/* item/group == 192 */
__kernel void
filter_in3_out32(__global const float * __restrict__ packed_input,
		 __global float * __restrict__ packed_output,
		 __global float * __restrict__ biases,
		 unsigned int hsz,
		 unsigned int wsz,
		 __global float * __restrict__ weight)
{
	int nInputPlanes = 3;
	int nOutputPlanes = 32;

	unsigned int yi = get_group_id(0);
	unsigned int lid = get_local_id(0);

	size_t in_step = wsz * nInputPlanes;

	__global const float *inp = packed_input;
	inp += in_step * yi;
	__global const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	__global const float *in1p = inp;

	__global const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	__global const float *in01 = in0p;
	__global const float *in11 = in1p;
	__global const float *in21 = in2p;

	__local float in_block0_base[(64+2)*3];
	__local float in_block1_base[(64+2)*3];
	__local float in_block2_base[(64+2)*3];
	__local float sum_buffer[192];

	__local float *in_block0 = in_block0_base + 3;
	__local float *in_block1 = in_block1_base + 3;
	__local float *in_block2 = in_block2_base + 3;

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
		int nelem = min((int)wsz - xi0, 64);
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
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int xi1=0; xi1<nelem; xi1++) {
			int xi = xi0 + xi1;

			if (lid < 96) { // 3input x 32output
				float sum = 0;

				CUM(sum, w0, in_block0[(xi1 - 1)*3+(int)ioff]);
				CUM(sum, w1, in_block0[(xi1    )*3+(int)ioff]);
				CUM(sum, w2, in_block0[(xi1 + 1)*3+(int)ioff]);

				CUM(sum, w3, in_block1[(xi1 - 1)*3+(int)ioff]);
				CUM(sum, w4, in_block1[(xi1    )*3+(int)ioff]);
				CUM(sum, w5, in_block1[(xi1 + 1)*3+(int)ioff]);

				CUM(sum, w6, in_block2[(xi1 - 1)*3+(int)ioff]);
				CUM(sum, w7, in_block2[(xi1    )*3+(int)ioff]);
				CUM(sum, w8, in_block2[(xi1 + 1)*3+(int)ioff]);

				sum_buffer[lid] = sum;
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (lid < 32) {
				int oi = lid;
				float v = 0;
				__global float *out = packed_output + (yi*wsz + xi) * nOutputPlanes;

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

			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}


/* item / group == 128 */
__kernel 
__attribute__((reqd_work_group_size(128, 1, 1)))
void
filter_in128_out3(__global const float * __restrict__ packed_input,
		  __global float * __restrict__ packed_output,
		  __global float * __restrict__ biases,
		  unsigned int hsz,
		  unsigned int wsz,
		  __global float * __restrict__ weight)
{
	int nInputPlanes = 128;
	int nOutputPlanes = 3;

	unsigned int yi = get_group_id(0);
	unsigned int lid = get_local_id(0);
	int slid = get_local_id(0);

	size_t in_step = wsz * nInputPlanes;

	__global const float *inp = packed_input;
	inp += in_step * yi;
	__global const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}
	__global const float *in1p = inp;

	__global const float *in2p = inp + in_step;
	if (yi == hsz-1) {
		in2p = in1p;
	}

	__global const float *in01 = in0p;
	__global const float *in11 = in1p;
	__global const float *in21 = in2p;

	/* 5120byte */
	float lin00, lin01, lin02;
	float lin10, lin11, lin12;
	float lin20, lin21, lin22;

	__local float sum_buffer[3][128];

	/* 128 item / group */
	/* load 128 item (load 3elem/item) */

	/* 128  iplane
	 * 1    input
	 * 3    output  (27coeff)
	 */

	int ioff = lid;
	float bv = 0;

	int reduce_oi_32 = lid / 32U;
	int reduce_li_32 = lid % 32U;

	int reduce_oi_16 = lid / 16U;
	int reduce_li_16 = lid % 16U;

	int reduce_oi_8 = lid / 8U;
	int reduce_li_8 = lid % 8U;

	if (lid < 3) {
		bv = biases[lid];
	}

#define I128_O3_LOAD_COEF(I)						\
	float w0##I = weight[9*0*nInputPlanes + I*nInputPlanes + ioff]; \
	float w1##I = weight[9*1*nInputPlanes + I*nInputPlanes + ioff]; \
	float w2##I = weight[9*2*nInputPlanes + I*nInputPlanes + ioff];

	UNROLL9(I128_O3_LOAD_COEF);

	lin01 = lin02 = in01[lid];
	lin11 = lin12 = in11[lid];
	lin21 = lin22 = in21[lid];

	__global char *p0 = (__global char*)(in01 + lid + nInputPlanes);
	__global char *p1 = (__global char*)(in11 + lid + nInputPlanes);
	__global char *p2 = (__global char*)(in21 + lid + nInputPlanes);

	int addroff = 0;

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
			lin02 = *(__global float *)(p0 + addroff);
			lin12 = *(__global float *)(p1 + addroff);
			lin22 = *(__global float *)(p2 + addroff);
		}
		addroff += nInputPlanes * sizeof(float);

#define I128_O3(OI)							\
		{							\
			float sum = 0;					\
			CUM(sum, w##OI##0, lin00);			\
			CUM(sum, w##OI##1, lin01);		\
			CUM(sum, w##OI##2, lin02); \
						   \
			CUM(sum, w##OI##3, lin10); \
			CUM(sum, w##OI##4, lin11);		\
			CUM(sum, w##OI##5, lin12); \
						   \
			CUM(sum, w##OI##6, lin20); \
			CUM(sum, w##OI##7, lin21);		\
			CUM(sum, w##OI##8, lin22);			\
									\
			barrier(CLK_LOCAL_MEM_FENCE);			\
			sum_buffer[OI][lid] = sum;				\
			barrier(CLK_LOCAL_MEM_FENCE);			\
			if (lid < 64) {					\
				sum_buffer[OI][lid] += sum_buffer[OI][lid+64]; \
			}						\
			barrier(CLK_LOCAL_MEM_FENCE);			\
		}

		I128_O3(0);
		I128_O3(1);
		I128_O3(2);


		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 32*3) { /* 64x3 to 32x3 */
			sum_buffer[reduce_oi_32][reduce_li_32] += sum_buffer[reduce_oi_32][reduce_li_32+32];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < 16*3) { /* 32x3 to 16x3 */
			sum_buffer[reduce_oi_16][reduce_li_16] += sum_buffer[reduce_oi_16][reduce_li_16+16];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < 3) {
			float sum = 0;
			for (int i=0; i<16; i++) {
				sum += sum_buffer[lid][i];
			}

			float v = sum;
			__global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
			v += bv;
			float mtz = max(v, 0.0f);
			float ltz = min(v, 0.0f);
			v = ltz * 0.1f + mtz;
			out[lid] = v;
		}

	}
}


__kernel void
filter_intel_gen(__global const float * __restrict__ fin,
		 unsigned int nInputPlanes,
		 __global float * __restrict__ foutput,
		 unsigned int nOutputPlanes,
		 __global float * __restrict__ fbiases,
		 unsigned int hsz,
		 unsigned int wsz,
		 __global float * __restrict__ fw)
{
    __global const unsigned char *in = (__global unsigned char*)fin;
    __global const unsigned char *w = (__global unsigned char*)fw;
    __global const unsigned char *biases = (__global unsigned char*)fbiases;
    __global unsigned char *output = (__global unsigned char*)foutput;

    unsigned int OP_BLOCK_SIZE = 32;
    unsigned int IP_BLOCK_SIZE = 32;

    int nOutputPlane_block = nOutputPlanes / OP_BLOCK_SIZE;
    int nInputPlane_block = nInputPlanes / IP_BLOCK_SIZE;

    for (int dposy=0; dposy<3; dposy++) {
        bool dposy_last = dposy==2;
        bool dopsy_first = dposy==0;
    }
	
}


__kernel void
filter(__global const float * __restrict__ packed_input,
       int nInputPlanes,
       __global float * __restrict__ packed_output,
       int nOutputPlanes,
       __global float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       __global float * __restrict__ weight)
{
	unsigned int yi = get_group_id(0);
	unsigned int lid = get_local_id(0);

	__local float2 local_mem_base[(128/2) * (BLOCK_SIZE+2) * 3];
	__local float *local_mem = (__local float*)local_mem_base;

	__global const float * __restrict__ in = packed_input;
	size_t in_step = wsz * sizeof(float) * nInputPlanes;

	__global char *inp = (__global char*)packed_input;

	inp += in_step*yi;
	__global char *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}

	__global char *in1p = inp;
	__global char *in2p = inp + in_step;

	if (yi == hsz-1) {
		in2p = inp;
	}

	__global float *in01 = (__global float*)in0p;
	__global float *in11 = (__global float*)in1p;
	__global float *in21 = (__global float*)in2p;

	__local float *in_block0_base = local_mem;
	local_mem += nInputPlanes * (BLOCK_SIZE+2);
	__local float *in_block1_base = local_mem;
	local_mem += nInputPlanes * (BLOCK_SIZE+2);
	__local float *in_block2_base = local_mem;
	local_mem += nInputPlanes * (BLOCK_SIZE+2);

	__local float *in_block0 = in_block0_base+ nInputPlanes;
	__local float *in_block1 = in_block1_base+ nInputPlanes;
	__local float *in_block2 = in_block2_base+ nInputPlanes;

	unsigned int vec_width = min((int)128, (int)nOutputPlanes);

	for (int xi0=0; xi0<wsz; xi0+=BLOCK_SIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < nInputPlanes) {
			int bi;

			for (bi=0; bi<BLOCK_SIZE; bi++) {
				int xi = xi0 + bi;

				if (xi == wsz) {
					break;
				}

				in_block0[bi*nInputPlanes + lid] = in01[xi*nInputPlanes + lid];
				in_block1[bi*nInputPlanes + lid] = in11[xi*nInputPlanes + lid];
				in_block2[bi*nInputPlanes + lid] = in21[xi*nInputPlanes + lid];
			}

			{
				int xi = xi0 + bi;
				if (xi == wsz) {
					in_block0[bi*(int)nInputPlanes + lid] = in01[(xi-1)*(int)nInputPlanes + lid];
					in_block1[bi*(int)nInputPlanes + lid] = in11[(xi-1)*(int)nInputPlanes + lid];
					in_block2[bi*(int)nInputPlanes + lid] = in21[(xi-1)*(int)nInputPlanes + lid];
				} else {
					in_block0[bi*(int)nInputPlanes + lid] = in01[xi*(int)nInputPlanes + lid];
					in_block1[bi*(int)nInputPlanes + lid] = in11[xi*(int)nInputPlanes + lid];
					in_block2[bi*(int)nInputPlanes + lid] = in21[xi*(int)nInputPlanes + lid];
				}
			}

			{
				int xi = xi0-1;
				if (xi == -1) {
					in_block0[-1*(int)nInputPlanes + (int)lid] = in01[lid];
					in_block1[-1*(int)nInputPlanes + (int)lid] = in11[lid];
					in_block2[-1*(int)nInputPlanes + (int)lid] = in21[lid];
				} else {
					in_block0[-1*(int)nInputPlanes + (int)lid] = in01[xi*(int)nInputPlanes + lid];
					in_block1[-1*(int)nInputPlanes + (int)lid] = in11[xi*(int)nInputPlanes + lid];
					in_block2[-1*(int)nInputPlanes + (int)lid] = in21[xi*(int)nInputPlanes + lid];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		int rem = wsz - xi0;
		int vec_width4 = 128 * 4;

		if (rem >= BLOCK_SIZE) {
			{
				int bi = 0;
#define DECL_PTR(y,x)							\
				__local char *p##y##x = (__local char*)&in_block##y[nInputPlanes * (x-1)];

				UNROLL10x3(DECL_PTR);

				float intermediate_reg0 = 0;
				float intermediate_reg1 = 0;
				float intermediate_reg2 = 0;
				float intermediate_reg3 = 0;

				float intermediate_reg4 = 0;
				float intermediate_reg5 = 0;
				float intermediate_reg6 = 0;
				float intermediate_reg7 = 0;

				/*if (lid < nOutputPlanes)*/
				{
					__global float *w0 = weight + lid;
					int nInputPlanes4 = nInputPlanes * 4;

					for (int ipIndex4 = 0; ipIndex4 < nInputPlanes4; ipIndex4+=4) {

#define LOAD_INPUT2(y,x)					\
						float2 i##y##x##_2 = *(__local float2*)(p##y##x + ipIndex4);

						UNROLL10x3(LOAD_INPUT2);

#define LOAD_COEF(X)							\
						float w_##X = *(__global float*)(w + X * vec_width4);


						{
#define LOAD_INPUT1X(Y,X)						\
							float i##Y##X = i##Y##X##_2.x;

							UNROLL10x3(LOAD_INPUT1X);

							__global char *w = ((__global char*)w0 + (ipIndex4 * 128) * 9);
							UNROLL9(LOAD_COEF);

#define CALC(IDX,Y,I0,I1,I2,I3,I4,I5,I6,I7)				\
							CUM(intermediate_reg0, w_##IDX, i##Y##I0); \
							CUM(intermediate_reg1, w_##IDX, i##Y##I1); \
							CUM(intermediate_reg2, w_##IDX, i##Y##I2); \
							CUM(intermediate_reg3, w_##IDX, i##Y##I3); \
							CUM(intermediate_reg4, w_##IDX, i##Y##I4); \
							CUM(intermediate_reg5, w_##IDX, i##Y##I5); \
							CUM(intermediate_reg6, w_##IDX, i##Y##I6); \
							CUM(intermediate_reg7, w_##IDX, i##Y##I7);

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

						ipIndex4 += 4;
						{
#define LOAD_INPUT1Y(Y,X)						\
							float i##Y##X = i##Y##X##_2.y;

							UNROLL10x3(LOAD_INPUT1Y);

							__global char *w = ((__global char*)w0 + (ipIndex4 * 128) * 9);
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

#define STORE_RESULT(BI)						\
					{				\
						__global float *out = packed_output + (yi*wsz + (xi0+BI))*nOutputPlanes; \
									\
						{			\
							int opIndex = lid; \
							float bv = biases[opIndex]; \
							float v = intermediate_reg##BI; \
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

					UNROLL8(STORE_RESULT);
				}
			}
		} else {
			for (int bi=0; bi<BLOCK_SIZE; bi++) {
				__local char *p00 = (__local char*)&in_block0[-nInputPlanes   + bi * (int)nInputPlanes];
				__local char *p01 = (__local char*)&in_block0[                + bi * (int)nInputPlanes];
				__local char *p02 = (__local char*)&in_block0[+nInputPlanes*1 + bi * (int)nInputPlanes];

				__local char *p10 = (__local char*)&in_block1[-nInputPlanes   + bi * (int)nInputPlanes];
				__local char *p11 = (__local char*)&in_block1[                + bi * (int)nInputPlanes];
				__local char *p12 = (__local char*)&in_block1[+nInputPlanes*1 + bi * (int)nInputPlanes];

				__local char *p20 = (__local char*)&in_block2[-nInputPlanes   + bi * (int)nInputPlanes];
				__local char *p21 = (__local char*)&in_block2[                + bi * (int)nInputPlanes];
				__local char *p22 = (__local char*)&in_block2[+nInputPlanes*1 + bi * (int)nInputPlanes];

				float intermediate_reg0 = 0;

				/*if (lid < nOutputPlanes)*/
				{
					int xi = xi0 + bi;

					if (xi == wsz) {
						break;
					}

					__global float *w0 = weight + lid;
					int nInputPlanes4 = nInputPlanes * 4;

					for (int ipIndex4 = 0; ipIndex4 < nInputPlanes4; ipIndex4+=4) {
						float i00, i01, i02;
						float i10, i11, i12;
						float i20, i21, i22;

						float2 i00_2, i01_2, i02_2;
						float2 i10_2, i11_2, i12_2;
						float2 i20_2, i21_2, i22_2;

						i00_2 = *(__local float2*)(p00 + ipIndex4);
						i10_2 = *(__local float2*)(p10 + ipIndex4);
						i20_2 = *(__local float2*)(p20 + ipIndex4);

						i01_2 = *(__local float2*)(p01 + ipIndex4);
						i11_2 = *(__local float2*)(p11 + ipIndex4);
						i21_2 = *(__local float2*)(p21 + ipIndex4);

						i02_2 = *(__local float2*)(p02 + ipIndex4);
						i12_2 = *(__local float2*)(p12 + ipIndex4);
						i22_2 = *(__local float2*)(p22 + ipIndex4);

						{
							i00 = i00_2.x;
							i10 = i10_2.x;
							i20 = i20_2.x;

							i01 = i01_2.x;
							i11 = i11_2.x;
							i21 = i21_2.x;

							i02 = i02_2.x;
							i12 = i12_2.x;
							i22 = i22_2.x;

							__global char *w = ((__global char*)w0 + (ipIndex4 * 128) * 9);

							{
								CUM(intermediate_reg0, *(__global float*)(w + 0 * vec_width4), i00);
								CUM(intermediate_reg0, *(__global float*)(w + 1 * vec_width4), i01);
								CUM(intermediate_reg0, *(__global float*)(w + 2 * vec_width4), i02);

								CUM(intermediate_reg0, *(__global float*)(w + 3 * vec_width4), i10);
								CUM(intermediate_reg0, *(__global float*)(w + 4 * vec_width4), i11);
								CUM(intermediate_reg0, *(__global float*)(w + 5 * vec_width4), i12);

								CUM(intermediate_reg0, *(__global float*)(w + 6 * vec_width4), i20);
								CUM(intermediate_reg0, *(__global float*)(w + 7 * vec_width4), i21);
								CUM(intermediate_reg0, *(__global float*)(w + 8 * vec_width4), i22);
							}
						}

						ipIndex4 += 4;
						{
							i00 = i00_2.y;
							i10 = i10_2.y;
							i20 = i20_2.y;

							i01 = i01_2.y;
							i11 = i11_2.y;
							i21 = i21_2.y;

							i02 = i02_2.y;
							i12 = i12_2.y;
							i22 = i22_2.y;

							__global char *w = ((__global char*)w0 + (ipIndex4 * 128) * 9);

							{
								intermediate_reg0 += *(__global float*)(w + 0 * vec_width4) * i00;
								intermediate_reg0 += *(__global float*)(w + 1 * vec_width4) * i01;
								intermediate_reg0 += *(__global float*)(w + 2 * vec_width4) * i02;

								intermediate_reg0 += *(__global float*)(w + 3 * vec_width4) * i10;
								intermediate_reg0 += *(__global float*)(w + 4 * vec_width4) * i11;
								intermediate_reg0 += *(__global float*)(w + 5 * vec_width4) * i12;

								intermediate_reg0 += *(__global float*)(w + 6 * vec_width4) * i20;
								intermediate_reg0 += *(__global float*)(w + 7 * vec_width4) * i21;
								intermediate_reg0 += *(__global float*)(w + 8 * vec_width4) * i22;
							}
						}

					}

					__global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;

					{
						int opIndex = lid;
						float bv = biases[opIndex];
						float v = intermediate_reg0;
						v += bv;

						float mtz = max(v, 0.0f);
						float ltz = min(v, 0.0f);

						v = ltz * 0.1f + mtz;

						out[opIndex] = v;
					}
				}
			}
		}
	}
}
