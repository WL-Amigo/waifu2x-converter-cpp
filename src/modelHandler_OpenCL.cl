/* -*- mode: c -*- */

#define VEC_WIDTH 128
#define BLOCK_SIZE 8

__kernel void
filter(__global const float * __restrict__ packed_input,
       int nInputPlanes,
       __global float * __restrict__ packed_output,
       int nOutputPlanes,
       __global float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       __global float * __restrict__ weight,
       __local float * __restrict__ local_mem)
{
	unsigned int yi = get_group_id(0);
	unsigned int lid = get_local_id(0);

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

	unsigned int vec_width = min((int)VEC_WIDTH, (int)nOutputPlanes);

	for (int xi0=0; xi0<wsz; xi0+=BLOCK_SIZE) {
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < nInputPlanes) {
			int bi;

			for (bi=0; bi<BLOCK_SIZE; bi++) {
				int xi = xi0 + bi;

				if (xi == wsz - 1) {
					break;
				}

				in_block0[bi*nInputPlanes + lid] = in01[xi*nInputPlanes + lid];
				in_block1[bi*nInputPlanes + lid] = in11[xi*nInputPlanes + lid];
				in_block2[bi*nInputPlanes + lid] = in21[xi*nInputPlanes + lid];
			}

			{
				int xi = xi0 + bi;
				if (xi == wsz - 1) {
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
				if (xi < 0) {
					in_block0[-1*(int)nInputPlanes + (int)lid] = in01[lid];
					in_block1[-1*(int)nInputPlanes + (int)lid] = in01[lid];
					in_block2[-1*(int)nInputPlanes + (int)lid] = in01[lid];
				} else {
					in_block0[-1*(int)nInputPlanes + (int)lid] = in01[xi*(int)nInputPlanes + lid];
					in_block1[-1*(int)nInputPlanes + (int)lid] = in11[xi*(int)nInputPlanes + lid];
					in_block2[-1*(int)nInputPlanes + (int)lid] = in21[xi*(int)nInputPlanes + lid];
				}
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < nOutputPlanes) {
			for (int bi=0; bi<BLOCK_SIZE; bi++) {
				int xi = xi0 + bi;

				if (xi == wsz) {
					break;
				}

				float intermediate_reg = 0;

				for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
					float i00, i01, i02;
					float i10, i11, i12;
					float i20, i21, i22;

					i00 = in_block0[(bi-1) * (int)nInputPlanes + ipIndex];
					i10 = in_block1[(bi-1) * (int)nInputPlanes + ipIndex];
					i20 = in_block2[(bi-1) * (int)nInputPlanes + ipIndex];

					i01 = in_block0[bi * (int)nInputPlanes + ipIndex];
					i11 = in_block1[bi * (int)nInputPlanes + ipIndex];
					i21 = in_block2[bi * (int)nInputPlanes + ipIndex];

					i02 = in_block0[(bi+1) * (int)nInputPlanes + ipIndex];
					i12 = in_block1[(bi+1) * (int)nInputPlanes + ipIndex];
					i22 = in_block2[(bi+1) * (int)nInputPlanes + ipIndex];

					__global float *w = weight + (ipIndex * nOutputPlanes) * 9 + lid;

					if (lid < nOutputPlanes) {
						int opIndex = lid;
						float v = 0;

						v += w[0*vec_width] * i00;
						v += w[1*vec_width] * i01;
						v += w[2*vec_width] * i02;

						v += w[3*vec_width] * i10;
						v += w[4*vec_width] * i11;
						v += w[5*vec_width] * i12;

						v += w[6*vec_width] * i20;
						v += w[7*vec_width] * i21;
						v += w[8*vec_width] * i22;

						w += 9 * VEC_WIDTH;

						intermediate_reg += v;
					}
				}

				__global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;

				{
					int opIndex = lid;
					float bv = biases[opIndex];
					float v = intermediate_reg;
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

