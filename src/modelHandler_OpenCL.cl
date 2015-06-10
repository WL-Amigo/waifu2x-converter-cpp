/* -*- mode: c -*- */

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

	unsigned int vec_width = min((int)128, (int)nOutputPlanes);

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

		for (int bi=0; bi<BLOCK_SIZE; bi++) {
			__local char *p00 = (__local char*)&in_block0[-nInputPlanes + bi * (int)nInputPlanes];
			__local char *p01 = (__local char*)&in_block0[              + bi * (int)nInputPlanes];
			__local char *p02 = (__local char*)&in_block0[+nInputPlanes + bi * (int)nInputPlanes];

			__local char *p10 = (__local char*)&in_block1[-nInputPlanes + bi * (int)nInputPlanes];
			__local char *p11 = (__local char*)&in_block1[              + bi * (int)nInputPlanes];
			__local char *p12 = (__local char*)&in_block1[+nInputPlanes + bi * (int)nInputPlanes];

			__local char *p20 = (__local char*)&in_block2[-nInputPlanes + bi * (int)nInputPlanes];
			__local char *p21 = (__local char*)&in_block2[              + bi * (int)nInputPlanes];
			__local char *p22 = (__local char*)&in_block2[+nInputPlanes + bi * (int)nInputPlanes];

			if (lid < nOutputPlanes) {
				int xi = xi0 + bi;

				if (xi == wsz) {
					break;
				}

				float intermediate_reg = 0;

				__global float *w0 = weight + lid;

				for (int ipIndex4 = 0; ipIndex4 < nInputPlanes*4; ipIndex4+=4) {
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
							int opIndex = lid;
							float v = 0;
							int vec_width4 = 128 * 4;

							v += *(__global float*)(w + 0 * vec_width4) * i00;
							v += *(__global float*)(w + 1 * vec_width4) * i01;
							v += *(__global float*)(w + 2 * vec_width4) * i02;

							v += *(__global float*)(w + 3 * vec_width4) * i10;
							v += *(__global float*)(w + 4 * vec_width4) * i11;
							v += *(__global float*)(w + 5 * vec_width4) * i12;

							v += *(__global float*)(w + 6 * vec_width4) * i20;
							v += *(__global float*)(w + 7 * vec_width4) * i21;
							v += *(__global float*)(w + 8 * vec_width4) * i22;

							intermediate_reg += v;
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
							int opIndex = lid;
							float v = 0;
							int vec_width4 = 128 * 4;

							v += *(__global float*)(w + 0 * vec_width4) * i00;
							v += *(__global float*)(w + 1 * vec_width4) * i01;
							v += *(__global float*)(w + 2 * vec_width4) * i02;

							v += *(__global float*)(w + 3 * vec_width4) * i10;
							v += *(__global float*)(w + 4 * vec_width4) * i11;
							v += *(__global float*)(w + 5 * vec_width4) * i12;

							v += *(__global float*)(w + 6 * vec_width4) * i20;
							v += *(__global float*)(w + 7 * vec_width4) * i21;
							v += *(__global float*)(w + 8 * vec_width4) * i22;

							intermediate_reg += v;
						}
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

