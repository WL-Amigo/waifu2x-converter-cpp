/* -*- mode: c++ -*- */

#define BLOCK_SIZE 8

extern "C" __global__ void
filter(const float * __restrict__ packed_input,
       int nInputPlanes,
       float * __restrict__ packed_output,
       int nOutputPlanes,
       const float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       const float * __restrict__ weight)
{
	extern __shared__ float shared_buf[];

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
	if (yi == wsz-1) {
		in2p = inp;
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
			if (lid < nInputPlanes) {
				int bi;
				for (bi=0; bi<BLOCK_SIZE; bi++) {
					int xi = xi0 + bi;
					if (xi == wsz) {
						break;
					}

					/* load to shared */
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
			__syncthreads();

			if (0 && rem >= BLOCK_SIZE) {
			} else {
				for (int bi=0; bi<BLOCK_SIZE; bi++) {
					int xi = xi0+bi;
					if (xi == wsz) {
						break;
					}

#if 1
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
/*
						sum += i00 * weight[(9*ip+0) * 128 + op];
						sum += i01 * weight[(9*ip+1) * 128 + op];
						sum += i02 * weight[(9*ip+2) * 128 + op];

						sum += i10 * weight[(9*ip+3) * 128 + op];
						sum += i11 * weight[(9*ip+4) * 128 + op];
						sum += i12 * weight[(9*ip+5) * 128 + op];

						sum += i20 * weight[(9*ip+6) * 128 + op];
						sum += i21 * weight[(9*ip+7) * 128 + op];
						sum += i22 * weight[(9*ip+8) * 128 + op];
*/
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
#else
					float sum = 0;
					for (unsigned int ip=0; ip<nInputPlanes; ip++) {
						float i00, i01, i02;
						float i10, i11, i12;
						float i20, i21, i22;

						i01 = in01[xi*nInputPlanes+ip];
						i11 = in11[xi*nInputPlanes+ip];
						i21 = in21[xi*nInputPlanes+ip];

						if (xi == 0) {
							i00 = i01;
							i10 = i11;
							i20 = i21;
						} else {
							i00 = in01[(xi-1)*nInputPlanes+ip];
							i10 = in11[(xi-1)*nInputPlanes+ip];
							i20 = in21[(xi-1)*nInputPlanes+ip];
						}

						if (xi == wsz-1) {
							i02 = i01;
							i12 = i11;
							i22 = i21;
						} else {
							i02 = in01[(xi+1)*nInputPlanes+ip];
							i12 = in11[(xi+1)*nInputPlanes+ip];
							i22 = in21[(xi+1)*nInputPlanes+ip];
						}

						sum += i00 * weight[(9*ip+0) * 128 + op];
						sum += i01 * weight[(9*ip+1) * 128 + op];
						sum += i02 * weight[(9*ip+2) * 128 + op];

						sum += i10 * weight[(9*ip+3) * 128 + op];
						sum += i11 * weight[(9*ip+4) * 128 + op];
						sum += i12 * weight[(9*ip+5) * 128 + op];

						sum += i20 * weight[(9*ip+6) * 128 + op];
						sum += i21 * weight[(9*ip+7) * 128 + op];
						sum += i22 * weight[(9*ip+8) * 128 + op];
					}
#endif

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


/*
			float v = sum;
			float bv = biases[op];
			v += bv;
			float mtz = max(v,0.0f);
			float ltz = min(v,0.0f);

			v = ltz * 0.1f + mtz;

			float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;

			out[op] = v;

*/