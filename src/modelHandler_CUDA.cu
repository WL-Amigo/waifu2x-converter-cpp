/* -*- mode: c++ -*- */

extern "C" __global__ void
filter(const float * __restrict__ packed_input,
       int nInputPlanes,
       float * __restrict__ packed_output,
       int nOutputPlanes,
       float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       float * __restrict__ weight)
{
	unsigned int yi = blockIdx.x;

	size_t in_step = wsz * nInputPlanes;
	const float *inp = packed_input;
	inp += wsz * nInputPlanes;

	const float *in0p = inp - in_step;
	if (yi == 0) {
		in0p = inp;
	}

	const float *in1p = inp;

	const float *in2p = inp + in_step;
	if (yi == 0) {
		in2p = inp;
	}

	const float *in01 = in0p;
	const float *in11 = in1p;
	const float *in21 = in2p;

	for (int xi=0; xi<wsz; xi++) {
		for (unsigned int op=0; op<nOutputPlanes; op++) {
			float sum=0;

			for (unsigned int ip=0; ip<nInputPlanes; ip++) {
				float i00, i01, i02;
				float i10, i11, i12;
				float i20, i21, i22;

				i01 = in01[xi*nInputPlanes];
				i11 = in11[xi*nInputPlanes];
				i21 = in21[xi*nInputPlanes];

				if (xi == 0) {
					i00 = i01;
					i10 = i11;
					i20 = i21;
				} else {
					i00 = in01[(xi-1)*nInputPlanes];
					i10 = in11[(xi-1)*nInputPlanes];
					i20 = in21[(xi-1)*nInputPlanes];
				}

				if (xi == wsz) {
					i02 = i01;
					i12 = i11;
					i22 = i21;
				} else {
					i02 = in01[(xi+1)*nInputPlanes];
					i12 = in11[(xi+1)*nInputPlanes];
					i22 = in21[(xi+1)*nInputPlanes];
				}

				sum += i00 * weight[(9*nInputPlanes+0) * 128 + op];
				sum += i01 * weight[(9*nInputPlanes+1) * 128 + op];
				sum += i02 * weight[(9*nInputPlanes+2) * 128 + op];

				sum += i10 * weight[(9*nInputPlanes+3) * 128 + op];
				sum += i11 * weight[(9*nInputPlanes+4) * 128 + op];
				sum += i12 * weight[(9*nInputPlanes+5) * 128 + op];

				sum += i20 * weight[(9*nInputPlanes+6) * 128 + op];
				sum += i21 * weight[(9*nInputPlanes+7) * 128 + op];
				sum += i22 * weight[(9*nInputPlanes+8) * 128 + op];
			}

			float v = sum;
			float bv = biases[op];
			v += bv;
			float mtz = max(v,0.0f);
			float ltz = min(v,0.0f);

			v = ltz * 0.1f + mtz;

			float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
			out[op] = v;
		}
	}
}
