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

static __device__ float
warp_sum(float v) {
    v += __shfl_down(v, 1);
    v += __shfl_down(v, 2);
    v += __shfl_down(v, 4);
    v += __shfl_down(v, 8);
    v += __shfl_down(v, 16);

    return v;
}

extern "C" __global__ void
filter_i128_o128(const float * __restrict__ packed_input,
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

	int nInputPlanes = 128;
	int nOutputPlanes = 128;

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
			for (int xi0=0; xi0<wsz; xi0+=BLOCK_SIZE) {
				__syncthreads();
				if (lid < INPUT_BLOCK_SIZE) {
					int bi;
					for (bi=0; bi<X_BLOCK_SIZE; bi++) {
						int xi = xi0 + bi;
						if (xi == wsz) {
							break;
						}

						/* load to shared */
						in_block0[bi*INPUT_BLOCK_SIZE + lid] = in01[xi*nInputPlanes + ib0+lid];
						in_block1[bi*INPUT_BLOCK_SIZE + lid] = in11[xi*nInputPlanes + ib0+lid];
						in_block2[bi*INPUT_BLOCK_SIZE + lid] = in21[xi*nInputPlanes + ib0+lid];
					}

					{
						int xi = xi0 + bi;
						if (xi == wsz) {
							in_block0[bi*(int)INPUT_BLOCK_SIZE + lid] = in01[(xi-1)*(int)nInputPlanes + ib0+lid];
							in_block1[bi*(int)INPUT_BLOCK_SIZE + lid] = in11[(xi-1)*(int)nInputPlanes + ib0+lid];
							in_block2[bi*(int)INPUT_BLOCK_SIZE + lid] = in21[(xi-1)*(int)nInputPlanes + ib0+lid];
						} else {
							in_block0[bi*(int)INPUT_BLOCK_SIZE + lid] = in01[xi*(int)nInputPlanes + ib0+lid];
							in_block1[bi*(int)INPUT_BLOCK_SIZE + lid] = in11[xi*(int)nInputPlanes + ib0+lid];
							in_block2[bi*(int)INPUT_BLOCK_SIZE + lid] = in21[xi*(int)nInputPlanes + ib0+lid];
						}
					}

					{
						int xi = xi0-1;
						if (xi == -1) {
							in_block0[-1*(int)INPUT_BLOCK_SIZE + (int)lid] = in01[ib0+lid];
							in_block1[-1*(int)INPUT_BLOCK_SIZE + (int)lid] = in11[ib0+lid];
							in_block2[-1*(int)INPUT_BLOCK_SIZE + (int)lid] = in21[ib0+lid];
						} else {
							in_block0[-1*(int)INPUT_BLOCK_SIZE + (int)lid] = in01[xi*(int)nInputPlanes + ib0+lid];
							in_block1[-1*(int)INPUT_BLOCK_SIZE + (int)lid] = in11[xi*(int)nInputPlanes + ib0+lid];
							in_block2[-1*(int)INPUT_BLOCK_SIZE + (int)lid] = in21[xi*(int)nInputPlanes + ib0+lid];
						}
					}
				}
				__syncthreads();

				int rem = wsz - xi0;
				int op = lid + ob0;

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
						float *out = packed_output + (yi*wsz + (xi0+BI))*nOutputPlanes; \
									\
						{			\
							float v = sum##BI + out[op]; \
							v += biases[op]; \
									\
							float mtz = max(v, 0.0f); \
							float ltz = min(v, 0.0f); \
									\
							v = ltz * 0.1f + mtz; \
									\
							out[op] = v;	\
						}			\
					}

					if ((ib0+INPUT_BLOCK_SIZE) == nInputPlanes) {
						UNROLL8(RELU);
					} else if (ib0 == 0) {
						float *out = packed_output + (yi*wsz + (xi0))*nOutputPlanes;

						out[op+nOutputPlanes*0] = sum0;
						out[op+nOutputPlanes*1] = sum1;
						out[op+nOutputPlanes*2] = sum2;
						out[op+nOutputPlanes*3] = sum3;
						out[op+nOutputPlanes*4] = sum4;
						out[op+nOutputPlanes*5] = sum5;
						out[op+nOutputPlanes*6] = sum6;
						out[op+nOutputPlanes*7] = sum7;
					} else {
						float *out = packed_output + (yi*wsz + (xi0))*nOutputPlanes;

						out[op+nOutputPlanes*0] += sum0;
						out[op+nOutputPlanes*1] += sum1;
						out[op+nOutputPlanes*2] += sum2;
						out[op+nOutputPlanes*3] += sum3;
						out[op+nOutputPlanes*4] += sum4;
						out[op+nOutputPlanes*5] += sum5;
						out[op+nOutputPlanes*6] += sum6;
						out[op+nOutputPlanes*7] += sum7;
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
							v += biases[op];

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

#if 0
		/*for (unsigned int op=0; op<nOutputPlanes; op++) thread */
		{


					{


					}
				}
			} else {
			}
		}
#endif
	}

}
