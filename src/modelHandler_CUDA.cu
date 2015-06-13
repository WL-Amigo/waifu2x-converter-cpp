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
}
