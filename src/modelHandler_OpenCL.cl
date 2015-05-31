/* -*- mode: c -*- */

float
get_data(const float *p, int hsz, int wsz, int step, int yi, int xi, int num_plane, int plane)
{
    yi = std::min(hsz-1, yi);
    yi = std::max(0, yi);

    xi = std::min(wsz-1, xi);
    xi = std::max(0, xi);

    char *p1 = (char*)p;
    return ((float*)(p1 + yi*step))[xi*num_plane + plane];
}

__kernel void
filter(__global const float * __restrict__ packed_input,
       int nInputPlanes,
       __global float * __restrict__ packed_output,
       int nOutputPlanes,
       __global float * __restrict__ biases,
       unsigned long hsz,
       unsigned long wsz,
       __global float * __restrict__ weight,
       __global float * __restrict__ intermediate)
{
    unsigned long yi = get_global_id(1);
    unsigned long xi = get_global_id(0);

    __global const float * __restrict__ in = packed_input;
    size_t in_step = wsz * sizeof(float) * nInputPlanes;

    for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
        float i00 = get_data(in, hsz, wsz, in_step, yi-1, xi-1, ipIndex);
        float i01 = get_data(in, hsz, wsz, in_step, yi-1, xi  , ipIndex);
        float i02 = get_data(in, hsz, wsz, in_step, yi-1, xi+1, ipIndex);

        float i10 = get_data(in, hsz, wsz, in_step, yi  , xi-1, ipIndex);
        float i11 = get_data(in, hsz, wsz, in_step, yi  , xi  , ipIndex);
        float i12 = get_data(in, hsz, wsz, in_step, yi  , xi+1, ipIndex);

        float i20 = get_data(in, hsz, wsz, in_step, yi+1, xi-1, ipIndex);
        float i21 = get_data(in, hsz, wsz, in_step, yi+1, xi  , ipIndex);
        float i22 = get_data(in, hsz, wsz, in_step, yi+1, xi+1, ipIndex);

        __global float *w_base = weight + (ipIndex * nOutputPlanes) * 9;

        for (unsigned int opIndex = 0;
             opIndex < (unsigned int)nOutputPlanes;
             opIndex ++)
        {
            int oi_0 = opIndex % VEC_WIDTH;
            int oi_1 = (opIndex / VEC_WIDTH) * VEC_WIDTH;

            __global float *w = w_base + oi_1*9 + oi_0;
            float v = 0;

            v += w[0*VEC_WIDTH] * i00;
            v += w[1*VEC_WIDTH] * i01;
            v += w[2*VEC_WIDTH] * i02;

            v += w[3*VEC_WIDTH] * i10;
            v += w[4*VEC_WIDTH] * i11;
            v += w[5*VEC_WIDTH] * i12;

            v += w[6*VEC_WIDTH] * i20;
            v += w[7*VEC_WIDTH] * i21;
            v += w[8*VEC_WIDTH] * i22;

            if (ipIndex == 0) {
                intermediate[opIndex] = v;
            } else {
                intermediate[opIndex] += v;
            }
        }
    }

    __global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
    for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++) {
        float bv = biases[opIndex];
        float v = intermediate[opIndex];
        v += bv;

        float mtz = max(v, 0.0f);
        float ltz = min(v, 0.0f);

        v = ltz * 0.1f + mtz;

        out[opIndex] = v;
    }
}

