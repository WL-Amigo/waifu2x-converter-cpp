/* -*- mode: c -*- */

float
get_data(__global const float *p, int hsz, int wsz, int step, int yi, int xi, int num_plane, int plane)
{
    yi = min(hsz-1, yi);
    yi = max(0, yi);

    xi = min(wsz-1, xi);
    xi = max(0, xi);

    __global char *p1 = (__global char*)p;
    return ((__global float*)(p1 + yi*step))[xi*num_plane + plane];
}

__kernel void
filter(__global const float * __restrict__ packed_input,
       int nInputPlanes,
       __global float * __restrict__ packed_output,
       int nOutputPlanes,
       __global float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       __global float * __restrict__ weight,
       __local float * __restrict__ intermediate)
{
    unsigned int yi = get_group_id(1);
    unsigned int xi = get_group_id(0);

    unsigned int lid = get_local_id(0);

    __global const float * __restrict__ in = packed_input;
    size_t in_step = wsz * sizeof(float) * nInputPlanes;

    for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
        float i00 = get_data(in, hsz, wsz, in_step, yi-1, xi-1, nInputPlanes, ipIndex);
        float i01 = get_data(in, hsz, wsz, in_step, yi-1, xi  , nInputPlanes, ipIndex);
        float i02 = get_data(in, hsz, wsz, in_step, yi-1, xi+1, nInputPlanes, ipIndex);

        float i10 = get_data(in, hsz, wsz, in_step, yi  , xi-1, nInputPlanes, ipIndex);
        float i11 = get_data(in, hsz, wsz, in_step, yi  , xi  , nInputPlanes, ipIndex);
        float i12 = get_data(in, hsz, wsz, in_step, yi  , xi+1, nInputPlanes, ipIndex);

        float i20 = get_data(in, hsz, wsz, in_step, yi+1, xi-1, nInputPlanes, ipIndex);
        float i21 = get_data(in, hsz, wsz, in_step, yi+1, xi  , nInputPlanes, ipIndex);
        float i22 = get_data(in, hsz, wsz, in_step, yi+1, xi+1, nInputPlanes, ipIndex);

        __global float *w = weight + (ipIndex * nOutputPlanes) * 9 + lid;

        for (unsigned int opIndex_base = 0;
             opIndex_base < (unsigned int)nOutputPlanes;
             opIndex_base += VEC_WIDTH)
        {
            int opIndex = opIndex_base + lid;

            int oi_0 = lid;
            int oi_1 = opIndex_base;

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

            w += 9 * VEC_WIDTH;

            if (ipIndex == 0) {
                intermediate[opIndex] = v;
            } else {
                intermediate[opIndex] += v;
            }
        }
    }

    __global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
    if (lid == 1) {
        out[1] = 100;
    }

    for (unsigned int opIndex_base = 0;
         opIndex_base < nOutputPlanes;
         opIndex_base+=VEC_WIDTH)
    {
        unsigned int opIndex = opIndex_base + lid;
        float bv = biases[opIndex];
        float v = intermediate[opIndex];
        v += bv;

        float mtz = max(v, 0.0f);
        float ltz = min(v, 0.0f);

        v = ltz * 0.1f + mtz;

        out[opIndex] = v;
    }
}

