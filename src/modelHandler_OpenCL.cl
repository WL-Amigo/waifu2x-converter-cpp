/* -*- mode: c -*- */

float
get_data(__global const float *p, int hsz, int wsz, int step, int yi, int xi, int num_plane, int plane)
{
    xi = min(wsz-1, xi);
    xi = max(0, xi);

    return p[xi * num_plane];
}

__kernel void
filter(__global const float * __restrict__ packed_input,
       unsigned int nInputPlanes,
       __global float * __restrict__ packed_output,
       unsigned int nOutputPlanes,
       __global float * __restrict__ biases,
       unsigned int hsz,
       unsigned int wsz,
       __global float * __restrict__ weight,
       __local float * __restrict__ local_mem)
{
    unsigned int yi = get_group_id(1);
    unsigned int og = get_group_id(0);
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

    __global float *in01_base = (__global float*)in0p;
    __global float *in11_base = (__global float*)in1p;
    __global float *in21_base = (__global float*)in2p;

    __local float *intermediate = local_mem;
    local_mem += sizeof(float) * nOutputPlanes;

    unsigned int vec_width = min((int)VEC_WIDTH, (int)nOutputPlanes);
    unsigned int opHalf = nOutputPlanes / 2U;
    unsigned int opStart, opEnd;

    opStart = og * opHalf;

    /* input  (32 / item) x 4
     * .... .... .... .... 128
     * .... .... .... .... 128
     * output (1 / item) x 64
     */

    unsigned int outputIdx = lid / 4U;
    unsigned int inputIdx = lid & 3;

    unsigned int inputPlaneStart = inputIdx * 32;
    unsigned int inputPlaneEnd = (inputIdx+1) * 32;

    for (int xi=0; xi<wsz; xi++) {
        float v = 0;

        __global float *in01 = in01_base + xi * nInputPlanes;
        __global float *in11 = in11_base + xi * nInputPlanes;
        __global float *in21 = in21_base + xi * nInputPlanes;

        for (int ipIndex = inputPlaneStart; ipIndex < inputPlaneEnd; ipIndex++) {
            float i00, i01, i02;
            float i10, i11, i12;
            float i20, i21, i22;

            i01 = in01[ipIndex];
            i11 = in11[ipIndex];
            i21 = in21[ipIndex];

            if (xi == 0) {
                i00 = i01;
                i10 = i11;
                i20 = i21;
            } else {
                i00 = in01[ipIndex-(int)nInputPlanes];
                i10 = in11[ipIndex-(int)nInputPlanes];
                i20 = in21[ipIndex-(int)nInputPlanes];
            }

            if (xi == wsz-1) {
                i02 = i01;
                i12 = i11;
                i22 = i21;
            } else {
                i02 = in01[ipIndex+nInputPlanes];
                i12 = in11[ipIndex+nInputPlanes];
                i22 = in21[ipIndex+nInputPlanes];
            }

            __global float *w = weight + (ipIndex * nOutputPlanes) * 9 + outputIdx + opStart;

            v += w[0*nOutputPlanes] * i00;
            v += w[1*nOutputPlanes] * i01;
            v += w[2*nOutputPlanes] * i02;

            v += w[3*nOutputPlanes] * i10;
            v += w[4*nOutputPlanes] * i11;
            v += w[5*nOutputPlanes] * i12;

            v += w[6*nOutputPlanes] * i20;
            v += w[7*nOutputPlanes] * i21;
            v += w[8*nOutputPlanes] * i22;
        }

        intermediate[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int opIndex = opStart + lid;
        __global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;

        /* 256 to 64 reduction */
        if (lid < 64) {
            float sum = 0;

            sum = intermediate[lid*4 + 0]
                + intermediate[lid*4 + 1]
                + intermediate[lid*4 + 2]
                + intermediate[lid*4 + 3];

            float bv = biases[opIndex];
            float v = sum;

            v += bv;

            float mtz = max(v, 0.0f);
            float ltz = min(v, 0.0f);

            v = ltz * 0.1f + mtz;

            out[opIndex] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

