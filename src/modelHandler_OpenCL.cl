/* -*- mode: c -*- */

#define ITER32(F)                                             \
    F(0)  F(1)  F(2)  F(3)  F(4)  F(5)  F(6)  F(7)            \
    F(8)  F(9)  F(10) F(11) F(12) F(13) F(14) F(15)           \
    F(16) F(17) F(18) F(19) F(20) F(21) F(22) F(23)           \
    F(24) F(25) F(26) F(27) F(28) F(29) F(30) F(31)           \

#define ITER16(F)                                             \
    F(0)  F(1)  F(2)  F(3)  F(4)  F(5)  F(6)  F(7)            \
    F(8)  F(9)  F(10) F(11) F(12) F(13) F(14) F(15)           \

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
       __global float * weight,
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

    /* 4*256 = 1024 */
    __local float *intermediate = local_mem;
    local_mem += 256;

    unsigned int opPerG = nOutputPlanes / 4U;
    unsigned int opStart;

    opStart = og * opPerG;

    /* input  (16 / item) x 8
     * .... .... .... .... 128
     * .... .... .... .... 128
     * output (1 / item) x 32
     */

    unsigned int outputIdx = lid / 8U + opStart;
    unsigned int inputIdx = lid & 7;

    unsigned int inputPlaneStart = inputIdx * 16;
    unsigned int inputPlaneEnd = (inputIdx+1) * 16;

    /* 9*4*128 = 4608 */
    __local float *local_00 = local_mem; local_mem += nInputPlanes;
    __local float *local_01 = local_mem; local_mem += nInputPlanes;
    __local float *local_02 = local_mem; local_mem += nInputPlanes;

    __local float *local_10 = local_mem; local_mem += nInputPlanes;
    __local float *local_11 = local_mem; local_mem += nInputPlanes;
    __local float *local_12 = local_mem; local_mem += nInputPlanes;

    __local float *local_20 = local_mem; local_mem += nInputPlanes;
    __local float *local_21 = local_mem; local_mem += nInputPlanes;
    __local float *local_22 = local_mem; local_mem += nInputPlanes;

    if (lid < nInputPlanes) {
        __global float *in01 = in01_base;
        __global float *in11 = in11_base;
        __global float *in21 = in21_base;

        float v01 = in01[lid];
        float v11 = in11[lid];
        float v21 = in21[lid];

        local_01[lid] = local_02[lid] = v01;
        local_11[lid] = local_12[lid] = v11;
        local_21[lid] = local_22[lid] = v21;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __global float *w = weight + (inputPlaneStart * nOutputPlanes) * 9;

    /* w<ipi>_<coef> */

#define LOAD_COEF(ipi,coef)                     \
    float w##ipi##_##coef = w[(ipi)*9*nOutputPlanes + (coef) * nOutputPlanes + outputIdx];

#define LOAD_COEF0(ipi) LOAD_COEF(ipi,0)
#define LOAD_COEF1(ipi) LOAD_COEF(ipi,1)
#define LOAD_COEF2(ipi) LOAD_COEF(ipi,2)
#define LOAD_COEF3(ipi) LOAD_COEF(ipi,3)
#define LOAD_COEF4(ipi) LOAD_COEF(ipi,4)
#define LOAD_COEF5(ipi) LOAD_COEF(ipi,5)

    ITER16(LOAD_COEF0);
    ITER16(LOAD_COEF1);
    ITER16(LOAD_COEF2);
    ITER16(LOAD_COEF3);
    ITER16(LOAD_COEF4);
    ITER16(LOAD_COEF5);         /* 16x5 = 90reg */

    for (int xi=0; xi<wsz; xi++) {
        float v = 0;

        __global float *in01 = in01_base + xi * nInputPlanes;
        __global float *in11 = in11_base + xi * nInputPlanes;
        __global float *in21 = in21_base + xi * nInputPlanes;

        __local float *tmp0 = local_00;
        __local float *tmp1 = local_10;
        __local float *tmp2 = local_20;

        local_00 = local_01;
        local_01 = local_02;

        local_10 = local_11;
        local_11 = local_12;

        local_20 = local_21;
        local_21 = local_22;

        if (xi != wsz-1) {
            local_02 = tmp0;
            local_12 = tmp1;
            local_22 = tmp2;
            if (lid < nInputPlanes) {
                local_02[lid] = in01[(int)lid+(int)nInputPlanes];
                local_12[lid] = in11[(int)lid+(int)nInputPlanes];
                local_22[lid] = in21[(int)lid+(int)nInputPlanes];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#define IP1(ipi) {                                      \
            int ipIndex = ipi + inputPlaneStart;        \
            float i00, i01, i02;                        \
            float i10, i11, i12;                        \
            float i20, i21, i22;                        \
                                                        \
            i00 = local_00[ipIndex];                    \
            i01 = local_01[ipIndex];                    \
            i02 = local_02[ipIndex];                    \
            i10 = local_10[ipIndex];                    \
            i11 = local_11[ipIndex];                    \
            i12 = local_12[ipIndex];                    \
            i20 = local_20[ipIndex];                    \
            i21 = local_21[ipIndex];                    \
            i22 = local_22[ipIndex];                    \
                                                        \
            v += w##ipi##_0 * i00;                                  \
            v += w##ipi##_1 * i01;                                  \
            v += w##ipi##_2 * i02;  \
                                                                        \
            v += w##ipi##_3 * i10;  \
            v += w##ipi##_4 * i11;  \
            v += w##ipi##_5 * i12;  \
                                                                        \
            v += w[ipi*9*nOutputPlanes + 6*nOutputPlanes + outputIdx] * i20;  \
            v += w[ipi*9*nOutputPlanes + 7*nOutputPlanes + outputIdx] * i21; \
            v += w[ipi*9*nOutputPlanes + 8*nOutputPlanes + outputIdx] * i22;  \
                                                        \
        }

#if 1
        ITER16(IP1);
#else
        for (int ipi=0; ipi<16; ipi++) {
            IP1(ipi);
        }
#endif

        intermediate[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
        unsigned int opIndex = opStart + lid;
        __global float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;

        /* 256 to 64 reduction */
        if (lid < 32) {
            float sum = 0;

            sum = intermediate[lid*8 + 0]
                + intermediate[lid*8 + 1]
                + intermediate[lid*8 + 2]
                + intermediate[lid*8 + 3]
                + intermediate[lid*8 + 4]
                + intermediate[lid*8 + 5]
                + intermediate[lid*8 + 6]
                + intermediate[lid*8 + 7];

            float bv = biases[opIndex];
            float v = sum;

            v += bv;

            float mtz = max(v, 0.0f);
            float ltz = min(v, 0.0f);

            v = ltz * 0.1f + mtz;

            out[opIndex] = v;
        }
    }
}

