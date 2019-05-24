/*
* The MIT License (MIT)
* Copyright (c) 2015 amigo(white luckers), tanakamura, DeadSix27, YukihoAA and contributors
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

const unsigned char *w_cur = w_chunk_base;
unsigned char *output_base0 = output + ((x0+0)*nOutputPlanes + oi0*OP_BLOCK_SIZE)*sizeof(float);
unsigned char *output_base1 = output + ((x0+1)*nOutputPlanes + oi0*OP_BLOCK_SIZE)*sizeof(float);
unsigned char *output_base2 = output + ((x0+2)*nOutputPlanes + oi0*OP_BLOCK_SIZE)*sizeof(float);
unsigned char *output_base3 = output + ((x0+3)*nOutputPlanes + oi0*OP_BLOCK_SIZE)*sizeof(float);
unsigned char *output_base4 = output + ((x0+4)*nOutputPlanes + oi0*OP_BLOCK_SIZE)*sizeof(float);

vreg_t oreg00 = zero_vreg();
vreg_t oreg01 = zero_vreg();

vreg_t oreg10 = zero_vreg();
vreg_t oreg11 = zero_vreg();

vreg_t oreg20 = zero_vreg();
vreg_t oreg21 = zero_vreg();

vreg_t oreg30 = zero_vreg();
vreg_t oreg31 = zero_vreg();

vreg_t oreg40 = zero_vreg();
vreg_t oreg41 = zero_vreg();

for (int dposx=0; dposx<3; dposx++)
{
    int dposx2 = dposx-1;
    int dposx2_le = dposx-1;
    int dposx2_re = dposx-1;

    if (x0 == 0 && dposx == 0)
	{
        dposx2_le = 0;
    }

    if (x0 == width-5 && dposx == 2)
	{
        dposx2_re = 0;
    }

    int off0 = ((dposy2*width + x0 + 0 + dposx2_le)*nInputPlanes+ii0*IP_BLOCK_SIZE)*sizeof(float);
    int off1 = ((dposy2*width + x0 + 1 + dposx2)*nInputPlanes+ii0*IP_BLOCK_SIZE)*sizeof(float);
    int off2 = ((dposy2*width + x0 + 2 + dposx2)*nInputPlanes+ii0*IP_BLOCK_SIZE)*sizeof(float);
    int off3 = ((dposy2*width + x0 + 3 + dposx2)*nInputPlanes+ii0*IP_BLOCK_SIZE)*sizeof(float);
    int off4 = ((dposy2*width + x0 + 4 + dposx2_re)*nInputPlanes+ii0*IP_BLOCK_SIZE)*sizeof(float);
    const unsigned char *input_cur_x0 = in + off0;
    const unsigned char *input_cur_x1 = in + off1;
    const unsigned char *input_cur_x2 = in + off2;
    const unsigned char *input_cur_x3 = in + off3;
    const unsigned char *input_cur_x4 = in + off4;

#define accumulate(o0,o1,w0,w1,addr) {            \
        vreg_t ireg0 = load_vreg_broadcast(addr); \
        o0 = madd_vreg(w0, ireg0, o0);            \
        o1 = madd_vreg(w1, ireg0, o1);            \
    }

#define MUL_W_IN(I)                                               \
    {                                                             \
        int I2 = I;                                               \
        vreg_t wreg0, wreg1;                                      \
                                                                  \
        wreg0 = load_vreg(w_cur);                                 \
        wreg1 = load_vreg(w_cur + VEC_NELEM*sizeof(float));       \
                                                                  \
        accumulate(oreg00, oreg01, wreg0, wreg1, (input_cur_x0)); \
        accumulate(oreg10, oreg11, wreg0, wreg1, (input_cur_x1)); \
        accumulate(oreg20, oreg21, wreg0, wreg1, (input_cur_x2)); \
        accumulate(oreg30, oreg31, wreg0, wreg1, (input_cur_x3)); \
        accumulate(oreg40, oreg41, wreg0, wreg1, (input_cur_x4)); \
                                                                  \
        w_cur += OP_BLOCK_SIZE * sizeof(float);                   \
        input_cur_x0 += sizeof(float);                            \
        input_cur_x1 += sizeof(float);                            \
        input_cur_x2 += sizeof(float);                            \
        input_cur_x3 += sizeof(float);                            \
        input_cur_x4 += sizeof(float);                            \
    }

    for (int ii1=0; ii1<IP_BLOCK_SIZE; ii1+=4)
	{
        MUL_W_IN(ii1+0);
        MUL_W_IN(ii1+1);
        MUL_W_IN(ii1+2);
        MUL_W_IN(ii1+3);
    }
}

if (dposy == 0 && ii0 == 0)
{
    store_vreg(output_base0 + (        0)*sizeof(float), oreg00);
    store_vreg(output_base0 + (VEC_NELEM)*sizeof(float), oreg01);

    store_vreg(output_base1 + (        0)*sizeof(float), oreg10);
    store_vreg(output_base1 + (VEC_NELEM)*sizeof(float), oreg11);

    store_vreg(output_base2 + (        0)*sizeof(float), oreg20);
    store_vreg(output_base2 + (VEC_NELEM)*sizeof(float), oreg21);

    store_vreg(output_base3 + (        0)*sizeof(float), oreg30);
    store_vreg(output_base3 + (VEC_NELEM)*sizeof(float), oreg31);

    store_vreg(output_base4 + (        0)*sizeof(float), oreg40);
    store_vreg(output_base4 + (VEC_NELEM)*sizeof(float), oreg41);
}
else if (last_iter)
{
    vreg_t tmp00, tmp01;
    vreg_t tmp10, tmp11;
    vreg_t tmp20, tmp21;
    vreg_t tmp30, tmp31;
    vreg_t tmp40, tmp41;

    vreg_t mtz, ltz, bv0, bv1;

    bv0 = load_vreg(biases + (oi0*OP_BLOCK_SIZE+        0)*sizeof(float));
    bv1 = load_vreg(biases + (oi0*OP_BLOCK_SIZE+VEC_NELEM)*sizeof(float));

#define ReLU(addr, bv, N)                          \
    tmp##N = load_vreg(addr);                      \
    tmp##N = add_vreg(tmp##N, oreg##N);            \
    tmp##N = add_vreg(tmp##N, bv);                 \
    mtz = max_vreg(tmp##N, zero_vreg());           \
    ltz = min_vreg(tmp##N, zero_vreg());           \
    tmp##N = madd_vreg(ltz, set1_vreg(0.1f), mtz); \

    ReLU(output_base0 + (        0)*sizeof(float), bv0, 00);
    ReLU(output_base0 + (VEC_NELEM)*sizeof(float), bv1, 01);
    ReLU(output_base1 + (        0)*sizeof(float), bv0, 10);
    ReLU(output_base1 + (VEC_NELEM)*sizeof(float), bv1, 11);
    ReLU(output_base2 + (        0)*sizeof(float), bv0, 20);
    ReLU(output_base2 + (VEC_NELEM)*sizeof(float), bv1, 21);
    ReLU(output_base3 + (        0)*sizeof(float), bv0, 30);
    ReLU(output_base3 + (VEC_NELEM)*sizeof(float), bv1, 31);
    ReLU(output_base4 + (        0)*sizeof(float), bv0, 40);
    ReLU(output_base4 + (VEC_NELEM)*sizeof(float), bv1, 41);

    store_vreg(output_base0 + (        0)*sizeof(float), tmp00);
    store_vreg(output_base0 + (VEC_NELEM)*sizeof(float), tmp01);
    store_vreg(output_base1 + (        0)*sizeof(float), tmp10);
    store_vreg(output_base1 + (VEC_NELEM)*sizeof(float), tmp11);
    store_vreg(output_base2 + (        0)*sizeof(float), tmp20);
    store_vreg(output_base2 + (VEC_NELEM)*sizeof(float), tmp21);
    store_vreg(output_base3 + (        0)*sizeof(float), tmp30);
    store_vreg(output_base3 + (VEC_NELEM)*sizeof(float), tmp31);
    store_vreg(output_base4 + (        0)*sizeof(float), tmp40);
    store_vreg(output_base4 + (VEC_NELEM)*sizeof(float), tmp41);
}
else
{
    vreg_t tmp;
#define ADD_TO_MEM(addr, val) \
    tmp = load_vreg(addr);    \
    tmp = add_vreg(tmp, val); \
    store_vreg(addr, tmp);

    ADD_TO_MEM(output_base0 + (        0)*sizeof(float), oreg00);
    ADD_TO_MEM(output_base0 + (VEC_NELEM)*sizeof(float), oreg01);
    ADD_TO_MEM(output_base1 + (        0)*sizeof(float), oreg10);
    ADD_TO_MEM(output_base1 + (VEC_NELEM)*sizeof(float), oreg11);
    ADD_TO_MEM(output_base2 + (        0)*sizeof(float), oreg20);
    ADD_TO_MEM(output_base2 + (VEC_NELEM)*sizeof(float), oreg21);
    ADD_TO_MEM(output_base3 + (        0)*sizeof(float), oreg30);
    ADD_TO_MEM(output_base3 + (VEC_NELEM)*sizeof(float), oreg31);
    ADD_TO_MEM(output_base4 + (        0)*sizeof(float), oreg40);
    ADD_TO_MEM(output_base4 + (VEC_NELEM)*sizeof(float), oreg41);
}

#undef MUL_W_IN
#ifdef accumulate
#undef accumulate
#endif
#undef ADD_TO_MEM
#undef ReLU
