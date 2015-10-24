#include <stdlib.h>
#include "w2xconv.h"

int
main(int argc, char **argv)
{
    int block_size = 512;
    int proc = 0;

    if (argc >= 2) {
        block_size = atoi(argv[1]);
    }

    if (argc >= 3) {
        proc = atoi(argv[2]);
    }

    struct W2XConv *c = w2xconv_init_with_processor(proc, 0, 1);

    w2xconv_load_models(c, "models");

    float *dst = malloc(block_size * block_size * sizeof(float));
    float *src = malloc(block_size * block_size * sizeof(float));

    w2xconv_apply_filter_y(c,
                           W2XCONV_FILTER_DENOISE1,
                           (unsigned char*)dst, block_size * sizeof(float),
                           (unsigned char*)src, block_size * sizeof(float),
                           block_size, block_size,
                           block_size * 2);
    w2xconv_fini(c);

    return 0;
}
