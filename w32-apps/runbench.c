#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "w2xconv.h"

int
main(int argc, char **argv)
{
    int block_size = 512;
    int proc = 0;
    size_t num_proc;
    int num_thread = 0;
    const struct W2XConvProcessor *proc_list;
    
    proc_list = w2xconv_get_processor_list(&num_proc);

    if (argc>=2 && strcmp(argv[1],"-l") == 0) {
        size_t i;
        for (i=0; i<num_proc; i++) {
            printf("type=%2d, subtype=%2d, name=%s\n",
                   proc_list[i].type,
                   proc_list[i].sub_type,
                   proc_list[i].dev_name);
        }
        exit(0);
    }
    if (argc >= 2) {
        block_size = atoi(argv[1]);
    }

    if (argc >= 3) {
        proc = atoi(argv[2]);
    }

    if (argc >= 4) {
        num_thread = atoi(argv[3]);
    }

    struct W2XConv *c = w2xconv_init_with_processor(proc, num_thread, 1);
    puts(proc_list[proc].dev_name);

    int num_maps[7] = {
        32,
        32,
        64,
        64,
        128,
        128,
        1
    };

    int total = 0;
    int yi, xi, i;

    total += num_maps[0];
    for (i=1; i<7; i++) {
        total += num_maps[i-1] * num_maps[i];
    }

    float *bias = calloc(total, sizeof(float));
    float *coef = calloc(total * 3 * 3, sizeof(float));
    float *dst = calloc(block_size * block_size, sizeof(float));
    float *src = calloc(block_size * block_size, sizeof(float));

    for (yi=0; yi<block_size; yi++) {
        for (xi=0; xi<block_size; xi++) {
            src[yi*block_size + xi] = rand() / (float)RAND_MAX;
        }
    }
    for (i=0; i< (total * 3 * 3); i++) {
        coef[i] = (rand() / (float)RAND_MAX);
    }

    w2xconv_set_model_3x3(c,
                          W2XCONV_FILTER_SCALE2x,
                          7,
                          1,
                          num_maps,
                          coef,
                          bias);

    w2xconv_apply_filter_y(c,
                           W2XCONV_FILTER_SCALE2x,
                           (unsigned char*)dst, block_size * sizeof(float),
                           (unsigned char*)src, block_size * sizeof(float),
                           block_size, block_size,
                           block_size * 2);
    w2xconv_fini(c);

    return 0;
}
