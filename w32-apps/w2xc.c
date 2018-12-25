#include <string.h>
#include <stdlib.h>
#include "w2xconv.h"

int
main(int argc, char **argv)
{
    char *dst_path;
    size_t path_len;
    struct W2XConv *c;
    int r;
    char *src_path;

    if (argc < 2) {
        puts("usage : w2xc <in.png>");
        return 1;
    }

    src_path = argv[1];

    path_len = strlen(src_path);
    dst_path = malloc(path_len + 5);
    dst_path[0] = 'm';
    dst_path[1] = 'a';
    dst_path[2] = 'i';
    dst_path[3] = '_';
    strcpy(dst_path+4, argv[1]);

    c = w2xconv_init(W2XCONV_GPU_AUTO, 0, 0);
    r = w2xconv_load_models(c, "models\\rgb");
    if (r < 0) {
        goto error;
    }

    r = w2xconv_convert_file(c, dst_path, src_path, 1, 2.0, 512);
    if (r < 0) {
        goto error;
    }

    w2xconv_fini(c);

    return 0;

error:
    {
        char *err = w2xconv_strerror(&c->last_error);
        puts(err);
        w2xconv_free(err);
    }

    w2xconv_fini(c);

    return 1;
}
