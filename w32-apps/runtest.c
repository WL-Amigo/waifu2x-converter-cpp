#include "w2xconv.h"

int
main(int argc, char **argv)
{
    struct W2XConv *c = w2xconv_init(1, 0, 1);
    const char *models = "models_rgb";
    if (argc >= 2) {
        models = argv[1];
    }
    w2xconv_load_models(c, models);
    w2xconv_test(c, 0);
    w2xconv_fini(c);

    return 0;
}
