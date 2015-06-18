#include "w2xconv.h"

int
main()
{
    struct W2XConv *c = w2xconv_init(1, 0, 1);
    w2xconv_load_models(c, "models");
    w2xconv_test(c, 512);
    w2xconv_fini(c);
}
