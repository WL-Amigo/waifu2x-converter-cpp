#ifndef W2XC_ENV_HPP
#define W2XC_ENV_HPP

#include "w2xconv.h"

struct OpenCLDev;
struct CUDADev;

namespace w2xc {
struct ThreadPool;
}

struct ComputeEnv {
    int num_cl_dev;
    int num_cuda_dev;
    OpenCLDev *cl_dev_list;
    CUDADev *cuda_dev_list;
    double transfer_wait;

    unsigned int pref_block_size;

#if defined(_WIN32) || defined(__linux)
    w2xc::ThreadPool *tpool;
#endif
    ComputeEnv();
};

extern void clearError(W2XConv *conv);

#endif
