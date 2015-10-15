#ifndef BUFFER_HPP
#define BUFFER_HPP

#if defined(_MSC_VER)
#include <malloc.h>
#elif defined(X86OPT)
#include <mm_malloc.h>
#else
#include <malloc.h>
#define _mm_malloc(size,align) memalign(align,size)
#define _mm_free(ptr) free(ptr)
#endif

#include <stdlib.h>
#include <string>
#include "CLlib.h"
#include "CUDAlib.h"
#include "threadPool.hpp"
#include "sec.hpp"
#include "Env.hpp"
#include "w2xconv.h"

struct OpenCLDev {
    std::string name;

    cl_platform_id platform;
    cl_context context;
    cl_device_id devid;
    cl_command_queue queue;
    cl_kernel ker_filter;
    cl_kernel ker_filter_in1_out32;
    cl_kernel ker_filter_in128_out1;
    cl_kernel ker_filter_in3_out32;
    cl_kernel ker_filter_in128_out3;
    cl_program program;
};

struct CUDADev {
    std::string name;

    int dev;
    CUcontext context;
    CUmodule mod;
    CUfunction filter_i1_o32;
    CUfunction filter_i32;
    CUfunction filter_i64;
    CUfunction filter_i128;
    CUfunction filter_i64_o64;
    CUfunction filter_i64_o128;
    CUfunction filter_i128_o128;
    CUfunction filter_i128_o1;
    CUfunction filter_i3_o32;
    CUfunction filter_i128_o3;

    CUstream stream;
};

struct Processor {
    enum type {
        OpenCL,
        CUDA,
        HOST,
        EMPTY
    } type;
    int devid;

    Processor(enum type tp, int devid)
        :type(tp), devid(devid)
    {}
};

struct Buffer {
    ComputeEnv *env;
    size_t byte_size;

    void *host_ptr;
    cl_mem *cl_ptr_list;
    CUdeviceptr *cuda_ptr_list;

    bool host_valid;
    bool *cl_valid_list;
    bool *cuda_valid_list;

    Processor last_write;

    Buffer(ComputeEnv *env, size_t byte_size);

    Buffer(Buffer const &rhs) = delete;
    Buffer &operator = (Buffer const &rhs) = delete;
    Buffer &operator = (Buffer &&rhs) = delete;

    ~Buffer();
    void clear(ComputeEnv *env);
    void release(ComputeEnv *env);
    void invalidate(ComputeEnv *env);
    cl_mem get_read_ptr_cl(ComputeEnv *env,int devid, size_t read_byte_size);
    cl_mem get_write_ptr_cl(ComputeEnv *env,int devid);
    CUdeviceptr get_read_ptr_cuda(ComputeEnv *env,int devid, size_t read_byte_size);
    CUdeviceptr get_write_ptr_cuda(ComputeEnv *env,int devid);
    void *get_write_ptr_host(ComputeEnv *env);
    void *get_read_ptr_host(ComputeEnv *env, size_t read_byte_size);
    bool prealloc(W2XConv *conv, ComputeEnv *env);
};

#endif
