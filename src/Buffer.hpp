/*
* The MIT License (MIT)
* This file is part of waifu2x-converter-cpp
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

#ifndef BUFFER_HPP
#define BUFFER_HPP

#ifdef _WIN32
#include <malloc.h>

#define w2xc_aligned_malloc _aligned_malloc
#define w2xc_aligned_free _aligned_free

#elif defined __ANDROID__
#include <stdlib.h>
static inline void * w2xc_aligned_malloc(size_t size, size_t alignment)
{
    return memalign(alignment, size);
}
#define w2xc_aligned_free free


#else
#include <stdlib.h>
static inline void * w2xc_aligned_malloc(size_t size, size_t alignment)
{
    void *ret;
    int r = posix_memalign(&ret, alignment, size);
    if (r != 0)
	{
        return NULL;
    }
    return ret;
}

#define w2xc_aligned_free free

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
    enum type
	{
        OpenCL,
        CUDA,
        HOST,
        EMPTY
    } type;
    int devid;

    Processor(enum type tp, int devid) :type(tp), devid(devid) {}
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
