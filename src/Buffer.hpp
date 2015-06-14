#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <malloc.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "CLlib.h"
#include "CUDAlib.h"

struct OpenCLDev {
    cl_platform_id platform;
    cl_context context;
    cl_device_id devid;
    cl_command_queue queue;
    cl_kernel ker_filter;
    cl_kernel ker_filter_in1_out32;
    cl_kernel ker_filter_in128_out1;
    cl_program program;
};

struct CUDADev {
    int dev;
    CUcontext context;
    CUmodule mod;
    CUfunction filter_i32;
    CUfunction filter_i64;
    CUfunction filter_i128;
    CUfunction filter_i64_o64;
    CUfunction filter_i64_o128;
    CUfunction filter_i128_o128;
    CUstream stream;
};

struct ComputeEnv {
    int num_cl_dev;
    int num_cuda_dev;
    OpenCLDev *cl_dev_list;
    CUDADev *cuda_dev_list;

    ComputeEnv()
        :num_cl_dev(0),
         num_cuda_dev(0),
         cl_dev_list(nullptr),
         cuda_dev_list(nullptr)
    {}
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

    Buffer(ComputeEnv *env, size_t byte_size)
        :env(env),
         byte_size(byte_size),
         last_write(Processor::EMPTY, 0)
    {
        int num_cl_dev = env->num_cl_dev;
        int num_cuda_dev = env->num_cuda_dev;

        cl_ptr_list = new cl_mem[num_cl_dev];
        cl_valid_list = new bool[num_cl_dev];

        cuda_ptr_list = new CUdeviceptr[num_cuda_dev];
        cuda_valid_list = new bool[num_cuda_dev];

        clear(env);
    }

    Buffer(Buffer const &rhs) = delete;
    Buffer &operator = (Buffer const &rhs) = delete;
    Buffer &operator = (Buffer &&rhs) = delete;

    ~Buffer() {
        release(env);

        delete [] cl_ptr_list;
        delete [] cl_valid_list;
        delete [] cuda_ptr_list;
        delete [] cuda_valid_list;
    }

    void clear(ComputeEnv *env) {
        int num_cl_dev = env->num_cl_dev;
        int num_cuda_dev = env->num_cuda_dev;
        int i;

        for (i=0; i<num_cl_dev; i++) {
            cl_valid_list[i] = false;
            cl_ptr_list[i] = nullptr;
        }
        for (i=0; i<num_cuda_dev; i++) {
            cuda_valid_list[i] = false;
            cuda_ptr_list[i] = 0;
        }

        host_valid = false;
        host_ptr = nullptr;
    }

    void release(ComputeEnv *env) {
        int num_cl_dev = env->num_cl_dev;
        int num_cuda_dev = env->num_cuda_dev;
        int i;

        for (i=0; i<num_cl_dev; i++) {
            if (cl_ptr_list[i]) {
                clReleaseMemObject(cl_ptr_list[i]);
            }

            cl_ptr_list[i] = nullptr;
            cl_valid_list[i] = false;
        }
        for (i=0; i<num_cuda_dev; i++) {
            if (cuda_ptr_list[i]) {
                cuMemFree(cuda_ptr_list[i]);
            }

            cuda_ptr_list[i] = 0;
            cuda_valid_list[i] = false;
        }

        if (host_ptr) {
            _mm_free(host_ptr);
        }
        host_ptr = nullptr;
        host_valid = false;
    }

    void invalidate(ComputeEnv *env) {
        int num_cl_dev = env->num_cl_dev;
        int num_cuda_dev = env->num_cuda_dev;
        int i;

        for (i=0; i<num_cl_dev; i++) {
            cl_valid_list[i] = false;
        }
        for (i=0; i<num_cuda_dev; i++) {
            cuda_valid_list[i] = false;
        }

        host_valid = false;
    }



    cl_mem get_read_ptr_cl(ComputeEnv *env,int devid) {
        if (cl_valid_list[devid]) {
            return cl_ptr_list[devid];
        }

        if (host_valid == false) {
            /* xx */
            abort();
            return nullptr;
        }

        OpenCLDev *dev = &env->cl_dev_list[devid];
        if (cl_ptr_list[devid] == nullptr) {
            cl_int err;
            cl_ptr_list[devid] = clCreateBuffer(dev->context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                byte_size, host_ptr, &err);
        } else {
            clEnqueueWriteBuffer(dev->queue, cl_ptr_list[devid],
                                 CL_TRUE, 0, byte_size, host_ptr, 0, nullptr, nullptr);
        }

        cl_valid_list[devid] = true;

        return cl_ptr_list[devid];
    }

    cl_mem get_write_ptr_cl(ComputeEnv *env,int devid) {
        invalidate(env);

        OpenCLDev *dev = &env->cl_dev_list[devid];
        if (cl_ptr_list[devid] == nullptr) {
            cl_int err;
            cl_ptr_list[devid] = clCreateBuffer(dev->context,
                                                CL_MEM_READ_WRITE,
                                                byte_size, nullptr, &err);
        }

        last_write.type = Processor::OpenCL;
        last_write.devid = devid;

        cl_valid_list[devid] = true;
        return cl_ptr_list[devid];
    }


    CUdeviceptr get_read_ptr_cuda(ComputeEnv *env,int devid) {
        if (cuda_valid_list[devid]) {
            return cuda_ptr_list[devid];
        }

        if (host_valid == false) {
            /* xx */
            abort();
            return 0;
        }

        CUDADev *dev = &env->cuda_dev_list[devid];
        cuCtxPushCurrent(dev->context);

        if (cuda_ptr_list[devid] == 0) {
            CUresult err;
            err = cuMemAlloc(&cuda_ptr_list[devid], byte_size);
            if (err != CUDA_SUCCESS) {
                abort();
            }
        }

        cuMemcpyHtoD(cuda_ptr_list[devid], host_ptr, byte_size);
        cuda_valid_list[devid] = true;

        CUcontext old;
        cuCtxPopCurrent(&old);

        return cuda_ptr_list[devid];
    }

    CUdeviceptr get_write_ptr_cuda(ComputeEnv *env,int devid) {
        invalidate(env);

        CUDADev *dev = &env->cuda_dev_list[devid];
        cuCtxPushCurrent(dev->context);

        if (cuda_ptr_list[devid] == 0) {
            CUresult err;
            err = cuMemAlloc(&cuda_ptr_list[devid], byte_size);
            if (err != CUDA_SUCCESS) {
                abort();
            }
        }

        last_write.type = Processor::CUDA;
        last_write.devid = devid;

        cuda_valid_list[devid] = true;
        CUcontext old;
        cuCtxPopCurrent(&old);

        return cuda_ptr_list[devid];
    }



    void *get_read_ptr_host(ComputeEnv *env) {
        if (host_valid) {
            return host_ptr;
        }

        if (host_ptr == nullptr) {
            host_ptr = _mm_malloc(byte_size, 64);
        }

        if (last_write.type == Processor::OpenCL) {
            OpenCLDev *dev = &env->cl_dev_list[last_write.devid];
            clEnqueueReadBuffer(dev->queue, cl_ptr_list[last_write.devid],
                                CL_TRUE, 0, byte_size, host_ptr, 0, nullptr, nullptr);
        } else if (last_write.type == Processor::CUDA) {
            CUDADev *dev = &env->cuda_dev_list[last_write.devid];
            cuCtxPushCurrent(dev->context);
            cuMemcpyDtoH(host_ptr, cuda_ptr_list[last_write.devid], byte_size);
        } else {
            abort();
        }

        host_valid = true;
        return host_ptr;
    }

    void *get_write_ptr_host(ComputeEnv *env) {
        invalidate(env);

        last_write.type = Processor::HOST;
        last_write.devid = 0;

        if (host_ptr == nullptr) {
            host_ptr = _mm_malloc(byte_size, 64);
        }

        host_valid = true;

        return host_ptr;
    }
};

#endif
