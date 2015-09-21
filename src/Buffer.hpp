#ifndef BUFFER_HPP
#define BUFFER_HPP

#if defined(_MSC_VER)
#include <malloc.h>
#elif defined(__GNUC__)
#include <mm_malloc.h>
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



    cl_mem get_read_ptr_cl(ComputeEnv *env,int devid, size_t read_byte_size) {
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
                                                CL_MEM_READ_WRITE,
                                                byte_size, nullptr, &err);
        }
        clEnqueueWriteBuffer(dev->queue, cl_ptr_list[devid],
                             CL_TRUE, 0, read_byte_size, host_ptr, 0, nullptr, nullptr);

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


    CUdeviceptr get_read_ptr_cuda(ComputeEnv *env,int devid, size_t read_byte_size) {
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

        //double t0 = getsec();
        cuMemcpyHtoD(cuda_ptr_list[devid], host_ptr, read_byte_size);
        //double t1 = getsec();
        //env->transfer_wait = t1-t0;

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



    void *get_read_ptr_host(ComputeEnv *env, size_t read_byte_size) {
        if (host_valid) {
            return host_ptr;
        }

        if (host_ptr == nullptr) {
            host_ptr = _mm_malloc(byte_size, 64);
        }

        if (last_write.type == Processor::OpenCL) {
            OpenCLDev *dev = &env->cl_dev_list[last_write.devid];
            clEnqueueReadBuffer(dev->queue, cl_ptr_list[last_write.devid],
                                CL_TRUE, 0, read_byte_size, host_ptr, 0, nullptr, nullptr);
        } else if (last_write.type == Processor::CUDA) {
            CUDADev *dev = &env->cuda_dev_list[last_write.devid];
            cuCtxPushCurrent(dev->context);
            //double t0 = getsec();
            cuMemcpyDtoH(host_ptr, cuda_ptr_list[last_write.devid], read_byte_size);
            //double t1 = getsec();
            //env->transfer_wait = t1-t0;

            CUcontext old;
            cuCtxPopCurrent(&old);
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

    bool prealloc(ComputeEnv *env) {
        int devid;
        if (host_ptr == nullptr) {
            host_ptr = _mm_malloc(byte_size, 64);
            if (host_ptr == nullptr) {
                return false;
            }
        }

        switch (env->target_processor.type) {
        case W2XCONV_PROC_HOST:
            break;

        case W2XCONV_PROC_OPENCL:
            devid = env->target_processor.devid;
            if (cl_ptr_list[devid] == nullptr) {
                cl_int err;
                OpenCLDev *dev = &env->cl_dev_list[devid];
                cl_ptr_list[devid] = clCreateBuffer(dev->context,
                                                    CL_MEM_READ_WRITE,
                                                    byte_size, nullptr, &err);
                if (cl_ptr_list[devid] == nullptr) {
                    return false;
                }

                /* touch memory to force allocation */
                char data = 0;
                err = clEnqueueWriteBuffer(dev->queue, cl_ptr_list[devid],
                                           CL_TRUE, 0, 1, &data, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    clReleaseMemObject(cl_ptr_list[devid]);
                    cl_ptr_list[devid] = nullptr;
                    return false;
                }

            }
            break;

        case W2XCONV_PROC_CUDA:
            devid = env->target_processor.devid;

            if (cuda_ptr_list[devid] == 0) {
                CUresult err;
                CUDADev *dev = &env->cuda_dev_list[devid];
                cuCtxPushCurrent(dev->context);
                err = cuMemAlloc(&cuda_ptr_list[devid], byte_size);
                CUcontext old;
                cuCtxPopCurrent(&old);

                if (err != CUDA_SUCCESS) {
                    return false;
                }
            }
            break;

        }

        return true;
    }
};

#endif
