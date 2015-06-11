#define CLLIB_EXTERN

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "modelHandler.hpp"
#include "common.hpp"
#include "sec.hpp"
#include "CLlib.h"

static const char prog[] = 
#include "modelHandler_OpenCL.cl.h"
        ;

#define S_(a) #a
#define S(a) S_(a)

namespace w2xc {


#ifdef _WIN32
static HMODULE handle;
#else
static void *handle;
#endif

static int
cllib_init(void)
{
#ifdef _WIN32
        handle = LoadLibrary("OpenCL.dll");
#else
        handle = dlopen("libOpenCL.so.1", RTLD_LAZY);

#define GetProcAddress dlsym

#endif

        if (!handle) {
                return -1;
        }

#define LOAD(name)                              \
        p_##name = (__decltype(p_##name)) GetProcAddress(handle, #name); \
        if (p_##name == NULL) {                 \
                return -1;                      \
        }

        LOAD(clGetDeviceInfo);
        LOAD(clGetPlatformIDs);
        LOAD(clGetDeviceIDs);
        LOAD(clGetPlatformInfo);
        LOAD(clCreateProgramWithSource);
        LOAD(clBuildProgram);
        LOAD(clGetProgramBuildInfo);
        LOAD(clReleaseProgram);
        LOAD(clCreateKernel);
        LOAD(clCreateBuffer);
        LOAD(clEnqueueWriteBuffer);
        LOAD(clFlush);
        LOAD(clReleaseMemObject);
        LOAD(clEnqueueReadBuffer);
        LOAD(clFinish);
        LOAD(clEnqueueNDRangeKernel);
        LOAD(clReleaseKernel);
        LOAD(clSetKernelArg);
        LOAD(clCreateCommandQueue);
        LOAD(clCreateContext);
        LOAD(clReleaseCommandQueue);
        LOAD(clReleaseContext);
        LOAD(clWaitForEvents);

        return 0;
}

bool
initOpenCL(ComputeEnv *env)
{
        int r = cllib_init();
        if (r < 0) {
                return false;
        }

        cl_uint num_plt;
        cl_platform_id plts[16];
        clGetPlatformIDs(16, plts, &num_plt);
        bool found = false;
        cl_int err;

        cl_platform_id platform;
        cl_context context;
        cl_device_id dev;
        cl_command_queue queue;
        cl_kernel ker;
        cl_program program;

        for (unsigned int i=0; i<num_plt; i++) {
                size_t sz;
                cl_uint num_dev;

                clGetPlatformInfo(plts[i], CL_PLATFORM_NAME, 0, nullptr, &sz);
                std::vector<char> name(sz);
                clGetPlatformInfo(plts[i], CL_PLATFORM_NAME, sz, &name[0], &sz);

                if (strstr(&name[0], "Intel") != NULL) {
                        continue;
                }

                clGetDeviceIDs(plts[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_dev);
                if (num_dev == 0) {
                        continue;
                }

                std::vector<cl_device_id> devs(num_dev);
                clGetDeviceIDs(plts[i], CL_DEVICE_TYPE_GPU, num_dev, &devs[0], &num_dev);

                platform = plts[i];
                dev = devs[0];


                cl_context_properties props[] =
                        {CL_CONTEXT_PLATFORM, (cl_context_properties)(plts[i]), 0};
                cl_context ctxt = clCreateContext(props, 1, &devs[0], NULL, NULL, &err);
                if (err != CL_SUCCESS) {
                        continue;
                }

                context = ctxt;

                found = true;
                break;
        }

        if (!found) {
                return false;
        }

        const char *source[1] = {prog};
        size_t src_len[1] = {sizeof(prog)-1};

        program = clCreateProgramWithSource(context, 1, source, src_len, &err);
        if (err != CL_SUCCESS) {
                clReleaseContext(context);
                return false;
        }

        err = clBuildProgram(program, 1, &dev, "-DVEC_WIDTH=" S(GPU_VEC_WIDTH), nullptr, nullptr);

        if (err != CL_SUCCESS) {
                size_t log_len;
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_len);

                std::vector<char> log(log_len+1);
                clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_len, &log[0], &log_len);
                log[log_len] = '\0';

                puts(&log[0]);

                clReleaseProgram(program);
                clReleaseContext(context);
                return false;
        }

        ker = clCreateKernel(program, "filter", &err);
        if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                clReleaseContext(context);
                return false;
        }

        queue = clCreateCommandQueue(context, dev, 0, &err);
        if (err != CL_SUCCESS) {
                clReleaseProgram(program);
                clReleaseContext(context);
                clReleaseKernel(ker);
                return false;
        }

        size_t dev_name_len;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &dev_name_len);
        std::vector<char> dev_name(dev_name_len+1);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, dev_name_len, &dev_name[0], &dev_name_len);

        printf("use GPU: %s\n",
               &dev_name[0]);

        env->num_cl_dev = 1;
        env->cl_dev_list = new OpenCLDev[1];

        env->cl_dev_list[0].platform = platform;
        env->cl_dev_list[0].context = context;
        env->cl_dev_list[0].devid = dev;
        env->cl_dev_list[0].queue = queue;
        env->cl_dev_list[0].ker = ker;

        return true;
}


void
filter_OpenCL_impl(ComputeEnv *env,
                   Buffer *packed_input_buf,
                   Buffer *packed_output_buf,
                   int nInputPlanes,
                   int nOutputPlanes,
                   const float *fbiases,
                   const float *weight,
                   cv::Size ipSize,
                   int nJob)
{
        int w = ipSize.width;
        int h = ipSize.height;
        cl_int err;

        OpenCLDev *dev = &env->cl_dev_list[0];
        cl_context context = dev->context;

        cl_mem cl_packed_input = packed_input_buf->get_read_ptr_cl(env, 0);
        cl_mem cl_packed_output = packed_output_buf->get_write_ptr_cl(env, 0);

        cl_mem cl_fbiases = clCreateBuffer(context,
                                           CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * nOutputPlanes,
                                           (void*)fbiases, &err
                );

        cl_mem cl_weight = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * GPU_VEC_WIDTH * nInputPlanes * 9,
                                          (void*)weight, &err
                );

        int ai = 0;

        clSetKernelArg(dev->ker, ai++, sizeof(cl_mem), &cl_packed_input);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_int), &nInputPlanes);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_mem), &cl_packed_output);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_int), &nOutputPlanes);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_mem), &cl_fbiases);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_int), &h);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_int), &w);
        clSetKernelArg(dev->ker, ai++, sizeof(cl_mem), &cl_weight);

        size_t local_size = 0;
        //local_size += sizeof(float) * 256;
        //local_size += sizeof(float) * GPU_VEC_WIDTH;
        local_size += sizeof(float) * nInputPlanes * (GPU_BLOCK_SIZE+2) * 3;

        clSetKernelArg(dev->ker, ai++, local_size, nullptr);

        cl_event event;

        size_t vec_width = std::min(GPU_VEC_WIDTH, nOutputPlanes);

        unsigned int nout = 2;
        if (nOutputPlanes == 1) {
                nout = 1;
        }

        size_t gws[3] = {h*vec_width, 1, 1};
        size_t lws[3] = {vec_width, 1, 1};

        err = clEnqueueNDRangeKernel(dev->queue,
                                     dev->ker,
                                     1,
                                     nullptr, gws, lws,
                                     0, nullptr, &event);
        if (err != CL_SUCCESS) {
                printf("enqueue ndrange error : %d\n", err);
                exit(1);
        }

        err = clWaitForEvents(1, &event);
        if (err != CL_SUCCESS) {
                printf("wait ndrange error : %d\n", err);
                exit(1);
        }

        if (err != CL_SUCCESS) {
                printf("read buffer error : %d\n", err);
                exit(1);
        }

        clReleaseMemObject(cl_fbiases);
        clReleaseMemObject(cl_weight);
}

}
