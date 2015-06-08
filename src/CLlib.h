#ifndef CLLIB_H
#define CLLIB_H

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

#ifndef CLLIB_EXTERN
#define CLLIB_EXTERN extern
#endif

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clGetDeviceInfo)(cl_device_id    /* device */,
                     cl_device_info  /* param_name */, 
                     size_t          /* param_value_size */, 
                     void *          /* param_value */,
                     size_t *        /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clGetDeviceInfo p_clGetDeviceInfo

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clGetPlatformIDs)(cl_uint          /* num_entries */,
                      cl_platform_id * /* platforms */,
                      cl_uint *        /* num_platforms */) CL_API_SUFFIX__VERSION_1_0;

#define clGetPlatformIDs p_clGetPlatformIDs

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clGetDeviceIDs)(cl_platform_id   /* platform */,
    cl_device_type   /* device_type */, 
    cl_uint          /* num_entries */, 
    cl_device_id *   /* devices */, 
    cl_uint *        /* num_devices */) CL_API_SUFFIX__VERSION_1_0;

#define clGetDeviceIDs p_clGetDeviceIDs

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL 
(*p_clGetPlatformInfo)(cl_platform_id   /* platform */, 
                       cl_platform_info /* param_name */,
                       size_t           /* param_value_size */, 
                       void *           /* param_value */,
                       size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clGetPlatformInfo p_clGetPlatformInfo

CLLIB_EXTERN CL_API_ENTRY cl_program CL_API_CALL
(*p_clCreateProgramWithSource)(cl_context        /* context */,
                               cl_uint           /* count */,
                               const char **     /* strings */,
                               const size_t *    /* lengths */,
                               cl_int *          /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clCreateProgramWithSource p_clCreateProgramWithSource

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clBuildProgram)(cl_program           /* program */,
                    cl_uint              /* num_devices */,
                    const cl_device_id * /* device_list */,
                    const char *         /* options */, 
                    void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
                    void *               /* user_data */) CL_API_SUFFIX__VERSION_1_0;

#define clBuildProgram p_clBuildProgram


CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clGetProgramBuildInfo)(cl_program            /* program */,
                          cl_device_id          /* device */,
                           cl_program_build_info /* param_name */,
                           size_t                /* param_value_size */,
                           void *                /* param_value */,
                           size_t *              /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clGetProgramBuildInfo p_clGetProgramBuildInfo

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clReleaseProgram)(cl_program /* program */) CL_API_SUFFIX__VERSION_1_0;
#define clReleaseProgram p_clReleaseProgram

CLLIB_EXTERN CL_API_ENTRY cl_kernel CL_API_CALL
(*p_clCreateKernel)(cl_program      /* program */,
                   const char *    /* kernel_name */,
                    cl_int *        /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clCreateKernel p_clCreateKernel


CLLIB_EXTERN CL_API_ENTRY cl_mem CL_API_CALL
(*p_clCreateBuffer)(cl_context   /* context */,
                    cl_mem_flags /* flags */,
                    size_t       /* size */,
                    void *       /* host_ptr */,
                    cl_int *     /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clCreateBuffer p_clCreateBuffer

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clEnqueueWriteBuffer)(cl_command_queue   /* command_queue */, 
                          cl_mem             /* buffer */, 
                          cl_bool            /* blocking_write */, 
                          size_t             /* offset */, 
                          size_t             /* cb */, 
                          const void *       /* ptr */, 
                          cl_uint            /* num_events_in_wait_list */, 
                          const cl_event *   /* event_wait_list */, 
                          cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_0;

#define clEnqueueWriteBuffer p_clEnqueueWriteBuffer

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clFlush)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

#define clFlush p_clFlush

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clReleaseMemObject)(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;

#define clReleaseMemObject p_clReleaseMemObject

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clEnqueueReadBuffer)(cl_command_queue    /* command_queue */,
                         cl_mem              /* buffer */,
                         cl_bool             /* blocking_read */,
                         size_t              /* offset */,
                         size_t              /* cb */, 
                         void *              /* ptr */,
                         cl_uint             /* num_events_in_wait_list */,
                         const cl_event *    /* event_wait_list */,
                         cl_event *          /* event */) CL_API_SUFFIX__VERSION_1_0;

#define clEnqueueReadBuffer p_clEnqueueReadBuffer

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clFinish)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

#define clFinish p_clFinish

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clEnqueueNDRangeKernel)(cl_command_queue /* command_queue */,
                            cl_kernel        /* kernel */,
                            cl_uint          /* work_dim */,
                            const size_t *   /* global_work_offset */,
                            const size_t *   /* global_work_size */,
                            const size_t *   /* local_work_size */,
                            cl_uint          /* num_events_in_wait_list */,
                            const cl_event * /* event_wait_list */,
                            cl_event *       /* event */) CL_API_SUFFIX__VERSION_1_0;

#define clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clReleaseKernel)(cl_kernel   /* kernel */) CL_API_SUFFIX__VERSION_1_0;

#define clReleaseKernel p_clReleaseKernel


CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clSetKernelArg)(cl_kernel    /* kernel */,
                    cl_uint      /* arg_index */,
                    size_t       /* arg_size */,
                    const void * /* arg_value */) CL_API_SUFFIX__VERSION_1_0;

#define clSetKernelArg p_clSetKernelArg


CLLIB_EXTERN CL_API_ENTRY cl_context CL_API_CALL
(*p_clCreateContext)(const cl_context_properties * /* properties */,
                     cl_uint                       /* num_devices */,
                     const cl_device_id *          /* devices */,
                     void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
                     void *                        /* user_data */,
                     cl_int *                      /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clCreateContext p_clCreateContext

CLLIB_EXTERN CL_API_ENTRY cl_command_queue CL_API_CALL
(*p_clCreateCommandQueue)(cl_context                     /* context */, 
                          cl_device_id                   /* device */, 
                          cl_command_queue_properties    /* properties */,
                          cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

#define clCreateCommandQueue p_clCreateCommandQueue


CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clReleaseCommandQueue)(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

#define clReleaseCommandQueue p_clReleaseCommandQueue


CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clReleaseContext)(cl_context /* context */) CL_API_SUFFIX__VERSION_1_0;

#define clReleaseContext p_clReleaseContext

CLLIB_EXTERN CL_API_ENTRY cl_int CL_API_CALL
(*p_clWaitForEvents)(cl_uint             /* num_events */,
                     const cl_event *    /* event_list */) CL_API_SUFFIX__VERSION_1_0;

#define clWaitForEvents p_clWaitForEvents



#ifdef __cplusplus
}
#endif


#endif
