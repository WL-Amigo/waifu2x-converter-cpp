/*
* The MIT License (MIT)
* Copyright (c) 2015 amigo(white luckers), tanakamura, DeadSix27, YukihoAA and contributors
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

#ifndef CUDALIB_H
#define CUDALIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef uintptr_t CUdeviceptr;

typedef enum cudaError_enum
{
	CUDA_SUCCESS = 0
} CUresult;

typedef enum CUjit_option_enum
{
	CU_JIT_MAX_REGISTERS = 0,
	CU_JIT_THREADS_PER_BLOCK = 1,
	CU_JIT_WALL_TIME = 2,
	CU_JIT_INFO_LOG_BUFFER = 3,
	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
	CU_JIT_ERROR_LOG_BUFFER = 5,
	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
	CU_JIT_OPTIMIZATION_LEVEL = 7,
	CU_JIT_CACHE_MODE=14,
} CUjit_option;

typedef enum CUdevice_attribute_enum
{
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
} CUdevice_attribute;

typedef enum CUjit_cacheMode_enum
{
	CU_JIT_CACHE_OPTION_NONE = 0,
	CU_JIT_CACHE_OPTION_CG,
	CU_JIT_CACHE_OPTION_CA
} CUjit_cacheMode;

typedef enum CUfunc_cache_enum
{
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3
} CUfunc_cache;

typedef enum CUsharedconfig_enum
{
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2
} CUsharedconfig;

typedef int CUdevice;

typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

#define CU_CTX_SCHED_BLOCKING_SYNC 0x4

typedef CUresult (CUDAAPI * tcuInit)(unsigned int Flags);
typedef CUresult (CUDAAPI * tcuDriverGetVersion)(int *ver);
typedef CUresult (CUDAAPI * tcuDeviceGetCount)(int *count);
typedef CUresult (CUDAAPI * tcuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (CUDAAPI * tcuCtxCreate)(CUcontext *ret, unsigned int flags, CUdevice dev);
typedef CUresult (CUDAAPI * tcuCtxDestroy)(CUcontext ret);
typedef CUresult (CUDAAPI * tcuModuleLoadData)(CUmodule *module, const void *image);
typedef CUresult (CUDAAPI * tcuModuleLoadDataEx)(CUmodule *module, const void *image, unsigned int n, CUjit_option *o, void **ov);
typedef CUresult (CUDAAPI * tcuModuleUnload)(CUmodule mod);
typedef CUresult (CUDAAPI * tcuModuleGetFunction)(CUfunction *hfunc, CUmodule mod, const char *name);
typedef CUresult (CUDAAPI * tcuStreamCreate)(CUstream *str, unsigned int Flags);
typedef CUresult (CUDAAPI * tcuStreamDestroy)(CUstream str);
typedef CUresult (CUDAAPI * tcuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (CUDAAPI * tcuMemFree)(CUdeviceptr dptr);
typedef CUresult (CUDAAPI * tcuCtxSetCurrent)(CUcontext ctxt);
typedef CUresult (CUDAAPI * tcuCtxPushCurrent)(CUcontext ctxt);
typedef CUresult (CUDAAPI * tcuCtxPopCurrent)(CUcontext *ctxt);
typedef CUresult (CUDAAPI * tcuStreamSynchronize)(CUstream stream);
typedef CUresult (CUDAAPI * tcuMemcpyHtoD)(CUdeviceptr dst, const void *src, size_t byte);
typedef CUresult (CUDAAPI * tcuMemcpyHtoDAsync)(CUdeviceptr dst, const void *src, size_t byte, CUstream str);
typedef CUresult (CUDAAPI * tcuMemcpyDtoH)(void *dst, CUdeviceptr src, size_t byte);
typedef CUresult (CUDAAPI * tcuLaunchKernel)
(
	CUfunction f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes,
	CUstream str, void **kernelParams, void **extra
);
typedef CUresult (CUDAAPI * tcuCtxSetCacheConfig)(CUfunc_cache c);
typedef CUresult (CUDAAPI * tcuFuncSetSharedMemConfig)(CUfunction func, CUsharedconfig config);
typedef CUresult (CUDAAPI * tcuCtxSetSharedMemConfig)(CUsharedconfig config);
typedef CUresult (CUDAAPI * tcuDeviceGetAttribute)(int *pi, CUdevice_attribute attr, CUdevice dev);
typedef CUresult (CUDAAPI * tcuProfilerStart)(void);

#define FOR_EACH_CUDA_FUNC(F,F_V2) \
    F(cuInit)                      \
    F(cuDriverGetVersion)          \
    F(cuDeviceGetCount)            \
    F(cuDeviceGetName)             \
    F_V2(cuCtxCreate)              \
    F_V2(cuCtxDestroy)             \
    F(cuModuleLoadData)            \
    F(cuModuleLoadDataEx)          \
    F(cuModuleUnload)              \
    F(cuModuleGetFunction)         \
    F(cuStreamCreate)              \
    F_V2(cuStreamDestroy)          \
    F_V2(cuMemAlloc)               \
    F_V2(cuMemFree)                \
    F_V2(cuMemcpyHtoD)             \
    F_V2(cuMemcpyHtoDAsync)        \
    F_V2(cuMemcpyDtoH)             \
    F(cuCtxSetCurrent)             \
    F(cuStreamSynchronize)         \
    F_V2(cuCtxPushCurrent)         \
    F_V2(cuCtxPopCurrent)          \
    F(cuLaunchKernel)              \
    F(cuCtxSetCacheConfig)         \
    F(cuFuncSetSharedMemConfig)    \
    F(cuCtxSetSharedMemConfig)     \
    F(cuDeviceGetAttribute)        \
    F(cuProfilerStart)             \


#define CUDA_PROTOTYPE(name) \
    extern t##name name;

FOR_EACH_CUDA_FUNC(CUDA_PROTOTYPE,CUDA_PROTOTYPE)

#endif
