#ifndef CUDALIB_H
#define CUDALIB_H

#include <stddef.h>

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef uintptr_t CUdeviceptr;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0
} CUresult;
typedef int CUdevice;

typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

#define CU_CTX_SCHED_BLOCKING_SYNC 0x4

typedef CUresult (* CUDAAPI tcuInit)(unsigned int Flags);
typedef CUresult (* CUDAAPI tcuDriverGetVersion)(int *ver);
typedef CUresult (* CUDAAPI tcuDeviceGetCount)(int *count);
typedef CUresult (* CUDAAPI tcuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (* CUDAAPI tcuCtxCreate)(CUcontext *ret, unsigned int flags, CUdevice dev);
typedef CUresult (* CUDAAPI tcuCtxDestroy)(CUcontext ret);
typedef CUresult (* CUDAAPI tcuModuleLoadData)(CUmodule *module, const void *image);
typedef CUresult (* CUDAAPI tcuModuleUnload)(CUmodule mod);
typedef CUresult (* CUDAAPI tcuModuleGetFunction)(CUfunction *hfunc, CUmodule mod, const char *name);
typedef CUresult (* CUDAAPI tcuStreamCreate)(CUstream *str, unsigned int Flags);
typedef CUresult (* CUDAAPI tcuStreamDestroy)(CUstream str);
typedef CUresult (* CUDAAPI tcuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (* CUDAAPI tcuMemFree)(CUdeviceptr dptr);
typedef CUresult (* CUDAAPI tcuCtxSetCurrent)(CUcontext ctxt);
typedef CUresult (* CUDAAPI tcuCtxPushCurrent)(CUcontext ctxt);
typedef CUresult (* CUDAAPI tcuCtxPopCurrent)(CUcontext *ctxt);
typedef CUresult (* CUDAAPI tcuStreamSynchronize)(CUstream stream);
typedef CUresult (* CUDAAPI tcuMemcpyHtoD)(CUdeviceptr dst, const void *src, size_t byte);
typedef CUresult (* CUDAAPI tcuMemcpyDtoH)(void *dst, CUdeviceptr src, size_t byte);
typedef CUresult (* CUDAAPI tcuLaunchKernel)(CUfunction f,
                                             unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                             unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                             unsigned int sharedMemBytes,
                                             CUstream str, void **kernelParams, void **extra);

#define FOR_EACH_CUDA_FUNC(F)                   \
    F(cuInit)                                   \
    F(cuDriverGetVersion)                       \
    F(cuDeviceGetCount)                         \
    F(cuDeviceGetName)                          \
    F(cuCtxCreate)                              \
    F(cuCtxDestroy)                              \
    F(cuModuleLoadData)                          \
    F(cuModuleUnload)                            \
    F(cuModuleGetFunction)                       \
    F(cuStreamCreate)                            \
    F(cuStreamDestroy)                            \
    F(cuMemAlloc)                                 \
    F(cuMemFree)                                  \
    F(cuMemcpyHtoD)                               \
    F(cuMemcpyDtoH)                               \
    F(cuCtxSetCurrent)                            \
    F(cuStreamSynchronize)                        \
    F(cuCtxPushCurrent)                           \
    F(cuCtxPopCurrent)                           \
    F(cuLaunchKernel)


#define CUDA_PROTOTYPE(name)                    \
    extern t##name name;

FOR_EACH_CUDA_FUNC(CUDA_PROTOTYPE)

#endif
