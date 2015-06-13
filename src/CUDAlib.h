#ifndef CUDALIB_H
#define CUDALIB_H

#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0
} CUresult;
typedef int CUdevice;

typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;

typedef CUresult (* CUDAAPI tcuInit)(unsigned int Flags);
typedef CUresult (* CUDAAPI tcuDriverGetVersion)(int *ver);
typedef CUresult (* CUDAAPI tcuDeviceGetCount)(int *count);
typedef CUresult (* CUDAAPI tcuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (* CUDAAPI tcuCtxCreate)(CUcontext *ret, unsigned int flags, CUdevice dev);
typedef CUresult (* CUDAAPI tcuCtxDestroy)(CUcontext ret);
typedef CUresult (* CUDAAPI tcuModuleLoadData)(CUmodule *module, const void *image);
typedef CUresult (* CUDAAPI tcuModuleUnload)(CUmodule mod);
typedef CUresult (* CUDAAPI tcuModuleGetFunction)(CUfunction *hfunc, CUmodule mod, const char *name);

#define FOR_EACH_CUDA_FUNC(F)                   \
    F(cuInit)                                   \
    F(cuDriverGetVersion)                       \
    F(cuDeviceGetCount)                         \
    F(cuDeviceGetName)                          \
    F(cuCtxCreate)                              \
    F(cuCtxDestroy)                              \
    F(cuModuleLoadData)                          \
    F(cuModuleUnload)                            \
    F(cuModuleGetFunction)


#define CUDA_PROTOTYPE(name)                    \
    extern t##name name;

FOR_EACH_CUDA_FUNC(CUDA_PROTOTYPE)

#endif
