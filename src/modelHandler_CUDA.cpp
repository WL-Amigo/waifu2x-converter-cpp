#include "CUDAlib.h"
#include "Buffer.hpp"
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
static HMODULE handle;
#else
#include <dlfcn.h>
static void *handle;
#endif

static const char prog[] = 
#include "modelHandler_CUDA.ptx.h"
	;


#define CUDA_DECL(name)				\
	t##name name;

FOR_EACH_CUDA_FUNC(CUDA_DECL)

namespace w2xc {

static int
cudalib_init(void)
{
#ifdef _WIN32
	handle = LoadLibrary("nvcuda.dll");
#else
	handle = dlopen("libcuda.so.1", RTLD_LAZY);

#define GetProcAddress dlsym

#endif
	if (!handle) {
		return -1;
	}

#define LOAD(name)					\
	name = (t##name) GetProcAddress(handle, #name); \
	if (name == NULL) {				\
		return -1;				\
	}

	FOR_EACH_CUDA_FUNC(LOAD);

	return 0;
}


bool
initCUDA(ComputeEnv *env)
{
	if (cudalib_init() < 0) {
		return false;
	}

	CUresult r = cuInit(0);
	if (r != CUDA_SUCCESS) {
		return false;
	}

	int dev_count;
	cuDeviceGetCount(&dev_count);

	if (dev_count == 0) {
		return false;
	}

	CUcontext ctxt;
	CUdevice dev = 0;
	CUmodule mod;

	r = cuCtxCreate(&ctxt, 0, dev);
	if (r != CUDA_SUCCESS) {
		return false;
	}

	r = cuModuleLoadData(&mod, prog);
	if (r != CUDA_SUCCESS) {
		cuCtxDestroy(ctxt);
		return false;
	}

	CUfunction filter;

	r = cuModuleGetFunction(&filter, mod, "filter");
	if (r != CUDA_SUCCESS) {
		cuModuleUnload(mod);
		cuCtxDestroy(ctxt);
	}

	char name [1024];
	cuDeviceGetName(name, sizeof(name), dev);
	printf("CUDA : %s\n", name);

	env->num_cuda_dev = 1;
	env->cuda_dev_list = new CUDADev[1];
	env->cuda_dev_list[0].dev = dev;
	env->cuda_dev_list[0].context = ctxt;
	env->cuda_dev_list[0].mod = mod;
	env->cuda_dev_list[0].filter = filter;
}

}
