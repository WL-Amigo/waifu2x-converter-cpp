#include "CUDAlib.h"
#include "Buffer.hpp"
#include "params.h"
#include <stdio.h>
#include <string.h>

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

FOR_EACH_CUDA_FUNC(CUDA_DECL, CUDA_DECL)

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

#define LOAD_V2(name)					\
	name = (t##name) GetProcAddress(handle, #name "_v2"); \
	if (name == NULL) {				\
		return -1;				\
	}

	FOR_EACH_CUDA_FUNC(LOAD, LOAD_V2);

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
	CUstream stream;

	r = cuStreamCreate(&stream, 0);

	r = cuCtxCreate(&ctxt, CU_CTX_SCHED_BLOCKING_SYNC, dev);
	if (r != CUDA_SUCCESS) {
		return false;
	}

	r = cuStreamCreate(&stream, 0);
	if (r != CUDA_SUCCESS) {
		cuCtxDestroy(ctxt);
		return false;
	}

	r = cuModuleLoadData(&mod, prog);
	if (r != CUDA_SUCCESS) {
		cuCtxDestroy(ctxt);
		cuStreamDestroy(stream);
		return false;
	}

	CUfunction filter;

	r = cuModuleGetFunction(&filter, mod, "filter");
	if (r != CUDA_SUCCESS) {
		cuModuleUnload(mod);
		cuCtxDestroy(ctxt);
		cuStreamDestroy(stream);
		return false;
	}

	char name [1024];
	cuDeviceGetName(name, sizeof(name), dev);
	printf("CUDA : %s\n", name);

	cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);

	env->num_cuda_dev = 1;
	env->cuda_dev_list = new CUDADev[1];
	env->cuda_dev_list[0].dev = dev;
	env->cuda_dev_list[0].context = ctxt;
	env->cuda_dev_list[0].mod = mod;
	env->cuda_dev_list[0].filter = filter;
	env->cuda_dev_list[0].stream = stream;

	return true;
}


void
filter_CUDA_impl(ComputeEnv *env,
		 Buffer *packed_input_buf,
		 Buffer *packed_output_buf,
		 int nInputPlanes,
		 int nOutputPlanes,
		 const float *biases,
		 const float *weight,
		 int ip_width,
		 int ip_height,
		 int nJob)
{
	CUresult r;
	CUDADev *dev = &env->cuda_dev_list[0];
	CUdeviceptr packed_input = packed_input_buf->get_read_ptr_cuda(env, 0);
	CUdeviceptr packed_output = packed_output_buf->get_write_ptr_cuda(env, 0);

	cuCtxPushCurrent(dev->context);

	CUdeviceptr d_fbiases = 0;
	size_t bias_size = sizeof(float) * nOutputPlanes;
	r = cuMemAlloc(&d_fbiases, bias_size);
	if (r != CUDA_SUCCESS) {
		printf("fail: alloc bias %d.", (int)r);
		exit(1);
	}
	r = cuMemcpyHtoD(d_fbiases, biases, bias_size);
	if (r != CUDA_SUCCESS) {
		puts("fail: copy to bias");
		exit(1);
	}
	CUdeviceptr d_weight = 0;
	size_t weight_size = sizeof(float) * 128 * nInputPlanes * 9;
	r = cuMemAlloc(&d_weight, weight_size);
	if (r != CUDA_SUCCESS) {
		puts("fail: alloc weight");
		exit(1);
	}

	r = cuMemcpyHtoD(d_weight, weight, weight_size);
	if (r != CUDA_SUCCESS) {
		puts("fail: copy to weight");
		exit(1);
	}

	size_t nInputPlanes2 = nInputPlanes;
	size_t nOutputPlanes2 = nOutputPlanes;
	size_t h = ip_height;
	size_t w = ip_width;

	void *args[8] = {&packed_input,
			 &nInputPlanes2,
			 &packed_output,
			 &nOutputPlanes2,
			 &d_fbiases,
			 &h,
			 &w,
			 &d_weight};

	r = cuLaunchKernel(dev->filter,
			   h, 1, 1,
			   nOutputPlanes, 1, 1,
			   sizeof(float) * nInputPlanes * (GPU_BLOCK_SIZE+2) * 3,
			   dev->stream, args, NULL);
	if (r != CUDA_SUCCESS) {
		puts("fail: launch");
		exit(1);
	}

	r = cuStreamSynchronize(dev->stream);
	if (r != CUDA_SUCCESS) {
		puts("fail: stream sync");
		exit(1);
	}

	cuMemFree(d_weight);
	cuMemFree(d_fbiases);

	CUcontext old;
	cuCtxPopCurrent(&old);
}


}
