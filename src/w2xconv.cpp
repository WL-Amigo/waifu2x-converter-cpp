#define W2XCONV_IMPL
#define _WIN32_WINNT 0x0600

#define ENABLE_AVX 1

#include <thread>

#ifdef X86OPT
//#if (defined __GNUC__) || (defined __clang__)
#ifndef _WIN32
#include <cpuid.h>
#else
#ifndef HAVE_OPENCV
#include <intrin.h>
#endif
#endif
#endif // X86OPT

#ifdef ARMOPT
#if defined __ANDROID__
#include <cpu-features.h>
#elif __has_include(<sys/auxv.h>)
#include <sys/auxv.h>
#endif
#endif

#ifdef PPCOPT
#if __has_include(<sys/auxv.h>)
#include <sys/auxv.h>
#endif
#endif

#include <limits.h>
#include <sstream>

#include "w2xconv.h"
#include "sec.hpp"
#include "Buffer.hpp"
#include "modelHandler.hpp"
#include "convertRoutine.hpp"
#include "filters.hpp"
#include "cvwrap.hpp"
#include "tstring.hpp"

struct W2XConvImpl
{
	std::string dev_name;

	ComputeEnv env;

	std::vector<std::unique_ptr<w2xc::Model> > noise0_models;
	std::vector<std::unique_ptr<w2xc::Model> > noise1_models;
	std::vector<std::unique_ptr<w2xc::Model> > noise2_models;
	std::vector<std::unique_ptr<w2xc::Model> > noise3_models;
	std::vector<std::unique_ptr<w2xc::Model> > scale2_models;
};

static std::vector<struct W2XConvProcessor> processor_list;

static void global_init2(void)
{
	{
		W2XConvProcessor host;
		host.type = W2XCONV_PROC_HOST;
		host.sub_type = W2XCONV_PROC_HOST_OPENCV;
		host.dev_id = 0;
		host.dev_name = "Generic";
		host.num_core = std::thread::hardware_concurrency();

#ifdef X86OPT
#ifdef _WIN32
#define x_cpuid(p,eax) __cpuid(p, eax)
		typedef int cpuid_t;
#else
#define x_cpuid(p,eax) __get_cpuid(eax, &(p)[0], &(p)[1], &(p)[2], &(p)[3]);
		typedef unsigned int cpuid_t;
#endif
		cpuid_t v[4];
		cpuid_t data[4*3+1];

		x_cpuid(v, 0x80000000);
		if ((unsigned int)v[0] >= 0x80000004)
		{
			x_cpuid(data+4*0, 0x80000002);
			x_cpuid(data+4*1, 0x80000003);
			x_cpuid(data+4*2, 0x80000004);
			data[12] = 0;

			host.dev_name = strdup((char*)data);
		}
		else
		{
			x_cpuid(data, 0x0);
			data[4] = 0;
			host.dev_name = strdup((char*)(data + 1));
		}

		x_cpuid(v, 1);

		if (ENABLE_AVX && (v[2] & 0x18000000) == 0x18000000)
		{
			if (v[2] & (1<<12))
			{
				host.sub_type = W2XCONV_PROC_HOST_FMA;
			}
			else
			{
				host.sub_type = W2XCONV_PROC_HOST_AVX;
			}
		}
		else if (v[2] & (1<<0))
		{
			host.sub_type = W2XCONV_PROC_HOST_SSE3;
		}
#endif // X86OPT

#ifdef ARMOPT
		bool have_neon = false;
#if defined(__ARM_NEON)
// armv8 or -march=armv7-a for all files
		have_neon = true;
#elif defined(__ANDROID__)
		int hwcap = android_getCpuFeatures();
		if (hwcap & ANDROID_CPU_ARM_FEATURE_NEON)
		{
			have_neon = true;
		}
#elif __has_include(<sys/auxv.h>)
		unsigned long hwcap = 0;
#ifdef __FreeBSD__
		elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
#else
		hwcap = getauxval(AT_HWCAP);
#endif
#ifndef HWCAP_ARM_NEON
#ifdef HWCAP_NEON
#define HWCAP_ARM_NEON HWCAP_NEON
#else
#define HWCAP_ARM_NEON (1UL << 12)
#endif
#endif
		if (hwcap & HWCAP_ARM_NEON)
		{
			have_neon = true;
		}
#endif
		if (have_neon)
		{
			host.dev_name = "ARM NEON";
			host.sub_type = W2XCONV_PROC_HOST_NEON;
		}
#endif // ARMOPT

#ifdef PPCOPT
		bool have_altivec = false;
#if defined(__ALTIVEC__)
// powerpc64 or -maltivec for all files
		have_altivec = true;
#elif __has_include(<sys/auxv.h>)
		unsigned long hwcap = 0;
#ifdef __FreeBSD__
		elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
#else
		hwcap = getauxval(AT_HWCAP);
#endif
#ifndef PPC_FEATURE_HAS_ALTIVEC
#define PPC_FEATURE_HAS_ALTIVEC (1UL << 28)
#endif
		if (hwcap & PPC_FEATURE_HAS_ALTIVEC)
		{
			have_altivec = true;
		}
#endif // __has_include(<sys/auxv.h>)
		if (have_altivec)
		{
			host.dev_name = "PowerPC AltiVec";
			host.sub_type = W2XCONV_PROC_HOST_ALTIVEC;
		}
#endif // PPCOPT

		processor_list.push_back(host);
	}

	w2xc::initOpenCLGlobal(&processor_list);
	w2xc::initCUDAGlobal(&processor_list);


	/*
	 * <priority>
	 * 1: NV CUDA
	 * 2: OCL GPU
	 * 3: host AVX
	 * 4: OCL GPU (intel gen)
	 * 5: host
	 * 6: other
	 *
	 * && orderd by num_core
	 */
	std::sort(
		processor_list.begin(),
		processor_list.end(),
		[&](W2XConvProcessor const &p0,W2XConvProcessor const &p1)
		{
			bool p0_is_opencl_gpu = (p0.type == W2XCONV_PROC_OPENCL) && ((p0.sub_type&W2XCONV_PROC_OPENCL_DEVICE_MASK) == W2XCONV_PROC_OPENCL_DEVICE_GPU);

			bool p1_is_opencl_gpu = (p1.type == W2XCONV_PROC_OPENCL) && ((p1.sub_type&W2XCONV_PROC_OPENCL_DEVICE_MASK) == W2XCONV_PROC_OPENCL_DEVICE_GPU);


			bool p0_is_opencl_intel_gpu = (p0.type == W2XCONV_PROC_OPENCL) && (p0.sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU);

			bool p1_is_opencl_intel_gpu = (p1.type == W2XCONV_PROC_OPENCL) && (p1.sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU);

			bool p0_host_avx = (p0.type == W2XCONV_PROC_HOST) && (p0.sub_type >= W2XCONV_PROC_HOST_AVX);

			bool p1_host_avx = (p1.type == W2XCONV_PROC_HOST) && (p1.sub_type >= W2XCONV_PROC_HOST_AVX);

			if (p0.type == p1.type)
			{
				if (p0.type == W2XCONV_PROC_OPENCL)
				{
					if (p0.sub_type != p1.sub_type)
					{
						if (p0_is_opencl_gpu)
						{
							return true;
						}

						if (p1_is_opencl_gpu)
						{
							return false;
						}
					}
				}

				if (p0.num_core != p1.num_core)
				{
					return p0.num_core > p1.num_core;
				}
			}
			else
			{
				if (p0.type == W2XCONV_PROC_CUDA)
				{
					return true;
				}

				if (p1.type == W2XCONV_PROC_CUDA)
				{
					return false;
				}

				if (p0_is_opencl_intel_gpu)
				{
					if (p1_host_avx)
					{
					  return false;
					}
				}

				if (p1_is_opencl_intel_gpu)
				{
					if (p0_host_avx)
					{
					  return false;
					}
				}

				if (p0_is_opencl_gpu)
				{
				return true;
				}

				if (p1_is_opencl_gpu)
				{
					return false;
				}
			}

			if (p0.type == W2XCONV_PROC_HOST)
			{
				return true;
			}

			if (p1.type == W2XCONV_PROC_HOST)
			{
				return false;
			}

			/* ?? */
			return p0.dev_id < p1.dev_id;
		}
	);
}

#ifdef _WIN32
#include <windows.h>
static INIT_ONCE global_init_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK global_init1(PINIT_ONCE initOnce, PVOID Parameter, PVOID *Context)
{
	global_init2();
	return TRUE;
}

static void global_init(void) 
{
	InitOnceExecuteOnce(&global_init_once, global_init1, nullptr, nullptr);
}
#else
#include <pthread.h>

static pthread_once_t global_init_once = PTHREAD_ONCE_INIT;

static void global_init1()
{
	global_init2();
}

static void global_init()
{
	pthread_once(&global_init_once, global_init1);
}

#endif


const struct W2XConvProcessor * w2xconv_get_processor_list(size_t *ret_num)
{
	global_init();

	*ret_num = processor_list.size();
	return &processor_list[0];
}

static int select_device(enum W2XConvGPUMode gpu)
{
	size_t n = processor_list.size();
	if (gpu == W2XCONV_GPU_FORCE_OPENCL)
	{
		for (size_t i=0; i<n; i++)
		{
			if (processor_list[i].type == W2XCONV_PROC_OPENCL)
			{
				return (int) i;
			}
		}
	}

	int host_proc = 0;
	for (int i=0; i<n; i++)
	{
		if (processor_list[i].type == W2XCONV_PROC_HOST)
		{
			host_proc = i;
			break;
		}
	}

	if (gpu == W2XCONV_GPU_AUTO)
	{
		/* 1. CUDA
		 * 2. AMD GPU OpenCL
		 * 3. FMA
		 * 4. AVX
		 * 5. Intel GPU OpenCL
		 */

		for (int i=0; i<n; i++)
		{
			if (processor_list[i].type == W2XCONV_PROC_CUDA)
			{
				return i;
			}
		}

		for (int i=0; i<n; i++)
		{
			if ((processor_list[i].type == W2XCONV_PROC_OPENCL) && (processor_list[i].sub_type == W2XCONV_PROC_OPENCL_AMD_GPU))
			{
				return i;
			}
		}

		if (processor_list[host_proc].sub_type == W2XCONV_PROC_HOST_FMA || processor_list[host_proc].sub_type == W2XCONV_PROC_HOST_AVX)
		{
			return host_proc;
		}

		for (int i=0; i<n; i++)
		{
			if ((processor_list[i].type == W2XCONV_PROC_OPENCL) && (processor_list[i].sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU))
			{
				return i;
			}
		}

		return host_proc;
	}

	/* (gpu == GPU_DISABLE) */
	for (int i=0; i<n; i++)
	{
		if (processor_list[i].type == W2XCONV_PROC_HOST)
		{
			return i;
		}
	}

	return 0; // ??
}

W2XConv * w2xconv_init(enum W2XConvGPUMode gpu, int nJob, int log_level)
{
	return w2xconv_init_with_tta(gpu, nJob, log_level, false);
}

W2XConv * w2xconv_init_with_tta(enum W2XConvGPUMode gpu, int nJob, int log_level, bool tta_mode)
{
	global_init();

	int proc_idx = select_device(gpu);
	return w2xconv_init_with_processor_and_tta(proc_idx, nJob, log_level, tta_mode);
}

struct W2XConv * w2xconv_init_with_processor(int processor_idx, int nJob, int log_level)
{
	return w2xconv_init_with_processor_and_tta(processor_idx, nJob, log_level, false);
}

struct W2XConv * w2xconv_init_with_processor_and_tta(int processor_idx, int nJob, int log_level, bool tta_mode)
{
	global_init();

	struct W2XConv *c = new struct W2XConv;
	struct W2XConvImpl *impl = new W2XConvImpl;
	struct W2XConvProcessor *proc = &processor_list[processor_idx];

	if (nJob == 0)
	{
		nJob = std::thread::hardware_concurrency();
	}

	bool r;

	switch (proc->type)
	{
		case W2XCONV_PROC_CUDA:
		{
			w2xc::initCUDA(&impl->env, proc->dev_id);
			break;
		}
		case W2XCONV_PROC_OPENCL:
		{
			r = w2xc::initOpenCL(c, &impl->env, proc);
			if (!r) {
				return NULL;
			}
			break;
		}
		default: //FutureNote: if PROC_HOST is breaking too.. why not just default: break.. and if aesthetics.. why not case, then default?
		case W2XCONV_PROC_HOST:
			break;
	}

#if defined(_WIN32) || defined(__linux)
	impl->env.tpool = w2xc::initThreadPool(nJob);
#endif

	w2xc::modelUtility::getInstance().setNumberOfJobs(nJob);

	c->impl = impl;
	c->log_level = log_level;
	c->tta_mode = tta_mode;
	c->target_processor = proc;
	c->last_error.code = W2XCONV_NOERROR;
	c->flops.flop = 0;
	c->flops.filter_sec = 0;
	c->flops.process_sec = 0;

	return c;
}

void clearError(W2XConv *conv)
{
	switch (conv->last_error.code)
	{
		case W2XCONV_NOERROR:
		case W2XCONV_ERROR_Y_MODEL_MISMATCH_TO_RGB_F32:
		case W2XCONV_ERROR_WIN32_ERROR:
		case W2XCONV_ERROR_LIBC_ERROR:
		case W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y:
		{
			break;
		}
		case W2XCONV_ERROR_WIN32_ERROR_PATH:
		{
			free(conv->last_error.u.win32_path.path);
			break;
		}
		case W2XCONV_ERROR_LIBC_ERROR_PATH:
		{
			free(conv->last_error.u.libc_path.path);
			break;
		}
		case W2XCONV_ERROR_MODEL_LOAD_FAILED:
		case W2XCONV_ERROR_IMREAD_FAILED:
		case W2XCONV_ERROR_IMWRITE_FAILED:
		{
			free(conv->last_error.u.path);
			break;
		}
		default:
		{
			break;
		}
	}
}

char * w2xconv_strerror(W2XConvError *e)
{
	std::ostringstream oss;
	char *str;

	switch (e->code)
		{
		case W2XCONV_NOERROR:
		{
			oss << "no error";
			break;
		}
		case W2XCONV_ERROR_OPENCL:
		{
			oss << "opencl_err: " << e->u.errno_;
			break;
		}
		case W2XCONV_ERROR_WIN32_ERROR:
		{
			oss << "win32_err: " << e->u.errno_;
			break;
		}
		case W2XCONV_ERROR_WIN32_ERROR_PATH:
		{
			oss << "win32_err: " << e->u.win32_path.errno_ << "(" << e->u.win32_path.path << ")";
			break;
		}
		case W2XCONV_ERROR_LIBC_ERROR:
		{
			oss << strerror(e->u.errno_);
			break;
		}
		case W2XCONV_ERROR_LIBC_ERROR_PATH:
		{
			str = strerror(e->u.libc_path.errno_);
			oss << str << "(" << e->u.libc_path.path << ")";
			break;
		}
		case W2XCONV_ERROR_MODEL_LOAD_FAILED:
		{
			oss << "model load failed: " << e->u.path;
			break;
		}
		case W2XCONV_ERROR_IMREAD_FAILED:
		{
			oss << "cv::imread(\"" << e->u.path << "\") failed";
			break;
		}
		case W2XCONV_ERROR_IMWRITE_FAILED:
		{
			oss << "cv::imwrite(\"" << e->u.path << "\") failed";
			break;
		}
		case W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y:
		{
			oss << "cannot apply rgb model to yuv.";
			break;
		}
		case W2XCONV_ERROR_Y_MODEL_MISMATCH_TO_RGB_F32:
		{
			oss << "cannot apply y model to rgb_f32.";
			break;
		}
		case W2XCONV_ERROR_SCALE_LIMIT:
		{
			oss << "image scale is too big to convert.";
			break;
		}	
		case W2XCONV_ERROR_SIZE_LIMIT:
		{
			oss << "image width (or height) under 40px cannot converted in this scale."; 
			break;
		}	
		case W2XCONV_ERROR_WEBP_SIZE_LIMIT:
		{
			oss << "output size too big for webp format. use png or jpg instead."; 
			break;
		}
		case W2XCONV_ERROR_WEBP_LOSSY_SIZE_LIMIT:
		{
			oss << "output size too big for lossy webp format. use -q 101 for lossless webp instead."; 
			break;
		}
	}

	return strdup(oss.str().c_str());
}

void w2xconv_free(void *p)
{
	free(p);
}

static void setPathError(W2XConv *conv, enum W2XConvErrorCode code, _tstring const &path)
{
	std::string strpath = _tstr2str(path);
	clearError(conv);

	conv->last_error.code = code;
	conv->last_error.u.path = strdup(strpath.c_str());
}

static void setError(W2XConv *conv, enum W2XConvErrorCode code)
{
	clearError(conv);
	conv->last_error.code = code;
}

int w2xconv_load_models(W2XConv *conv, const TCHAR *model_dir)
{
	struct W2XConvImpl *impl = conv->impl;

	_tstring modelFileName(model_dir);

	impl->noise0_models.clear();
	impl->noise1_models.clear();
	impl->noise2_models.clear();
	impl->noise3_models.clear();
	impl->scale2_models.clear();
	
	//FutureNote: Maybe use loop instead of if-spam?
	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + _T("/noise0_model.json"), impl->noise0_models))
	{
		setPathError(conv, W2XCONV_ERROR_MODEL_LOAD_FAILED, modelFileName + _T("/noise0_model.json"));
		return -1;
	}

	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + _T("/noise1_model.json"), impl->noise1_models))
	{
		setPathError(conv, W2XCONV_ERROR_MODEL_LOAD_FAILED, modelFileName + _T("/noise1_model.json"));
		return -1;
	}

	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + _T("/noise2_model.json"), impl->noise2_models))
	{
		setPathError(conv, W2XCONV_ERROR_MODEL_LOAD_FAILED, modelFileName + _T("/noise2_model.json"));
		return -1;
	}

	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + _T("/noise3_model.json"), impl->noise3_models))
	{
		setPathError(conv, W2XCONV_ERROR_MODEL_LOAD_FAILED, modelFileName + _T("/noise3_model.json"));
		return -1;
	}

	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + _T("/scale2.0x_model.json"), impl->scale2_models))
	{
		setPathError(conv, W2XCONV_ERROR_MODEL_LOAD_FAILED, modelFileName + _T("/scale2.0x_model.json"));
		return -1;
	}

	return 0;
}

void w2xconv_set_model_3x3
(
	struct W2XConv *conv,
	enum W2XConvFilterType filter_type,
	int layer_depth,
	int num_input_plane,
	const int *num_map, // num_map[layer_depth]
	const float *coef_list, // coef_list[layer_depth][num_map][3x3]
	const float *bias // bias[layer_depth][num_map]
)
{
	struct W2XConvImpl *impl = conv->impl;
	std::vector<std::unique_ptr<w2xc::Model> > *models = nullptr;

	switch (filter_type)
	{
		case W2XCONV_FILTER_DENOISE0:
		{
			models = &impl->noise0_models;
			break;
		}
		case W2XCONV_FILTER_DENOISE1:
		{
			models = &impl->noise1_models;
			break;
		}
		case W2XCONV_FILTER_DENOISE2:
		{
			models = &impl->noise2_models;
			break;
		}
		case W2XCONV_FILTER_DENOISE3:
		{
			models = &impl->noise3_models;
			break;
		}
		case W2XCONV_FILTER_SCALE2x:
		{
			models = &impl->scale2_models;
			break;
		}
	}

	models->clear();
	
	w2xc::modelUtility::generateModelFromMEM
	(
		layer_depth,
		num_input_plane,
		num_map,
		coef_list,
		bias,
		*models
	);
}

void w2xconv_fini(struct W2XConv *conv)
{
	struct W2XConvImpl *impl = conv->impl;
	clearError(conv);

	w2xc::finiCUDA(&impl->env);
	w2xc::finiOpenCL(&impl->env);
#if defined(_WIN32) || defined(__linux)
	w2xc::finiThreadPool(impl->env.tpool);
#endif

	delete impl;
	delete conv;
}

#ifdef HAVE_OPENCV
static void apply_denoise
(
	struct W2XConv *conv,
	cv::Mat &image,
	int denoise_level,
	int blockSize,
	enum w2xc::image_format fmt
)
{
	struct W2XConvImpl *impl = conv->impl;
	ComputeEnv *env = &impl->env;

	std::vector<cv::Mat> imageSplit;
	cv::Mat *input;
	cv::Mat *output;
	cv::Mat imageY;

	if (IS_3CHANNEL(fmt))
	{
		input = &image;
		output = &image;
	}
	else
	{
		cv::split(image, imageSplit);
		imageSplit[0].copyTo(imageY);
		input = &imageY;
		output = &imageSplit[0];
	}

	W2Mat output_2;
	W2Mat input_2(*input);

	if (denoise_level == 0)
	{
		w2xc::convertWithModels(conv, env, input_2, output_2, impl->noise0_models, &conv->flops, blockSize, fmt, conv->log_level);
	}
	else if (denoise_level == 1)
	{
		w2xc::convertWithModels(conv, env, input_2, output_2, impl->noise1_models, &conv->flops, blockSize, fmt, conv->log_level);
	}
	else if (denoise_level == 2)
	{
		w2xc::convertWithModels(conv, env, input_2, output_2, impl->noise2_models, &conv->flops, blockSize, fmt, conv->log_level);
	}
	else if (denoise_level == 3)
	{
		w2xc::convertWithModels(conv, env, input_2, output_2, impl->noise3_models, &conv->flops, blockSize, fmt, conv->log_level);
	}

	output_2.to_cvmat(output);

	if (! IS_3CHANNEL(fmt))
	{
		cv::merge(imageSplit, image);
	}
}

static void apply_scale
(
	struct W2XConv *conv,
	cv::Mat &image,
	int iterTimesTwiceScaling,
	int blockSize,
	enum w2xc::image_format fmt
)
{
	struct W2XConvImpl *impl = conv->impl;
	ComputeEnv *env = &impl->env;

	// 2x scaling
	for (int nIteration = 0; nIteration < iterTimesTwiceScaling; nIteration++)
	{
		if (conv->log_level >= 3)
		{
			printf("2x Scaling:\n");
		}
		cv::Size imageSize = image.size();
		imageSize.width *= 2;
		imageSize.height *= 2;
		cv::Mat image2xNearest;
		cv::Mat imageY;
		std::vector<cv::Mat> imageSplit;
		cv::Mat image2xBicubic;
		cv::Mat *input, *output;

		cv::resize(image, image2xNearest, imageSize, 0, 0, cv::INTER_NEAREST);

		if (IS_3CHANNEL(fmt))
		{
			input = &image2xNearest;
			output = &image;
		}
		else
		{
			cv::split(image2xNearest, imageSplit);
			imageSplit[0].copyTo(imageY);
			// generate bicubic scaled image and
			// convert RGB -> YUV and split
			imageSplit.clear();
			cv::resize(image,image2xBicubic,imageSize,0,0,cv::INTER_CUBIC);
			cv::split(image2xBicubic, imageSplit);
			input = &imageY;
			output = &imageSplit[0];
		}

		W2Mat output_2;
		W2Mat input_2(*input);

		if(!w2xc::convertWithModels(
			conv,
			env,
			input_2,
			output_2,
			impl->scale2_models,
			&conv->flops, blockSize, fmt,
			conv->log_level
		))
		{
			std::cerr << "w2xc::convertWithModels : something error has occured.\nstop." << std::endl;
			std::exit(1);
		}

		output_2.to_cvmat(output);

		if (!IS_3CHANNEL(fmt))
		{
			cv::merge(imageSplit, image);
		}
	} // 2x scaling : end
}

static inline float clipf(float min, float v, float max)
{
	v = std::max(min,v);
	v = std::min(max,v);

	return v;
}

template <typename SRC_TYPE, int src_max, int ridx, int bidx>
static void preproc_rgb2yuv(cv::Mat *dst, cv::Mat *src)
{
	int w = src->size().width;
	int h = src->size().height;

	float div = 1.0f / src_max;

	for (int yi=0; yi<h; yi++)
	{
		const SRC_TYPE *src_line = (SRC_TYPE*)src->ptr(yi);
		float *dst_line = (float*)dst->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float b = src_line[xi*3 + bidx] * div;
			float g = src_line[xi*3 + 1] * div;
			float r = src_line[xi*3 + ridx] * div;

			float Y = clipf(0.0f, b*0.114f + g*0.587f + r*0.299f, 1.0f);
			float U = clipf(0.0f, (b-Y) * 0.492f + 0.5f,          1.0f);
			float V = clipf(0.0f, (r-Y) * 0.877f + 0.5f,          1.0f);

			dst_line[xi*3 + 0] = Y;
			dst_line[xi*3 + 1] = U;
			dst_line[xi*3 + 2] = V;
		}
	}
}

template <typename SRC_TYPE, int ridx, int bidx>
static bool set_nearest_nontransparent(float *r, float *g, float *b, const SRC_TYPE *s, int xi)
{
	SRC_TYPE a = s[xi*4+3];
	if (a == 0) {
		return false;
	}

	*r = (float)s[xi*4+ridx];
	*g = (float)s[xi*4+1];
	*b = (float)s[xi*4+bidx];

	return true;
}


template <typename SRC_TYPE, int src_max, int ridx, int bidx>
static void preproc_rgba2yuv(cv::Mat *dst_yuv, cv::Mat *dst_alpha, cv::Mat *src, float bkgd_r, float bkgd_g, float bkgd_b)
{
	int w = src->size().width;
	int h = src->size().height;

	float div = 1.0f / src_max;
	float alpha_coef = 1.0f / src_max;

	for (int yi=0; yi<h; yi++)
	{
		const SRC_TYPE *src_line = (SRC_TYPE*)src->ptr(yi);
		const SRC_TYPE *src_line0 = NULL, *src_line2 = NULL;

		if (yi != 0)
		{
			src_line0 = (SRC_TYPE*)src->ptr(yi-1);
		}

		if (yi != h-1)
		{
			src_line2 = (SRC_TYPE*)src->ptr(yi+1);
		}

		float *dst_yuv_line = (float*)dst_yuv->ptr(yi);
		float *dst_alpha_line = (float*)dst_alpha->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float r = src_line[xi*4 + ridx] * div;
			float g = src_line[xi*4 + 1] * div;
			float b = src_line[xi*4 + bidx] * div;
			SRC_TYPE a = src_line[xi*4 + 3];
			if (a == 0)
			{
				r = bkgd_r;
				g = bkgd_g;
				b = bkgd_b;

#if 0
				if (yi == 0 || yi == h-1 || xi == 0 || xi == w-1)
				{
					/* xx */
					r = bkgd_r;
					g = bkgd_g;
					b = bkgd_b;
				}
				else
				{
					/* set nearest non-transparental color */
					SRC_TYPE near_a;
					bool set = false;

					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line0, xi-1);
					}
					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line0, xi);
					}
					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line0, xi+1);
					}

					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line,  xi-1);
					}
					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line,  xi+1);
					}

					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line2, xi-1);
					}
					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line2, xi);
					}
					if (!set)
					{
						set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line2, xi+1);
					}

					if (set)
					{
						r *= div;
						g *= div;
						b *= div;
					}
					else
					{
						r = bkgd_r;
						g = bkgd_g;
						b = bkgd_b;
					}
				}
#endif
			}
			else
			{
				SRC_TYPE ra = src_max - a;
				r = r * (a * alpha_coef) + bkgd_r * (ra * alpha_coef);
				g = g * (a * alpha_coef) + bkgd_g * (ra * alpha_coef);
				b = b * (a * alpha_coef) + bkgd_b * (ra * alpha_coef);
			}
			float Y = clipf(0.0f, b*0.114f + g*0.587f + r*0.299f, 1.0f);
			float U = clipf(0.0f, (b-Y) * 0.492f + 0.5f,          1.0f);
			float V = clipf(0.0f, (r-Y) * 0.877f + 0.5f,          1.0f);

			dst_yuv_line[xi*3 + 0] = Y;
			dst_yuv_line[xi*3 + 1] = U;
			dst_yuv_line[xi*3 + 2] = V;
			dst_alpha_line[xi] = a * div;
		}
	}
}

template <typename SRC_TYPE, int src_max, int ridx, int bidx>
static void preproc_rgb2rgb(cv::Mat *dst, cv::Mat *src)
{
	int w = src->size().width;
	int h = src->size().height;

	float div = 1.0f / src_max;

	for (int yi=0; yi<h; yi++)
	{
		const SRC_TYPE *src_line = (SRC_TYPE*)src->ptr(yi);
		float *dst_line = (float*)dst->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float r = src_line[xi*3 + ridx] * div;
			float g = src_line[xi*3 + 1] * div;
			float b = src_line[xi*3 + bidx] * div;

			dst_line[xi*3 + 0] = r;
			dst_line[xi*3 + 1] = g;
			dst_line[xi*3 + 2] = b;
		}
	}
}
template <typename SRC_TYPE, int src_max, int ridx, int bidx>
static void preproc_rgba2rgb(cv::Mat *dst_rgb,cv::Mat *dst_alpha, cv::Mat *src, float bkgd_r, float bkgd_g, float bkgd_b)
{
	int w = src->size().width;
	int h = src->size().height;

	float div = 1.0f / src_max;
	float alpha_coef = 1.0f / src_max;

	for (int yi=0; yi<h; yi++)
	{
		const SRC_TYPE *src_line = (SRC_TYPE*)src->ptr(yi);
		const SRC_TYPE *src_line0 = NULL, *src_line2 = NULL;

		if (yi != 0)
		{
			src_line0 = (SRC_TYPE*)src->ptr(yi-1);
		}

		if (yi != h-1)
		{
			src_line2 = (SRC_TYPE*)src->ptr(yi+1);
		}

		float *dst_rgb_line = (float*)dst_rgb->ptr(yi);
		float *dst_alpha_line = (float*)dst_alpha->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float r = src_line[xi*4 + ridx] * div;
			float g = src_line[xi*4 + 1] * div;
			float b = src_line[xi*4 + bidx] * div;
			SRC_TYPE a = src_line[xi*4 + 3];
			if (a == 0)
			{
				r = bkgd_r;
				g = bkgd_g;
				b = bkgd_b;
#if 0
				if (yi == 0 || yi == h-1 || xi == 0 || xi == w-1)
				{
					/* xx */
					r = bkgd_r;
					g = bkgd_g;
					b = bkgd_b;
				}
				else
				{
					/* set nearest non-transparental color */
					SRC_TYPE near_a;
					bool set = false;

					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line0, xi-1);
					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line0, xi);
					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line0, xi+1);

					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line,  xi-1);
					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line,  xi+1);

					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line2, xi-1);
					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line2, xi);
					if (!set) set = set_nearest_nontransparent<SRC_TYPE,ridx,bidx>(&r, &g, &b, src_line2, xi+1);

					if (set)
					{
						r *= div;
						g *= div;
						b *= div;
					}
					else
					{
						r = bkgd_r;
						g = bkgd_g;
						b = bkgd_b;
					}
				}
#endif
			}
			else
			{
				SRC_TYPE ra = src_max - a;
				r = r * (a * alpha_coef) + bkgd_r * (ra * alpha_coef);
				g = g * (a * alpha_coef) + bkgd_g * (ra * alpha_coef);
				b = b * (a * alpha_coef) + bkgd_b * (ra * alpha_coef);

				r = std::min(1.0f, r);
				g = std::min(1.0f, g);
				b = std::min(1.0f, b);
			}
			dst_rgb_line[xi*3 + 0] = r;
			dst_rgb_line[xi*3 + 1] = g;
			dst_rgb_line[xi*3 + 2] = b;
			dst_alpha_line[xi] = a * div;
		}
	}
}


template <typename DST_TYPE, int dst_max, int ridx, int bidx>
static void postproc_rgb2rgba(cv::Mat *dst, cv::Mat *src_rgb, cv::Mat *src_alpha, float bkgd_r, float bkgd_g, float bkgd_b)
{
	int w = dst->size().width;
	int h = dst->size().height;

	for (int yi=0; yi<h; yi++)
	{
		const float *src_rgb_line = (float*)src_rgb->ptr(yi);
		const float *src_alpha_line = (float*)src_alpha->ptr(yi);
		DST_TYPE *dst_line = (DST_TYPE*)dst->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float r = src_rgb_line[xi*3 + 0];
			float g = src_rgb_line[xi*3 + 1];
			float b = src_rgb_line[xi*3 + 2];
			float a = src_alpha_line[xi];

			/*       data = src*alpha + bkgd*(1-alpha)    */
			/* -src*alpha = bkgd*(1-alpha) - data         */
			/*  src*alpha = data - bkgd*(1-alpha)         */
			/*  src*alpha = data - bkgd + bkgd * alpha    */
			/*        src = (data - bkgd)/alpha+bkgd      */

			r = (r - bkgd_r)/a + bkgd_r;
			g = (g - bkgd_g)/a + bkgd_g;
			b = (b - bkgd_b)/a + bkgd_b;

			r = clipf(0.0f, r * dst_max, dst_max);
			g = clipf(0.0f, g * dst_max, dst_max);
			b = clipf(0.0f, b * dst_max, dst_max);
			a = clipf(0.0f, a * dst_max, dst_max);

			dst_line[xi*4 + ridx] = (DST_TYPE)r;
			dst_line[xi*4 + 1] = (DST_TYPE)g;
			dst_line[xi*4 + bidx] = (DST_TYPE)b;
			dst_line[xi*4 + 3] = (DST_TYPE)a;
		}
	}
}

template <typename DST_TYPE, int dst_max, int ridx, int bidx>
static void postproc_rgb2rgb(cv::Mat *dst, cv::Mat *src_rgb)
{
	int w = dst->size().width;
	int h = dst->size().height;

	for (int yi=0; yi<h; yi++)
	{
		const float *src_rgb_line = (float*)src_rgb->ptr(yi);
		DST_TYPE *dst_line = (DST_TYPE*)dst->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float r = src_rgb_line[xi*3 + 0];
			float g = src_rgb_line[xi*3 + 1];
			float b = src_rgb_line[xi*3 + 2];

			r = clipf(0.0f, r * dst_max, dst_max);
			g = clipf(0.0f, g * dst_max, dst_max);
			b = clipf(0.0f, b * dst_max, dst_max);

			dst_line[xi*3 + ridx] = (DST_TYPE)r;
			dst_line[xi*3 + 1] = (DST_TYPE)g;
			dst_line[xi*3 + bidx] = (DST_TYPE)b;
		}
	}
}

template <typename DST_TYPE, int dst_max, int ridx, int bidx>
static void postproc_yuv2rgba(cv::Mat *dst, cv::Mat *src_yuv, cv::Mat *src_alpha, float bkgd_r, float bkgd_g, float bkgd_b)
{
	int w = dst->size().width;
	int h = dst->size().height;

	for (int yi=0; yi<h; yi++)
	{
		const float *src_yuv_line = (float*)src_yuv->ptr(yi);
		const float *src_alpha_line = (float*)src_alpha->ptr(yi);
		DST_TYPE *dst_line = (DST_TYPE*)dst->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float a = src_alpha_line[xi];
			float y = src_yuv_line[xi*3 + 0];
			float cr = src_yuv_line[xi*3 + 1];
			float cb = src_yuv_line[xi*3 + 2];
			float C0 = 2.032f, C1 = -0.395f, C2 = -0.581f, C3 = 1.140f;

			float b = y + (cb-0.5f)*C3;
			float g = y + (cb-0.5f)*C2 + (cr-0.5f)*C1;
			float r = y + (cr-0.5f)*C0;

			r = (r - bkgd_r)/a + bkgd_r;
			g = (g - bkgd_g)/a + bkgd_g;
			b = (b - bkgd_b)/a + bkgd_b;

			r = clipf(0.0f, r * dst_max, dst_max);
			g = clipf(0.0f, g * dst_max, dst_max);
			b = clipf(0.0f, b * dst_max, dst_max);
			a = clipf(0.0f, a * dst_max, dst_max);

			dst_line[xi*4 + ridx] = (DST_TYPE)r;
			dst_line[xi*4 + 1] = (DST_TYPE)g;
			dst_line[xi*4 + bidx] = (DST_TYPE)b;
			dst_line[xi*4 + 3] = (DST_TYPE)a;
		}
	}
}

template <typename DST_TYPE, int dst_max, int ridx, int bidx>
static void postproc_yuv2rgb(cv::Mat *dst, cv::Mat *src_yuv)
{
	int w = dst->size().width;
	int h = dst->size().height;

	for (int yi=0; yi<h; yi++)
	{
		const float *src_yuv_line = (float*)src_yuv->ptr(yi);
		DST_TYPE *dst_line = (DST_TYPE*)dst->ptr(yi);

		for (int xi=0; xi<w; xi++)
		{
			float y = src_yuv_line[xi*3 + 0];
			float cr = src_yuv_line[xi*3 + 1];
			float cb = src_yuv_line[xi*3 + 2];

			float C0 = 2.032f, C1 = -0.395f, C2 = -0.581f, C3 = 1.140f;

			float b = y + (cb-0.5f)*C3;
			float g = y + (cb-0.5f)*C2 + (cr-0.5f)*C1;
			float r = y + (cr-0.5f)*C0;

			r = clipf(0.0f, r * dst_max, dst_max);
			g = clipf(0.0f, g * dst_max, dst_max);
			b = clipf(0.0f, b * dst_max, dst_max);

			dst_line[xi*3 + ridx] = (DST_TYPE)r;
			dst_line[xi*3 + 1] = (DST_TYPE)g;
			dst_line[xi*3 + bidx] = (DST_TYPE)b;
		}
	}
}

static int read_int2(FILE *fp)
{
    unsigned int c0 = fgetc(fp);
    unsigned int c1 = fgetc(fp);

    return (c0<<8) | (c1);
}

static unsigned int read_uint4(FILE *fi)
{
	unsigned char oneBytes[4];
	if (fread(oneBytes, 1, 4, fi) == 4)
	{
		return (oneBytes[0]<<24) | (oneBytes[1]<<16) | (oneBytes[2]<<8) | (oneBytes[3]); // unsinged char will be automatically promoted to unsinged int
	}
	return 0;
}

static int read_int4(FILE *fi)
{
	return (int) read_uint4(fi);
}

/*
	Skip one PNG chunk.
	
	Seeks 8Bytes back, reads the chunk length,
	then skips over the already read signature.
	With the aquired chunk_size it skips over chunk_size,
	then over the final part, the crc sum.
*/
void skip_sig(FILE *png_fp, char *sig)
{
	//DEBUG printf("sig(%.4s)\n", sig);
	fseek(png_fp, -8L, SEEK_CUR);
	unsigned int chunk_size = read_uint4(png_fp);
	//DEBUG printf("chunk_size: %u\n", chunk_size);
	fseek(png_fp, 4L, SEEK_CUR);
	fseek(png_fp, chunk_size, SEEK_CUR);
	unsigned int crc = read_int4(png_fp);
	//DEBUG printf("crc: %08X\n",crc);
}

enum PNG_TYPE
{
	Grayscale = 0,
	Truecolor = 2,
	Indexed = 3,
	GrayscaleAlpha = 4,
	TruecolorAlpha = 6,
};

//This checks if the file type is png, it defalts to the user inputted bkgd_colour otherwise.
//The returning bool is whether the function excecuted successfully or not.
void get_png_background_colour(FILE *png_fp, bool *has_alpha, struct w2xconv_rgb_float3 *bkgd_colour)
{
	*has_alpha = false;
	//png file signature
	const static unsigned char sig_png[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
	//byte signature of sub components
	const static unsigned char sig_ihdr[4] = {'I','H','D','R'};
	const static unsigned char sig_iend[4] = {'I','E','N','D'};
	const static unsigned char sig_bkgd[4] = {'b','K','G','D'};
	const static unsigned char sig_trns[4] = {'t','R','N','S'};
	const static unsigned char sig_gama[4] = {'g','A','M','A'};
	const static unsigned char sig_chrm[4] = {'c','H','R','M'};
	const static unsigned char sig_plte[4] = {'P','L','T','E'};
	const static unsigned char sig_phys[4] = {'p','H','Y','s'};
	const static unsigned char sig_time[4] = {'t','I','M','E'};
	const static unsigned char sig_text[4] = {'t','E','X','t'};
	const static unsigned char sig_ztxt[4] = {'z','T','X','t'};
	const static unsigned char sig_itxt[4] = {'i','T','X','t'};
	const static unsigned char sig_hist[4] = {'h','I','S','T'};
	const static unsigned char sig_splt[4] = {'s','P','L','T'};
	const static unsigned char sig_sbit[4] = {'s','B','I','T'};
	const static unsigned char sig_scal[4] = {'s','C','A','L'};
	const static unsigned char sig_offs[4] = {'o','F','F','s'};
	const static unsigned char sig_pcal[4] = {'p','C','A','L'};
	const static unsigned char sig_frac[4] = {'f','R','A','c'};
	const static unsigned char sig_gifg[4] = {'g','I','F','g'};
	const static unsigned char sig_gifx[4] = {'g','I','F','x'};
	const static unsigned char sig_gift[4] = {'g','I','F','t'};
	const static unsigned char sig_idat[4] = {'I','D','A','T'};
	const static unsigned char sig_srgb[4] = {'s','R','G','B'};
	const static unsigned char sig_vpag[4] = {'v','p','A','g'};
	const static unsigned char sig_actl[4] = {'a','c','T','L'};
	const static unsigned char sig_dsig[4] = {'d','S','I','G'};
	const static unsigned char sig_exif[4] = {'e','X','I','f'};
	const static unsigned char sig_iccp[4] = {'i','C','C','P'};
	const static unsigned char sig_ster[4] = {'s','T','E','R'};
	const static unsigned char sig_txmp[4] = {'t','X','M','P'};
	const static unsigned char sig_zxif[4] = {'z','x','I','f'};
	
	//webp file signature
	const static unsigned char sig_riff[4] = {'R','I','F','F'};
	const static unsigned char sig_webp[4] = {'W','E','B','P'};


	const static unsigned char *sig_ignores[] = // array of unused PNG chunks and their signature.
	{
		sig_gama, sig_chrm, sig_plte, sig_phys,
		sig_time, sig_text, sig_ztxt, sig_itxt,
		sig_hist, sig_splt, sig_sbit, sig_scal,
		sig_offs, sig_pcal, sig_frac, sig_gifg,
		sig_gifx, sig_gift, sig_idat, sig_srgb,
		sig_vpag, sig_actl, sig_dsig, sig_exif,
		sig_iccp, sig_ster, sig_txmp, sig_zxif,
		sig_ihdr
	};
	
	const static size_t sig_ignore_size = sizeof(sig_ignores) / sizeof(*sig_ignores);
	
	char sig[8];

	//checks if the file is at least 8 bytes long (png signature)
	size_t rdsz = fread(sig, 1, 8, png_fp);
	if (rdsz != 8)
	{
		//DEBUG printf("sig_png rdsz is not 8, rdsz: %zu, sig: %.8s\n", rdsz, sig);
		return;
	}
	
	//check if file signatures match
	if (memcmp(sig_png, sig, 8) != 0)
	{
		fseek(png_fp, -8L, SEEK_CUR);
		rdsz = fread(sig, 1, 4, png_fp);
		if (rdsz != 4)
		{
			//DEBUG printf("sig_webp rdsz is not 4, rdsz: %zu, sig: %.4s\n", rdsz, sig);
			return;
		}
		if (memcmp(sig_riff, sig, 4) != 0)
		{
			//DEBUG fseek(png_fp, -4L, SEEK_CUR);
			//DEBUG rdsz = fread(sig, 1, 8, png_fp);
			//DEBUG printf("sig_png/sig_webp does not match, sig: %.8s\n", sig);
			return;
		}
		int webp_size = read_int4(png_fp);
		rdsz = fread(sig, 1, 4, png_fp);
		if (rdsz != 4)
		{
			//DEBUG printf("sig_webp rdsz is not 4, rdsz: %zu, sig: %.4s\n", rdsz, sig);
			return;
		}
		if (memcmp(sig_webp, sig, 4) != 0)
		{
			//DEBUG fseek(png_fp, -4L, SEEK_CUR);
			//DEBUG rdsz = fread(sig, 1, 8, png_fp);
			//DEBUG printf("sig_png/sig_webp does not match, sig: %.8s\n", sig);
			return;
		}
		*has_alpha = true;
		return;
	}
	
	//check if ihdr is the required 13 bytes long
	int ihdr_size = read_int4(png_fp);
	if (ihdr_size != 13)
	{
		//DEBUG printf("ihdr_size is not 13, ihdr_size: %d, sig: %.8s\n", ihdr_size, sig);
		return;
	}

	rdsz = fread(sig, 1, 4, png_fp);
	if (rdsz != 4)
	{
		//DEBUG printf("first rdsz is not 4, rdsz: %zu, sig: %.4s\n", rdsz, sig);
		return;
	}
	
	//missing ihdr/invalid png
	if (memcmp(sig_ihdr, sig, 4) != 0)
	{
		//DEBUG printf("missing ihdr/invalid png: %s\n", sig);
		return;
	}
	
	//start of iheader reading
	int width = read_int4(png_fp);
	int height = read_int4(png_fp);
	int depth = fgetc(png_fp);
	int type = fgetc(png_fp);
	int compress = fgetc(png_fp);
	int filter = fgetc(png_fp);
	int interlace = fgetc(png_fp);

	/* use IMREAD_UNCHANGED
	 * if png has alpha channel or is indexed with tRNS chunk.
	 */
	if (type == PNG_TYPE::TruecolorAlpha || type == PNG_TYPE::Indexed || type == PNG_TYPE::GrayscaleAlpha)
	{
		if (depth == 8 || depth == 16 || (depth == 4 && type == PNG_TYPE::Indexed))
		{
			*has_alpha = true;
		}
	}
	else if (depth == 16)
	{
		*has_alpha = true;
	}

	//DEBUG printf("png type: %d, depth: %d, has_alpha: %s", type, depth, *has_alpha ? "true" : "false");

	//end of iheader reading

	if (*has_alpha) {
		if (type == PNG_TYPE::Indexed) // indexed/type 3 png require the tRNS chunk for alpha this will be checked later on.
		{
			*has_alpha = false;
		}
		//read rest of png
		while (true) {
			rdsz = fread(sig, 1, 4, png_fp);
			
			if (rdsz != 4)
			{
				//DEBUG printf("rdsz is not 4 rdsz: %zu, sig: %.4s\n", rdsz, sig);
				break;
			}
			
			// fseek(png_fp, -8L, SEEK_CUR);
			// unsigned int chunk_size = read_uint4(png_fp);
			// fseek(png_fp, 4L, SEEK_CUR);
			
			if (memcmp(sig, sig_iend,4) == 0) //end of PNG
			{
				//DEBUG printf("sig(%.4s)\n", sig);
				break; //end of png
			}
			else if (memcmp(sig, sig_trns,4) == 0) //alpha/tRNS chunk (unimplemented)
			{ 
				*has_alpha = true; // indexed/type 3 png with tRNS alpha chunk
				
				//DEBUG printf("sig(%.4s)\n", sig);
				fseek(png_fp, -8L, SEEK_CUR);
				unsigned int chunk_size = read_uint4(png_fp);
				//DEBUG printf("chunk_size: %u\n", chunk_size);
				fseek(png_fp, 4L, SEEK_CUR);
				
				fseek(png_fp, chunk_size, SEEK_CUR);
				
				unsigned int crc = read_int4(png_fp);
				//DEBUG printf("crc: %08X\n",crc);
			}
			else if (memcmp(sig, sig_bkgd,4) == 0) //background color chunk
			{
				//DEBUG printf("sig(%.4s)\n", sig);
				fseek(png_fp, -8L, SEEK_CUR);
				unsigned int chunk_size = read_uint4(png_fp);
				//DEBUG printf("chunk_size: %u\n", chunk_size);
				fseek(png_fp, 4L, SEEK_CUR);
				
				if (type == PNG_TYPE::Truecolor || type == PNG_TYPE::TruecolorAlpha) 
				{
					float r = (float) read_int2(png_fp);
					float g = (float) read_int2(png_fp);
					float b = (float) read_int2(png_fp);
					if (depth == 8) {
						bkgd_colour->r = r / 255.0f;
						bkgd_colour->g = g / 255.0f;
						bkgd_colour->b = b / 255.0f;
					} else {
						bkgd_colour->r = r / 65535.0f;
						bkgd_colour->g = g / 65535.0f;
						bkgd_colour->b = b / 65535.0f;
					}
					//DEBUG printf("bkgd rgb: %f,%f,%f\n", bkgd_colour->r, bkgd_colour->g, bkgd_colour->b);
					if (chunk_size != 6)
					{
						//DEBUG printf("bkgd chunk is larger than 6: %u\n", chunk_size);
						//possible crash/issue/invalid png maybe check?
					}
					break;
				}
				/* unused
				else if (type == PNG_TYPE::Grayscale || type == PNG_TYPE::GrayscaleAlpha)
				{
					// unsigned int c0 = fgetc(png_fp);
					// unsigned int c1 = fgetc(png_fp);
					// printf("gray: %u, %u\n", c0, c1);
				}
				else if (type == PNG_TYPE::Indexed)
				{
					// palette_index = fgetc(png_fp);
					// printf("palette_index: %d\n", palette_index);
				}
				*/
				else // keep looking for tRNS
				{
					fseek(png_fp, chunk_size, SEEK_CUR);
					unsigned int crc = read_int4(png_fp);
					//DEBUG printf("crc: %08X\n",crc);
				}
			}
			else
			{
				for(int i = 0; i < sig_ignore_size; i++)
				{
					if (memcmp(sig, sig_ignores[i], 4) == 0)
					{
						skip_sig(png_fp, sig);
					}
				}
			}
			// fseek(png_fp, chunk_size, SEEK_CUR);
			// unsigned int crc = read_int4(png_fp);
			// printf("crc: %08X\n",crc);
		}
	}
	return;
}

#define SLICER_PAD_SIZE 12
#define OUTPUT_SIZE_MAX 178700000
#define WEBP_LOSSY_OUTPUT_MAX 196000000
#define WEBP_MAX_WIDTH 16383

void slice_into_pieces(std::vector<cv::Mat> &pieces, const cv::Mat &image, const int max_scale=2)
{
	//char name[70]="";	// for imwrite test
	
	pieces.push_back(image);
	
	while(pieces[0].rows * pieces[0].cols > OUTPUT_SIZE_MAX / max_scale / max_scale)
	{
		int r=pieces[0].rows, c=pieces[0].cols;
		int h_r=r/2, h_c=c/2;
		
		// div in 4 and add padding to input.
		pieces.push_back(pieces[0](cv::Range(0,h_r+SLICER_PAD_SIZE), cv::Range(0,h_c+SLICER_PAD_SIZE)).clone());
		pieces.push_back(pieces[0](cv::Range(0,h_r+SLICER_PAD_SIZE), cv::Range(h_c-SLICER_PAD_SIZE,c)).clone());
		pieces.push_back(pieces[0](cv::Range(h_r-SLICER_PAD_SIZE,r), cv::Range(0,h_c+SLICER_PAD_SIZE)).clone());
		pieces.push_back(pieces[0](cv::Range(h_r-SLICER_PAD_SIZE,r), cv::Range(h_c-SLICER_PAD_SIZE,c)).clone());
		
		// delete piece
		pieces.erase(pieces.begin());
	}
	/*
	for(int i=0; i<pieces.size(); i++)
	{
		
		sprintf(name, "[test] step%d_slice%d_padded.webp", ld, i);
		
		cv::Mat test=pieces[i].clone(), testout = cv::Mat(pieces[i].size(), CV_MAKETYPE(src_depth,3));
		postproc_rgb2rgb<unsigned char, 255, 2, 0>(&testout, &test);
		
		cv::imwrite(name, testout);
	}
	*/
}

void merge_slices(cv::Mat *image, std::vector<cv::Mat> &pieces, const int max_scale=2)
{
	//int j=0; // for imwrite test merge
	//char name[70]="";	// for imwrite test
			
	// combine images
	while (pieces.size() > 1)
	{
		cv::Mat quarter[4], merged[3];
		int cut = (int) (SLICER_PAD_SIZE * max_scale);
		
		//double time_a = getsec(), time_b = 0;
		
		quarter[0]=pieces[0](cv::Range(0, pieces[0].rows - cut), cv::Range(0, pieces[0].cols - cut)).clone();
		quarter[1]=pieces[1](cv::Range(0, pieces[1].rows - cut), cv::Range(cut, pieces[1].cols)).clone();
		quarter[2]=pieces[2](cv::Range(cut, pieces[2].rows), cv::Range(0, pieces[2].cols - cut)).clone();
		quarter[3]=pieces[3](cv::Range(cut, pieces[3].rows), cv::Range(cut, pieces[3].cols)).clone();
		
		pieces.erase(pieces.begin(), pieces.begin()+4);
		
		//printf("merge horizon\n"); 
		hconcat(quarter[0], quarter[1], merged[0]);
		hconcat(quarter[2], quarter[3], merged[1]);
		
		// free memory
		quarter[0].release();
		quarter[1].release();
		quarter[2].release();
		quarter[3].release();
		
		/*printf("imwriting merged image - horizon\n"); 
		sprintf(name, "[test] merge_step%d_block0.webp", ld);
		
		cv::Mat test=merged[0].clone(), testout = cv::Mat(merged[0].size(), CV_MAKETYPE(src_depth,3));
		postproc_rgb2rgb<unsigned char, 255, 2, 0>(&testout, &test);
		
		cv::imwrite(name, testout);
		
		printf("imwriting merged image - vertical\n"); 
		sprintf(name, "[test] merge_step%d_block1.webp", ld);
		
		test=merged[1].clone(), testout = cv::Mat(merged[1].size(), CV_MAKETYPE(src_depth,3));
		postproc_rgb2rgb<unsigned char, 255, 2, 0>(&testout, &test);
		
		cv::imwrite(name, testout);*/
		
		//printf("merge vertical\n"); 
		vconcat(merged[0], merged[1], merged[2]);
		
		//time_b = getsec();
		//printf("took %f\n", time_b - time_a); 
		
		pieces.push_back(merged[2].clone());
		
		/*
		printf("imwriting merged image\n"); 
		sprintf(name, "[test] merge_step%d_merged.webp", ld);
		
		test=merged[2].clone(), testout = cv::Mat(merged[2].size(), CV_MAKETYPE(src_depth,3));
		postproc_rgb2rgb<unsigned char, 255, 2, 0>(&testout, &test);
		
		cv::imwrite(name, testout);*/
	}
	*image = pieces[0].clone();
}

void w2xconv_convert_mat
(
	struct W2XConv *conv,
	cv::Mat* image_dst, 
	cv::Mat* image_src, 
	int denoise_level, 
	double scale, 
	int blockSize,
	w2xconv_rgb_float3 background,
	bool has_alpha,
	bool dst_alpha
)
{				
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);
	enum w2xc::image_format fmt;
	//char name[70]="";	// for imwrite test

	int src_depth = CV_MAT_DEPTH(image_src->type());
	int src_cn = CV_MAT_CN(image_src->type());
	cv::Mat image = cv::Mat(image_src->size(), CV_32FC3);
	cv::Mat alpha;

	if (is_rgb)
	{
		if (has_alpha)
		{
			if (src_cn == 4)
			{
				// save alpha
				alpha = cv::Mat(image_src->size(), CV_32FC1);
				if (src_depth == CV_16U)
				{
					preproc_rgba2rgb<unsigned short, 65535, 2, 0>(&image, &alpha, image_src, background.r, background.g, background.b);
				}
				else
				{
					preproc_rgba2rgb<unsigned char, 255, 2, 0>(&image, &alpha, image_src, background.r, background.g, background.b);
				}
			}
			else
			{
				preproc_rgb2rgb<unsigned short, 65535, 2, 0>(&image, image_src);
			}
		}
		else
		{
			preproc_rgb2rgb<unsigned char, 255, 2, 0>(&image, image_src);
		}
		fmt = w2xc::IMAGE_RGB_F32;
	}
	else
	{
		if (has_alpha)
		{
			if (src_cn == 4)
			{
				// save alpha
				alpha = cv::Mat(image_src->size(), CV_32FC1);
				if (src_depth == CV_16U)
				{
					preproc_rgba2yuv<unsigned short, 65535, 2, 0>(&image, &alpha, image_src, background.r, background.g, background.b);
				}
				else
				{
					preproc_rgba2yuv<unsigned char, 255, 2, 0>(&image, &alpha, image_src, background.r, background.g, background.b);
				}
			}
			else
			{
				preproc_rgb2yuv<unsigned short, 65535, 2, 0>(&image, image_src);
			}
		}
		else
		{
			preproc_rgb2yuv<unsigned char, 255, 2, 0>(&image, image_src);
		}

		fmt = w2xc::IMAGE_Y;
	}

	image_src->release();
	
	int w2x_total_steps = 0;
	int w2x_current_step = 1;
	int iterTimesTwiceScaling = 0;
	
	if (scale > 1.0)
	{
		iterTimesTwiceScaling = static_cast<int>(std::ceil(std::log2(scale)));
		w2x_total_steps += iterTimesTwiceScaling;
	}
	
	if (denoise_level != -1)
	{
		// divide images in to 4^n pieces when output size is too big.
		std::vector<cv::Mat> pieces;
		
		if (conv->log_level >= 2)
		{
			printf("\nStep %02d/%02d: Denoising\n", w2x_current_step++, ++w2x_total_steps);
		}
			
		slice_into_pieces(pieces, image, 1);
		
		for(int i=0; i<pieces.size(); i++)
		{
			if (conv->log_level >= 2)
			{
				printf("Proccessing [%d/%zu] slices\n", i+1, pieces.size());
			}
			
			if(conv->tta_mode)
			{
				cv::Mat tta_mat;
				for(int ti=0; ti<8; ti++)
				{
					cv::Mat tmp=pieces[i].clone();
					
					if (conv->log_level >= 2)
					{
						printf("Working on TTA mode... step%d/8\n", ti+1);
					}
					
					for(int tj=0; tj < ti%4; tj++)
					{
						cv::transpose(tmp, tmp);
						cv::flip(tmp, tmp, 1);
					}
					
					if(ti >= 4)
					{
						cv::flip(tmp, tmp, 1);
					}
					
					apply_denoise(conv, tmp, denoise_level, blockSize, fmt);
					
					if(ti >= 4)
					{
						cv::flip(tmp, tmp, 1);
					}
					
					for(int tj=0; tj < ti%4; tj++)
					{
						cv::transpose(tmp, tmp);
						cv::flip(tmp, tmp, 0);
					}
					
					if(ti==0)
					{
						tta_mat=tmp.clone();
					}
					else
					{
						tta_mat+=tmp;
					}
				}
				tta_mat /= 8.0;
				pieces[i] = tta_mat.clone();
			}
			else
			{
				apply_denoise(conv, pieces[i], denoise_level, blockSize, fmt);
			}
		}
		
		if (pieces.size() > 1 && conv->log_level >= 2)
		{
			printf("Merging slices back to one image... in queue: %zu slices\n", pieces.size());
		}
		merge_slices(&image, pieces, 1);
	}

	if (scale != 1.0)
	{
		// calculate iteration times of 2x scaling and shrink ratio which will use at last
		double shrinkRatio = 0.0;
		if (static_cast<int>(scale) != std::pow(2, iterTimesTwiceScaling))
		{
			shrinkRatio = scale / std::pow(2.0, static_cast<double>(iterTimesTwiceScaling));
		}
		
		for(int ld = 0; ld < iterTimesTwiceScaling ; ld++)
		{
			// divide images in to 4^n pieces when output size is too big.
			std::vector<cv::Mat> pieces;
			
			if (conv->log_level >= 2)
			{
				printf("\nStep %02d/%02d: 2x Scaling\n", w2x_current_step++, w2x_total_steps);
			}
			
			slice_into_pieces(pieces, image);
			
			for(int i=0; i<pieces.size(); i++)
			{
				if (conv->log_level >= 2)
				{
					printf("Proccessing [%d/%zu] slices\n", i+1, pieces.size());
				}
				
				if(conv->tta_mode)
				{
					cv::Mat tta_mat;
					for(int ti=0; ti<8; ti++)
					{
						cv::Mat tmp=pieces[i].clone();
						
						if (conv->log_level >= 2)
						{
							printf("Working on TTA mode... step%d/8\n", ti+1);
						}
						
						for(int tj=0; tj < ti%4; tj++)
						{
							cv::transpose(tmp, tmp);
							cv::flip(tmp, tmp, 1);
						}
						
						if(ti >= 4)
						{
							cv::flip(tmp, tmp, 1);
						}
						
						apply_scale(conv, tmp, 1, blockSize, fmt);
						
						if(ti >= 4)
						{
							cv::flip(tmp, tmp, 1);
						}
						
						for(int j=0; j < ti%4; j++)
						{
							cv::transpose(tmp, tmp);
							cv::flip(tmp, tmp, 0);
						}
						
						if(ti==0)
						{
							tta_mat=tmp.clone();
						}
						else
						{
							tta_mat+=tmp;
						}
					}
					tta_mat /= 8.0;
					pieces[i] = tta_mat.clone();
				}
				else
				{
					apply_scale(conv, pieces[i], 1, blockSize, fmt);
				}
				
				/*
				sprintf(name, "[test] step%d_slice%d_converted.webp", ld, i);
				
				test=pieces[i].clone(), testout = cv::Mat(pieces[i].size(), CV_MAKETYPE(src_depth,3));
				postproc_rgb2rgb<unsigned char, 255, 2, 0>(&testout, &test);
				
				cv::imwrite(name, testout);*/
			}

			if (pieces.size() > 1 && conv->log_level >= 2)
			{
				printf("Merging slices back to one image... in queue: %zu slices\n", pieces.size());
			}
			merge_slices(&image, pieces);
		}

		if (shrinkRatio != 0.0)
		{
			cv::Size lastImageSize = image.size();
			lastImageSize.width = static_cast<int>(static_cast<double>(lastImageSize.width * shrinkRatio));
			lastImageSize.height = static_cast<int>(static_cast<double>(lastImageSize.height * shrinkRatio));
			cv::resize(image, image, lastImageSize, 0, 0, cv::INTER_LINEAR);
		}
	}

	if (alpha.empty() || !dst_alpha)
	{
		*image_dst = cv::Mat(image.size(), CV_MAKETYPE(src_depth,3));

		if (is_rgb)
		{
			if (src_depth == CV_16U)
			{
				postproc_rgb2rgb<unsigned short, 65535, 2, 0>(image_dst, &image);
			}
			else
			{
				postproc_rgb2rgb<unsigned char, 255, 2, 0>(image_dst, &image);
			}
		}
		else
		{
			if (src_depth == CV_16U)
			{
				postproc_yuv2rgb<unsigned short, 65535, 0, 2>(image_dst, &image);
			}
			else
			{
				postproc_yuv2rgb<unsigned char, 255, 0, 2>(image_dst, &image);
			}
		}
	}
	else
	{
		*image_dst = cv::Mat(image.size(), CV_MAKETYPE(src_depth,4));

		if (image.size() != alpha.size())
		{
			cv::resize(alpha, alpha, image.size(), 0, 0, cv::INTER_LINEAR);
		}

		if (is_rgb)
		{
			if (src_depth == CV_16U)
			{
				postproc_rgb2rgba<unsigned short, 65535, 2, 0>(image_dst, &image, &alpha, background.r, background.g, background.b);
			}
			else
			{
				postproc_rgb2rgba<unsigned char, 255, 2, 0>(image_dst, &image, &alpha, background.r, background.g, background.b);
			}
		}
		else
		{
			if (src_depth == CV_16U)
			{
				postproc_yuv2rgba<unsigned short, 65535, 0, 2>(image_dst, &image, &alpha, background.r, background.g, background.b);
			}
			else
			{
				postproc_yuv2rgba<unsigned char, 255, 0, 2>(image_dst, &image, &alpha, background.r, background.g, background.b);
			}
		}
	}
	
	/*printf("imwriting final_conv_mat image\n"); 
	sprintf(name, "[test] final_conv_mat.webp");
	
	cv::imwrite(name, *image_dst);*/
}

#if defined(_WIN32) && defined(_UNICODE)
	cv::Mat read_imageW(const WCHAR* filepath, int flags=cv::IMREAD_COLOR)
	{
		long lSize;
		char* imgBuffer;
		FILE* pFile = _wfopen(filepath, L"rb");
		cv::Mat image;
		
		if (!pFile)
		{
			image = cv::Mat::zeros(1, 1, CV_8U);
		}
		
		fseek(pFile, 0, SEEK_END);
		lSize = ftell(pFile);
		fseek(pFile, 0, SEEK_SET);
		
		imgBuffer = new char[lSize];
		
		if(!imgBuffer)
		{
			image = cv::Mat::zeros(1, 1, CV_8U);
		}
		
		fread(imgBuffer, 1, lSize, pFile);
		
		fclose(pFile);
		
		cv::_InputArray arr(imgBuffer, lSize);
		image = cv::imdecode(arr, flags);
		
		delete[] imgBuffer;
		
		return image.clone();
	}

	bool write_imageW(const WCHAR* filepath, cv::Mat& img, std::vector<int>& imwrite_params)
	{
		FILE* pFile;
		std::vector<uchar> imageBuffer;
		std::wstring ext_w = std::wstring(filepath);
		ext_w = ext_w.substr(ext_w.find_last_of(L'.'));
		std::string ext=wstr2str(ext_w);
		
		if(!cv::imencode(ext.c_str(),img, imageBuffer, imwrite_params))
		{
			return false;
		}
		
		pFile = _wfopen(filepath, L"wb+");
		if (!pFile)
		{
			return false;
		}
		
		fwrite(imageBuffer.data(), sizeof(unsigned char), imageBuffer.size(), pFile);
		
		fclose(pFile);
		return true;
	}

	#define read_image read_imageW
	#define write_image write_imageW

#else
	#define read_image cv::imread
	#define write_image cv::imwrite
#endif

int w2xconv_convert_file
(
	struct W2XConv *conv,
	const TCHAR *dst_path,
	const TCHAR *src_path,
	int denoise_level,
	double scale,
	int blockSize,
	int* imwrite_params
)
{
	double time_start = getsec();

	FILE *png_fp = nullptr;
	
	png_fp = _tfopen(src_path, _T("rb"));

	if (png_fp == nullptr)
	{
		setPathError(conv, W2XCONV_ERROR_IMREAD_FAILED, src_path);
		return -1;
	}

	bool has_alpha;
	//Background colour
	//float3 background(1.0f, 1.0f, 1.0f);
	w2xconv_rgb_float3 background;
	background.r = background.g = background.b = 1.0f;
	get_png_background_colour(png_fp, &has_alpha, &background);

	if (png_fp)
	{
		fclose(png_fp);
		png_fp = nullptr;
	}

	cv::Mat image_src, image_dst;

	/*
	 * IMREAD_COLOR                 : always BGR
	 * IMREAD_UNCHANGED + png       : BGR or BGRA
	 * IMREAD_UNCHANGED + otherwise : ???
	 */
	
	if (has_alpha)
	{
		image_src = read_image(src_path, cv::IMREAD_UNCHANGED);
	}
	else
	{
		image_src = read_image(src_path, cv::IMREAD_COLOR);
	}
	
	bool dst_alpha = false;
	bool dst_webp = false;
	{
		size_t len = _tcslen(dst_path);
		if (len >= 5) {
			if (_totlower(dst_path[len-5]) == _T('.') && _totlower(dst_path[len-4]) == _T('w') && _totlower(dst_path[len-3]) == _T('e') && _totlower(dst_path[len-2]) == _T('b') && _totlower(dst_path[len-1]) == _T('p'))
			{
				dst_webp=true;
				dst_alpha=true;
			}
		}
		if (len >= 3) {
			if (_totlower(dst_path[len-4]) == _T('.') && _totlower(dst_path[len-3]) == _T('p') && _totlower(dst_path[len-2]) == _T('n') && _totlower(dst_path[len-1]) == _T('g'))
			{
				dst_alpha = true;
			}
		}
	}

	// w2x converts 2x and down scales when scale_ratio is not power of 2 (ex: 2.28 -> scale x4 - > down scale)
	int max_scale = static_cast<int>(std::pow(2, std::ceil(std::log2(scale))));
	
	//printf("max_scale: %d\n", max_scale);
	
	// comment is for slicer function
	// output file pixel above 178,756,920px is limit. leave 56,920px for safe conversion. see issue #156
	// all images that needs slices, it will require 20 px padding to 2 edges (input should w > 40, h > 40).
	// with max_scale is 2, it only can converts less then (w+20) x (h+20) = 44,675,000 px.
	// with max_scale is 4, it only can converts less then (w+20) x (h+20) = 11,168,750 px.
	// with max_scale is 8, it only can converts less then (w+20) x (h+20) = 2,792,187 px.
	// with max_scale is 16, it only can converts less then (w+20) x (h+20) = 698,046 px.
	// with max_scale is 32, it only can converts less then (w+20) x (h+20) = 174,511 px.
	// with max_scale is 64, it only can converts less then (w+20) x (h+20) = 3,627 px.
	// with max_scale is 128, it only can converts less then (w+20) x (h+20) = 10,906 px.
	// with max_scale is 256, it only can converts less then (w+20) x (h+20) = 2,726 px.
	// with max_scale is 512, it only can converts less then (w+20) x (h+20) = 681 px. padding is all most eat everything (pieces are under 6px)
	// with max_scale is 1024, it only can converts less then (w+20) x (h+20) = 170 px, padding exceed limit (20 x 20 = 400).
	// with max_scale is 2048, it only can converts less then (w+20) x (h+20) = 42 px, which is no meaning to run w2x.
	// with max_scale is 4096, you cannot convert it at all.
	
	if (max_scale > 1 && image_src.rows * image_src.cols > OUTPUT_SIZE_MAX / max_scale / max_scale)
	{
		if (max_scale >= 512)
		{
			setError(conv, W2XCONV_ERROR_SCALE_LIMIT);
			return -1;
		}
	}
	
	// for webp limit
	if(dst_webp && (image_src.rows > WEBP_MAX_WIDTH / scale || image_src.cols > WEBP_MAX_WIDTH / scale)){
		setError(conv, W2XCONV_ERROR_WEBP_SIZE_LIMIT);
		return -1;
	}
	else if (dst_webp && imwrite_params[2] <= 100 && scale > 1.0 && image_src.rows * image_src.cols > WEBP_LOSSY_OUTPUT_MAX / scale / scale){
		setError(conv, W2XCONV_ERROR_WEBP_LOSSY_SIZE_LIMIT);
		return -1;
	}
	
	if (conv->log_level >= 2 && scale > 1.0 )
	{
		printf("Scaling image from %dx%d to %dx%d\n", image_src.cols, image_src.rows, (int) (image_src.cols * scale), (int) (image_src.rows * scale));
	}
	
	w2xconv_convert_mat(conv, &image_dst, &image_src, denoise_level, scale, blockSize, background, has_alpha, dst_alpha);
	
	if (conv->log_level >= 2)
	{
		printf("Writing image to file...\n\n");
	}
	
	std::vector<int> vec_imwrite_params;
	for (int i = 0; i < 6; i++)
	{
		vec_imwrite_params.push_back(imwrite_params[i]);
	}
	
	if (!write_image(dst_path, image_dst, vec_imwrite_params))
	{
		setPathError(conv, W2XCONV_ERROR_IMWRITE_FAILED, dst_path);
		return -1;
	}

	double time_end = getsec();

	conv->flops.process_sec += time_end - time_start;

	//printf("== %f == \n", conv->impl->env.transfer_wait);

	return 0;
}


static void convert_mat
(
	struct W2XConv *conv,
	cv::Mat &image,
	int denoise_level,
	double scale,
	int dst_w, int dst_h,
	int blockSize,
	enum w2xc::image_format fmt
)
{
	int w2x_total_steps = 0;
	int w2x_current_step = 1;
	int iterTimesTwiceScaling;
	
	if (scale != 1.0)
	{
		iterTimesTwiceScaling = static_cast<int>(std::ceil(std::log2(scale)));
		w2x_total_steps = w2x_total_steps + iterTimesTwiceScaling;
	}
	if (denoise_level != -1)
	{
		if (conv->log_level >= 2)
		{
			printf("Step %02d/%02d: Denoising\n", w2x_current_step++, ++w2x_total_steps);
		}
		apply_denoise(conv, image, denoise_level, blockSize, fmt);
	}

	if (scale != 1.0)
	{
		// calculate iteration times of 2x scaling and shrink ratio which will use at last
		double shrinkRatio = 0.0;
		if (static_cast<int>(scale) != std::pow(2, iterTimesTwiceScaling))
		{
			shrinkRatio = scale / std::pow(2.0, static_cast<double>(iterTimesTwiceScaling));
		}
		apply_scale(conv, image, iterTimesTwiceScaling, blockSize, fmt);

		if (shrinkRatio != 0.0)
		{
			cv::Size lastImageSize = image.size();
			lastImageSize.width = dst_w;
			lastImageSize.height = dst_h;
			cv::resize(image, image, lastImageSize, 0, 0, cv::INTER_LINEAR);
		}
	}
}


int w2xconv_convert_rgb
(
	struct W2XConv *conv,
	unsigned char *dst, size_t dst_step_byte, /* rgb24 (src_w*ratio, src_h*ratio) */
	unsigned char *src, size_t src_step_byte, /* rgb24 (src_w, src_h) */
	int src_w, int src_h,
	int denoise_level, /* 0:none, 1:L1 denoise, other:L2 denoise  */
	double scale,
	int block_size
)
{
	int dst_h = (int) (src_h * scale);
	int dst_w = (int) (src_w * scale);

	cv::Mat srci(src_h, src_w, CV_8UC3, src, src_step_byte);
	cv::Mat dsti(dst_h, dst_w, CV_8UC3, dst, dst_step_byte);
	cv::Mat image;

	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	if (is_rgb)
	{
		srci.copyTo(image);
		convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_RGB);
		image.copyTo(dsti);
	}
	else
	{
		srci.convertTo(image, CV_32F, 1.0 / 255.0);
		cv::cvtColor(image, image, cv::COLOR_RGB2YUV);
		convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_Y);

		cv::cvtColor(image, image, cv::COLOR_YUV2RGB);
		image.convertTo(dsti, CV_8U, 255.0);
	}

	return 0;
}

int w2xconv_convert_rgb_f32
(
	struct W2XConv *conv,
	unsigned char *dst, size_t dst_step_byte, /* rgb float32x3 normalized[0-1] (src_w*ratio, src_h*ratio) */
	unsigned char *src, size_t src_step_byte, /* rgb float32x3 normalized[0-1] (src_w, src_h) */
	int src_w, int src_h,
	int denoise_level, /* 0:none, 1:L1 denoise, other:L2 denoise  */
	double scale,
	int block_size
)
{
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	if (!is_rgb)
	{
		setError(conv, W2XCONV_ERROR_Y_MODEL_MISMATCH_TO_RGB_F32);
		return -1;
	}

	int dst_h = (int) (src_h * scale);
	int dst_w = (int) (src_w * scale);

	cv::Mat srci(src_h, src_w, CV_32FC3, src, src_step_byte);
	cv::Mat dsti(dst_h, dst_w, CV_32FC3, dst, dst_step_byte);
	cv::Mat image;

	srci.copyTo(image);
	convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_RGB_F32);
	image.copyTo(dsti);

	return 0;
}


int w2xconv_convert_yuv
(
	struct W2XConv *conv,
	unsigned char *dst, size_t dst_step_byte, /* float32x3 (src_w*ratio, src_h*ratio) */
	unsigned char *src, size_t src_step_byte, /* float32x3 (src_w, src_h) */
	int src_w, int src_h,
	int denoise_level, /* 0:none, 1:L1 denoise, other:L2 denoise  */
	double scale,
	int block_size
)
{
	int dst_h = (int) (src_h * scale);
	int dst_w = (int) (src_w * scale);

	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	if (is_rgb)
	{
		setError(conv, W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y);
		return -1;
	}

	cv::Mat srci(src_h, src_w, CV_32FC3, src, src_step_byte);
	cv::Mat dsti(dst_h, dst_w, CV_32FC3, dst, dst_step_byte);
	cv::Mat image = srci.clone();

	convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_Y);

	image.copyTo(dsti);

	return 0;
}
#endif // HAVE_OPENCV

int w2xconv_apply_filter_y
(
	struct W2XConv *conv,
	enum W2XConvFilterType type,
	unsigned char *dst, size_t dst_step_byte, /* float32x1 (src_w, src_h) */
	unsigned char *src, size_t src_step_byte, /* float32x1 (src_w, src_h) */
	int src_w, int src_h,
	int blockSize
)
{
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	if (is_rgb)
	{
		setError(conv, W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y);
		return -1;
	}

	struct W2XConvImpl *impl = conv->impl;
	ComputeEnv *env = &impl->env;

	W2Mat dsti(src_w, src_h, CV_32FC1, dst, (int) dst_step_byte);
	W2Mat srci(src_w, src_h, CV_32FC1, src, (int) src_step_byte);

	std::vector<std::unique_ptr<w2xc::Model> > *mp = NULL;

	switch (type)
	{
		case W2XCONV_FILTER_DENOISE0:
		{
			mp = &impl->noise0_models;
			break;
		}	
		case W2XCONV_FILTER_DENOISE1:
		{
			mp = &impl->noise1_models;
			break;
		}
		case W2XCONV_FILTER_DENOISE2:
		{
			mp = &impl->noise2_models;
			break;
		}
		case W2XCONV_FILTER_DENOISE3:
		{
			mp = &impl->noise3_models;
			break;
		}
		case W2XCONV_FILTER_SCALE2x:
		{
			mp = &impl->scale2_models;
			break;
		}
		default:
		{
			return -1;
		}
	}

	W2Mat result;
	w2xc::convertWithModels(conv, env, srci, result, *mp, &conv->flops, blockSize, w2xc::IMAGE_Y, conv->log_level);

	for (int yi=0; yi<src_h; yi++)
	{
		char *d0 = dsti.ptr<char>(yi);
		char *s0 = result.ptr<char>(yi);
		memcpy(d0, s0, src_w * sizeof(float));
	}
	return 0;
}

#ifdef HAVE_OPENCV
int w2xconv_test(struct W2XConv *conv, int block_size)
{
	int w = 200;
	int h = 100;
	int r;

	cv::Mat src_rgb = cv::Mat::zeros(h, w, CV_8UC3);
	cv::Mat dst = cv::Mat::zeros(h, w, CV_8UC3);

	cv::line
	(
		src_rgb,
		cv::Point(10, 10),
		cv::Point(20, 20),
		cv::Scalar(255,0,0),
		8
	);

	cv::line
	(
		src_rgb,
		cv::Point(20, 10),
		cv::Point(10, 20),
		cv::Scalar(0,255,0),
		8
	);

	cv::line
	(
		src_rgb,
		cv::Point(50, 30),
		cv::Point(10, 30),
		cv::Scalar(0,0,255), 
		1
	);

	cv::line
	(
		src_rgb,
		cv::Point(50, 80),
		cv::Point(10, 80),
		cv::Scalar(255,255,255),
		3
	);

	cv::Mat src_32fc3;
	cv::Mat src_yuv;

	src_rgb.convertTo(src_32fc3, CV_32F, 1.0 / 255.0);
	cv::cvtColor(src_32fc3, src_yuv, cv::COLOR_RGB2YUV);

	cv::Mat dst_rgb_x2(h*2, w*2, CV_8UC3);
	cv::Mat dst_rgb_f32_x2(h*2, w*2, CV_32FC3);
	cv::Mat dst_yuv_x2(h*2, w*2, CV_32FC3);

	cv::imwrite("test_src.png", src_rgb);

	w2xconv_convert_rgb
	(
		conv,
		dst_rgb_x2.data,
		dst_rgb_x2.step[0],
		src_rgb.data,
		src_rgb.step[0],
		w,
		h,
		1,
		2.0,
		block_size
	);

	cv::imwrite("test_rgb.png", dst_rgb_x2);


	w2xconv_convert_rgb_f32
	(
		conv,
		dst_rgb_f32_x2.data,
		dst_rgb_f32_x2.step[0],
		src_32fc3.data,
		src_32fc3.step[0],
		w,
		h,
		1,
		2.0,
		block_size
	);

	dst_rgb_f32_x2.convertTo(dst_rgb_x2, CV_8U, 255.0);
	cv::imwrite("test_rgb_f32.png", dst_rgb_x2);

	r = w2xconv_convert_yuv
	(
		conv,
		dst_yuv_x2.data,
		dst_yuv_x2.step[0],
		src_yuv.data,
		src_yuv.step[0],
		w,
		h,
		1,
		2.0,
		block_size
	);

	if (r < 0)
	{
		char *e = w2xconv_strerror(&conv->last_error);
		puts(e);
		w2xconv_free(e);
	}
	else
	{
		cv::cvtColor(dst_yuv_x2, dst_yuv_x2, cv::COLOR_YUV2RGB);
		dst_yuv_x2.convertTo(dst_rgb_x2, CV_8U, 255.0);
		cv::imwrite("test_yuv.png", dst_rgb_x2);
	}

	std::vector<cv::Mat> imageSplit;
	cv::split(src_yuv, imageSplit);
	cv::Mat split_src = imageSplit[0].clone();
	cv::Mat split_dst, dst_rgb;
	cv::Mat split_dsty(h, w, CV_32F);

	r = w2xconv_apply_filter_y
	(
		conv,
		W2XCONV_FILTER_DENOISE1,
		split_dsty.data,
		split_dsty.step[0],
		split_src.data,
		split_src.step[0],
		w, 
		h,
		block_size
	);

	if (r < 0)
	{
		char *e = w2xconv_strerror(&conv->last_error);
		puts(e);
		w2xconv_free(e);
	}
	else
	{
		imageSplit[0] = split_dsty.clone();
		cv::merge(imageSplit, split_dst);
		cv::cvtColor(split_dst, split_dst, cv::COLOR_YUV2RGB);
		split_dst.convertTo(dst_rgb, CV_8U, 255.0);
		cv::imwrite("test_apply.png", dst_rgb);
	}

	return 0;
}
#endif
