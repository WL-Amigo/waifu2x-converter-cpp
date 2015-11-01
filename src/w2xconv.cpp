#define W2XCONV_IMPL
#define _WIN32_WINNT 0x0600

#include <thread>

#ifdef X86OPT
//#if (defined __GNUC__) || (defined __clang__)
#ifndef _WIN32
#include <cpuid.h>
#endif
#endif // X86OPT

#ifdef ARMOPT
#if defined __ANDROID__
#include <cpu-features.h>
#elif (defined(__linux))
#include <sys/auxv.h>
#endif
#endif

#include <sstream>
#include "w2xconv.h"
#include "sec.hpp"
#include "Buffer.hpp"
#include "modelHandler.hpp"
#include "convertRoutine.hpp"
#include "filters.hpp"
#include "cvwrap.hpp"

struct W2XConvImpl {
	std::string dev_name;

	ComputeEnv env;

	std::vector<std::unique_ptr<w2xc::Model> > noise1_models;
	std::vector<std::unique_ptr<w2xc::Model> > noise2_models;
	std::vector<std::unique_ptr<w2xc::Model> > scale2_models;
};

static std::vector<struct W2XConvProcessor> processor_list;

static void
global_init2(void)
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
		if ((unsigned int)v[0] >= 0x80000004) {
			x_cpuid(data+4*0, 0x80000002);
			x_cpuid(data+4*1, 0x80000003);
			x_cpuid(data+4*2, 0x80000004);
			data[12] = 0;

			host.dev_name = strdup((char*)data);
		} else {
			x_cpuid(data, 0x0);
			data[4] = 0;
			host.dev_name = strdup((char*)(data + 1));
		}

		x_cpuid(v, 1);

		if ((v[2] & 0x18000000) == 0x18000000) {
			if (v[2] & (1<<12)) {
				host.sub_type = W2XCONV_PROC_HOST_FMA;
			} else {
				host.sub_type = W2XCONV_PROC_HOST_AVX;
			}
		} else if (v[2] & (1<<0)) {
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
		if (hwcap & ANDROID_CPU_ARM_FEATURE_NEON) {
			have_neon = true;
		}
#elif defined(__linux)
		int hwcap = getauxval(AT_HWCAP);
		if (hwcap & HWCAP_ARM_NEON) {
			have_neon = true;
		}
#endif
		if (have_neon) {
			host.dev_name = "ARM NEON";
			host.sub_type = W2XCONV_PROC_HOST_NEON;
		}
#endif

		processor_list.push_back(host);
	}

	w2xc::initOpenCLGlobal(&processor_list);
	w2xc::initCUDAGlobal(&processor_list);


	/*
	 * 1: NV CUDA
	 * 2: OCL GPU
	 * 3: host
	 * 4: other
	 *
	 * && orderd by num_core
	 */
	std::sort(processor_list.begin(),
		  processor_list.end(),
		  [&](W2XConvProcessor const &p0,
		      W2XConvProcessor const &p1)
		  {
			  bool p0_is_opencl_gpu =
				  (p0.type == W2XCONV_PROC_OPENCL) &&
				  ((p0.sub_type&W2XCONV_PROC_OPENCL_DEVICE_MASK) == W2XCONV_PROC_OPENCL_DEVICE_GPU)
				  ;

			  bool p1_is_opencl_gpu =
				  (p1.type == W2XCONV_PROC_OPENCL) &&
				  ((p1.sub_type&W2XCONV_PROC_OPENCL_DEVICE_MASK) == W2XCONV_PROC_OPENCL_DEVICE_GPU)
				  ;


			  bool p0_is_opencl_intel_gpu =
				  (p0.type == W2XCONV_PROC_OPENCL) &&
				  (p0.sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU)
				  ;

			  bool p1_is_opencl_intel_gpu =
				  (p1.type == W2XCONV_PROC_OPENCL) &&
				  (p1.sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU)
				  ;

			  bool p0_host_avx =
				  (p0.type == W2XCONV_PROC_HOST) &&
				  (p0.sub_type >= W2XCONV_PROC_HOST_AVX)
				  ;

			  bool p1_host_avx =
				  (p1.type == W2XCONV_PROC_HOST) &&
				  (p1.sub_type >= W2XCONV_PROC_HOST_AVX)
				  ;

			  if (p0.type == p1.type) {
				  if (p0.type == W2XCONV_PROC_OPENCL) {
					  if (p0.sub_type != p1.sub_type) {
						  if (p0_is_opencl_gpu) {
							  return true;
						  }

						  if (p1_is_opencl_gpu) {
							  return false;
						  }
					  }
				  }

				  if (p0.num_core != p1.num_core) {
					  return p0.num_core > p1.num_core;
				  }
			  } else {
				  if (p0.type == W2XCONV_PROC_CUDA) {
					  return true;
				  }

				  if (p1.type == W2XCONV_PROC_CUDA) {
					  return false;
				  }

				  if (p0_is_opencl_intel_gpu) {
					  if (p1_host_avx) {
						  return false;
					  }
				  }

				  if (p1_is_opencl_intel_gpu) {
					  if (p0_host_avx) {
						  return false;
					  }
				  }

				  if (p0_is_opencl_gpu) {
					  return true;
				  }

				  if (p1_is_opencl_gpu) {
					  return false;
				  }
			  }

			  if (p0.type == W2XCONV_PROC_HOST) {
				  return true;
			  }

			  if (p1.type == W2XCONV_PROC_HOST) {
				  return false;
			  }

			  /* ?? */
			  return p0.dev_id < p1.dev_id;
		  });
}

#ifdef _WIN32
#include <windows.h>
static INIT_ONCE global_init_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK
global_init1(PINIT_ONCE initOnce,
	     PVOID Parameter,
	     PVOID *Context)
{
	global_init2();
	return TRUE;
}

static void
global_init(void)
{
	InitOnceExecuteOnce(&global_init_once,
			    global_init1,
			    nullptr, nullptr);
}
#else

#include <pthread.h>

static pthread_once_t global_init_once = PTHREAD_ONCE_INIT;
static void
global_init1()
{
	global_init2();
}
static void
global_init()
{
	pthread_once(&global_init_once, global_init1);
}
#endif


const struct W2XConvProcessor *
w2xconv_get_processor_list(int *ret_num)
{
	global_init();

	*ret_num = processor_list.size();
	return &processor_list[0];
}

static int
select_device(enum W2XConvGPUMode gpu)
{
	int n = processor_list.size();
	if (gpu == W2XCONV_GPU_FORCE_OPENCL) {
		for (int i=0; i<n; i++) {
			if (processor_list[i].type == W2XCONV_PROC_OPENCL) {
				return i;
			}
		}
	}

	int host_proc = 0;
	for (int i=0; i<n; i++) {
		if (processor_list[i].type == W2XCONV_PROC_HOST) {
			host_proc = i;
			break;
		}
	}

	if (gpu == W2XCONV_GPU_AUTO) {
		/* 1. CUDA
		 * 2. AMD GPU OpenCL
		 * 3. FMA
		 * 4. AVX
		 * 5. Intel GPU OpenCL
		 */

		for (int i=0; i<n; i++) {
			if (processor_list[i].type == W2XCONV_PROC_CUDA) {
				return i;
			}
		}

		for (int i=0; i<n; i++) {
			if ((processor_list[i].type == W2XCONV_PROC_OPENCL) &&
			    (processor_list[i].sub_type == W2XCONV_PROC_OPENCL_AMD_GPU))
			{
				return i;
			}
		}

		if (processor_list[host_proc].sub_type == W2XCONV_PROC_HOST_FMA ||
		    processor_list[host_proc].sub_type == W2XCONV_PROC_HOST_AVX)
		{
			return host_proc;
		}

		for (int i=0; i<n; i++) {
			if ((processor_list[i].type == W2XCONV_PROC_OPENCL) &&
			    (processor_list[i].sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU))
			{
				return i;
			}
		}

		return host_proc;
	}

	/* (gpu == GPU_DISABLE) */
	for (int i=0; i<n; i++) {
		if (processor_list[i].type == W2XCONV_PROC_HOST) {
			return i;
		}
	}

	return 0;		// ??
}

W2XConv *
w2xconv_init(enum W2XConvGPUMode gpu,
             int nJob,
	     int enable_log)
{
	global_init();

	int proc_idx = select_device(gpu);
	return w2xconv_init_with_processor(proc_idx, nJob, enable_log);
}

struct W2XConv *
w2xconv_init_with_processor(int processor_idx,
			    int nJob,
			    int enable_log)
{
	global_init();

	struct W2XConv *c = new struct W2XConv;
	struct W2XConvImpl *impl = new W2XConvImpl;
	struct W2XConvProcessor *proc = &processor_list[processor_idx];

	if (nJob == 0) {
		nJob = std::thread::hardware_concurrency();
	}

	bool r;

	switch (proc->type) {
	case W2XCONV_PROC_CUDA:
		w2xc::initCUDA(&impl->env, proc->dev_id);
		break;

	case W2XCONV_PROC_OPENCL:
		r = w2xc::initOpenCL(c, &impl->env, proc);
		if (!r) {
			return NULL;
		}
		break;

	default:
	case W2XCONV_PROC_HOST:
		break;
	}

#if defined(_WIN32) || defined(__linux)
	impl->env.tpool = w2xc::initThreadPool(nJob);
#endif

	w2xc::modelUtility::getInstance().setNumberOfJobs(nJob);

	c->impl = impl;
	c->enable_log = enable_log;
	c->target_processor = proc;
	c->last_error.code = W2XCONV_NOERROR;
	c->flops.flop = 0;
	c->flops.filter_sec = 0;
	c->flops.process_sec = 0;

	return c;
}

void
clearError(W2XConv *conv)
{
	switch (conv->last_error.code) {
	case W2XCONV_NOERROR:
	case W2XCONV_ERROR_Y_MODEL_MISMATCH_TO_RGB_F32:
	case W2XCONV_ERROR_WIN32_ERROR:
	case W2XCONV_ERROR_LIBC_ERROR:
	case W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y:
		break;

	case W2XCONV_ERROR_WIN32_ERROR_PATH:
		free(conv->last_error.u.win32_path.path);
		break;

	case W2XCONV_ERROR_LIBC_ERROR_PATH:
		free(conv->last_error.u.libc_path.path);
		break;

	case W2XCONV_ERROR_MODEL_LOAD_FAILED:
	case W2XCONV_ERROR_IMREAD_FAILED:
	case W2XCONV_ERROR_IMWRITE_FAILED:
		free(conv->last_error.u.path);
		break;
	}
}

char *
w2xconv_strerror(W2XConvError *e)
{
	std::ostringstream oss;
	char *str;

	switch (e->code) {
	case W2XCONV_NOERROR:
		oss << "no error";
		break;

	case W2XCONV_ERROR_WIN32_ERROR:
		oss << "win32_err: " << e->u.errno_;
		break;

	case W2XCONV_ERROR_WIN32_ERROR_PATH:
		oss << "win32_err: " << e->u.win32_path.errno_ << "(" << e->u.win32_path.path << ")";
		break;

	case W2XCONV_ERROR_LIBC_ERROR:
		oss << strerror(e->u.errno_);
		break;

	case W2XCONV_ERROR_LIBC_ERROR_PATH:
		str = strerror(e->u.libc_path.errno_);
		oss << str << "(" << e->u.libc_path.path << ")";
		break;

	case W2XCONV_ERROR_MODEL_LOAD_FAILED:
		oss << "model load failed: " << e->u.path;
		break;

	case W2XCONV_ERROR_IMREAD_FAILED:
		oss << "cv::imread(\"" << e->u.path << "\") failed";
		break;

	case W2XCONV_ERROR_IMWRITE_FAILED:
		oss << "cv::imwrite(\"" << e->u.path << "\") failed";
		break;

	case W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y:
		oss << "cannot apply rgb model to yuv.";
		break;

	case W2XCONV_ERROR_Y_MODEL_MISMATCH_TO_RGB_F32:
		oss << "cannot apply y model to rgb_f32.";
		break;
	}

	return strdup(oss.str().c_str());
}

void
w2xconv_free(void *p)
{
	free(p);
}


static void
setPathError(W2XConv *conv,
             enum W2XConvErrorCode code,
             std::string const &path)
{
	clearError(conv);

	conv->last_error.code = code;
	conv->last_error.u.path = strdup(path.c_str());
}

static void
setError(W2XConv *conv,
	 enum W2XConvErrorCode code)
{
	clearError(conv);
	conv->last_error.code = code;
}


int
w2xconv_load_models(W2XConv *conv, const char *model_dir)
{
	struct W2XConvImpl *impl = conv->impl;

	std::string modelFileName(model_dir);

	impl->noise1_models.clear();
	impl->noise2_models.clear();
	impl->scale2_models.clear();

	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + "/noise1_model.json", impl->noise1_models)) {
		setPathError(conv,
			     W2XCONV_ERROR_MODEL_LOAD_FAILED,
			     modelFileName + "/noise1_model.json");
		return -1;
	}
	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + "/noise2_model.json", impl->noise2_models)) {
		setPathError(conv,
			     W2XCONV_ERROR_MODEL_LOAD_FAILED,
			     modelFileName + "/noise2_model.json");
		return -1;
	}
	if (!w2xc::modelUtility::generateModelFromJSON(modelFileName + "/scale2.0x_model.json", impl->scale2_models)) {
		setPathError(conv,
			     W2XCONV_ERROR_MODEL_LOAD_FAILED,
			     modelFileName + "/scale2.0x_model.json");
		return -1;

	}

	return 0;
}

void
w2xconv_set_model_3x3(struct W2XConv *conv,
		      enum W2XConvFilterType m,
		      int layer_depth,
		      int num_input_plane,
		      const int *num_map, // num_map[layer_depth]
		      const float *coef_list, // coef_list[layer_depth][num_map][3x3]
		      const float *bias // bias[layer_depth][num_map]
	)
{
	struct W2XConvImpl *impl = conv->impl;
	std::vector<std::unique_ptr<w2xc::Model> > *models = nullptr;

	switch (m) {
	case W2XCONV_FILTER_DENOISE1:
		models = &impl->noise1_models;
		break;
	case W2XCONV_FILTER_DENOISE2:
		models = &impl->noise2_models;
		break;
	case W2XCONV_FILTER_SCALE2x:
		models = &impl->scale2_models;
		break;
	}

	models->clear();
	w2xc::modelUtility::generateModelFromMEM(layer_depth,
						 num_input_plane,
						 num_map,
						 coef_list,
						 bias,
						 *models);
}

void
w2xconv_fini(struct W2XConv *conv)
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
static void
apply_denoise(struct W2XConv *conv,
	      cv::Mat &image,
	      int denoise_level,
	      int blockSize,
	      enum w2xc::image_format fmt)
{
	struct W2XConvImpl *impl = conv->impl;
	ComputeEnv *env = &impl->env;

	std::vector<cv::Mat> imageSplit;
	cv::Mat *input;
	cv::Mat *output;
	cv::Mat imageY;

	if (IS_3CHANNEL(fmt)) {
		input = &image;
		output = &image;
	} else {
		cv::split(image, imageSplit);
		imageSplit[0].copyTo(imageY);
		input = &imageY;
		output = &imageSplit[0];
	}

	W2Mat output_2;
	W2Mat input_2(extract_view_from_cvmat(*input));

	if (denoise_level == 1) {
		w2xc::convertWithModels(conv, env, input_2, output_2,
					impl->noise1_models,
					&conv->flops, blockSize, fmt, conv->enable_log);
	} else {
		w2xc::convertWithModels(conv, env, input_2, output_2,
					impl->noise2_models,
					&conv->flops, blockSize, fmt, conv->enable_log);
	}

	*output = copy_to_cvmat(output_2);

	if (! IS_3CHANNEL(fmt)) {
		cv::merge(imageSplit, image);
	}
}

static void
apply_scale(struct W2XConv *conv,
	    cv::Mat &image,
	    int iterTimesTwiceScaling,
	    int blockSize,
	    enum w2xc::image_format fmt)
{
	struct W2XConvImpl *impl = conv->impl;
	ComputeEnv *env = &impl->env;

	if (conv->enable_log) {
		std::cout << "start scaling" << std::endl;
	}

	// 2x scaling
	for (int nIteration = 0; nIteration < iterTimesTwiceScaling;
	     nIteration++) {

		if (conv->enable_log) {
			std::cout << "#" << std::to_string(nIteration + 1)
				  << " 2x scaling..." << std::endl;
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

		if (IS_3CHANNEL(fmt)) {
			input = &image2xNearest;
			output = &image;
		} else {
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
		W2Mat input_2(extract_view_from_cvmat(*input));

		if(!w2xc::convertWithModels(conv,
					    env,
					    input_2,
					    output_2,
					    impl->scale2_models,
					    &conv->flops, blockSize, fmt,
					    conv->enable_log))
		{
			std::cerr << "w2xc::convertWithModels : something error has occured.\n"
				"stop." << std::endl;
			std::exit(1);
		}

		*output = copy_to_cvmat(output_2);

		if (!IS_3CHANNEL(fmt)) {
			cv::merge(imageSplit, image);
		}

	} // 2x scaling : end
}

int
w2xconv_convert_file(struct W2XConv *conv,
		     const char *dst_path,
                     const char *src_path,
                     int denoise_level,
                     double scale,
		     int blockSize)
{
	double time_start = getsec();
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	cv::Mat image = cv::imread(src_path, cv::IMREAD_COLOR);
	enum w2xc::image_format fmt;

	if (image.data == nullptr) {
		setPathError(conv,
			     W2XCONV_ERROR_IMREAD_FAILED,
			     src_path);
		return -1;
	}

	if (is_rgb) {
		fmt = w2xc::IMAGE_BGR;
	} else {
		image.convertTo(image, CV_32F, 1.0 / 255.0);
		cv::cvtColor(image, image, cv::COLOR_RGB2YUV);
		fmt = w2xc::IMAGE_Y;
	}

	if (denoise_level != 0) {
		apply_denoise(conv, image, denoise_level, blockSize, fmt);
	}

	if (scale != 1.0) {
		// calculate iteration times of 2x scaling and shrink ratio which will use at last
		int iterTimesTwiceScaling = static_cast<int>(std::ceil(std::log2(scale)));
		double shrinkRatio = 0.0;
		if (static_cast<int>(scale)
		    != std::pow(2, iterTimesTwiceScaling))
		{
			shrinkRatio = scale / std::pow(2.0, static_cast<double>(iterTimesTwiceScaling));
		}

		apply_scale(conv, image, iterTimesTwiceScaling, blockSize, fmt);

		if (shrinkRatio != 0.0) {
			cv::Size lastImageSize = image.size();
			lastImageSize.width =
				static_cast<int>(static_cast<double>(lastImageSize.width
								     * shrinkRatio));
			lastImageSize.height =
				static_cast<int>(static_cast<double>(lastImageSize.height
								     * shrinkRatio));
			cv::resize(image, image, lastImageSize, 0, 0, cv::INTER_LINEAR);
		}
	}

	if (is_rgb) {
	} else {
		cv::cvtColor(image, image, cv::COLOR_YUV2RGB);
		image.convertTo(image, CV_8U, 255.0);
	}

	if (!cv::imwrite(dst_path, image)) {
		setPathError(conv,
			     W2XCONV_ERROR_IMWRITE_FAILED,
			     dst_path);
		return -1;
	}

	double time_end = getsec();

	conv->flops.process_sec += time_end - time_start;

	//printf("== %f == \n", conv->impl->env.transfer_wait);

	return 0;
}

static void
convert_mat(struct W2XConv *conv,
	    cv::Mat &image,
	    int denoise_level,
	    double scale,
	    int dst_w, int dst_h,
	    int blockSize,
	    enum w2xc::image_format fmt)
{
	if (denoise_level != 0) {
		apply_denoise(conv, image, denoise_level, blockSize, fmt);
	}

	if (scale != 1.0) {
		// calculate iteration times of 2x scaling and shrink ratio which will use at last
		int iterTimesTwiceScaling = static_cast<int>(std::ceil(std::log2(scale)));
		double shrinkRatio = 0.0;
		if (static_cast<int>(scale)
		    != std::pow(2, iterTimesTwiceScaling))
		{
			shrinkRatio = scale / std::pow(2.0, static_cast<double>(iterTimesTwiceScaling));
		}

		apply_scale(conv, image, iterTimesTwiceScaling, blockSize, fmt);

		if (shrinkRatio != 0.0) {
			cv::Size lastImageSize = image.size();
			lastImageSize.width = dst_w;
			lastImageSize.height = dst_h;
			cv::resize(image, image, lastImageSize, 0, 0, cv::INTER_LINEAR);
		}
	}
}


int
w2xconv_convert_rgb(struct W2XConv *conv,
		    unsigned char *dst, size_t dst_step_byte, /* rgb24 (src_w*ratio, src_h*ratio) */
		    unsigned char *src, size_t src_step_byte, /* rgb24 (src_w, src_h) */
		    int src_w, int src_h,
		    int denoise_level, /* 0:none, 1:L1 denoise, other:L2 denoise  */
		    double scale,
		    int block_size)
{
	int dst_h = src_h * scale;
	int dst_w = src_w * scale;

	cv::Mat srci(src_h, src_w, CV_8UC3, src, src_step_byte);
	cv::Mat dsti(dst_h, dst_w, CV_8UC3, dst, dst_step_byte);

	cv::Mat image;
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	if (is_rgb) {
		srci.copyTo(image);
		convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_RGB);
		image.copyTo(dsti);
	} else {
		srci.convertTo(image, CV_32F, 1.0 / 255.0);
		cv::cvtColor(image, image, cv::COLOR_RGB2YUV);
		convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_Y);

		cv::cvtColor(image, image, cv::COLOR_YUV2RGB);
		image.convertTo(dsti, CV_8U, 255.0);
	}

	return 0;
}

int
w2xconv_convert_rgb_f32(struct W2XConv *conv,
			unsigned char *dst, size_t dst_step_byte, /* rgb float32x3 normalized[0-1] (src_w*ratio, src_h*ratio) */
			unsigned char *src, size_t src_step_byte, /* rgb float32x3 normalized[0-1] (src_w, src_h) */
			int src_w, int src_h,
			int denoise_level, /* 0:none, 1:L1 denoise, other:L2 denoise  */
			double scale,
			int block_size)
{
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);

	if (!is_rgb) {
		setError(conv, W2XCONV_ERROR_Y_MODEL_MISMATCH_TO_RGB_F32);
		return -1;
	}

	int dst_h = src_h * scale;
	int dst_w = src_w * scale;

	cv::Mat srci(src_h, src_w, CV_32FC3, src, src_step_byte);
	cv::Mat dsti(dst_h, dst_w, CV_32FC3, dst, dst_step_byte);

	cv::Mat image;

	srci.copyTo(image);
	convert_mat(conv, image, denoise_level, scale, dst_w, dst_h, block_size, w2xc::IMAGE_RGB_F32);
	image.copyTo(dsti);

	return 0;
}


int
w2xconv_convert_yuv(struct W2XConv *conv,
		    unsigned char *dst, size_t dst_step_byte, /* float32x3 (src_w*ratio, src_h*ratio) */
		    unsigned char *src, size_t src_step_byte, /* float32x3 (src_w, src_h) */
		    int src_w, int src_h,
		    int denoise_level, /* 0:none, 1:L1 denoise, other:L2 denoise  */
		    double scale,
		    int block_size)
{
	int dst_h = src_h * scale;
	int dst_w = src_w * scale;

	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);
	if (is_rgb) {
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

int
w2xconv_apply_filter_y(struct W2XConv *conv,
		       enum W2XConvFilterType type,
		       unsigned char *dst, size_t dst_step_byte, /* float32x1 (src_w, src_h) */
		       unsigned char *src, size_t src_step_byte, /* float32x1 (src_w, src_h) */
		       int src_w, int src_h,
		       int blockSize)
{
	bool is_rgb = (conv->impl->scale2_models[0]->getNInputPlanes() == 3);
	if (is_rgb) {
		setError(conv, W2XCONV_ERROR_RGB_MODEL_MISMATCH_TO_Y);
		return -1;
	}

	struct W2XConvImpl *impl = conv->impl;
	ComputeEnv *env = &impl->env;

	W2Mat dsti(src_h, src_w, CV_32FC1, dst, dst_step_byte);
	W2Mat srci(src_h, src_w, CV_32FC1, src, src_step_byte);

	std::vector<std::unique_ptr<w2xc::Model> > *mp = NULL;

	switch (type) {
	case W2XCONV_FILTER_DENOISE1:
		mp = &impl->noise1_models;
		break;

	case W2XCONV_FILTER_DENOISE2:
		mp = &impl->noise2_models;
		break;

	case W2XCONV_FILTER_SCALE2x:
		mp = &impl->scale2_models;
		break;

	default:
		return -1;
	}

	W2Mat result;
	w2xc::convertWithModels(conv, env, srci, result,
				*mp,
				&conv->flops, blockSize, w2xc::IMAGE_Y, conv->enable_log);

	for (int yi=0; yi<src_h; yi++) {
		char *d0 = dsti.ptr<char>(yi);
		char *s0 = srci.ptr<char>(yi);

		memcpy(d0, s0, src_w * sizeof(float));
	}

	return 0;
}

#ifdef HAVE_OPENCV
int
w2xconv_test(struct W2XConv *conv, int block_size)
{
	int w = 200;
	int h = 100;
	int r;

	cv::Mat src_rgb = cv::Mat::zeros(h, w, CV_8UC3);
	cv::Mat dst = cv::Mat::zeros(h, w, CV_8UC3);

	cv::line(src_rgb,
		 cv::Point(10, 10),
		 cv::Point(20, 20),
		 cv::Scalar(255,0,0), 8);

	cv::line(src_rgb,
		 cv::Point(20, 10),
		 cv::Point(10, 20),
		 cv::Scalar(0,255,0), 8);

	cv::line(src_rgb,
		 cv::Point(50, 30),
		 cv::Point(10, 30),
		 cv::Scalar(0,0,255), 1);

	cv::line(src_rgb,
		 cv::Point(50, 80),
		 cv::Point(10, 80),
		 cv::Scalar(255,255,255), 3);

	cv::Mat src_32fc3;
	cv::Mat src_yuv;

	src_rgb.convertTo(src_32fc3, CV_32F, 1.0 / 255.0);
	cv::cvtColor(src_32fc3, src_yuv, cv::COLOR_RGB2YUV);

	cv::Mat dst_rgb_x2(h*2, w*2, CV_8UC3);
	cv::Mat dst_rgb_f32_x2(h*2, w*2, CV_32FC3);
	cv::Mat dst_yuv_x2(h*2, w*2, CV_32FC3);

	cv::imwrite("test_src.png", src_rgb);

	w2xconv_convert_rgb(conv,
			    dst_rgb_x2.data, dst_rgb_x2.step[0],
			    src_rgb.data, src_rgb.step[0],
			    w, h,
			    1,
			    2.0,
			    block_size);

	cv::imwrite("test_rgb.png", dst_rgb_x2);


	w2xconv_convert_rgb_f32(conv,
				dst_rgb_f32_x2.data, dst_rgb_f32_x2.step[0],
				src_32fc3.data, src_32fc3.step[0],
				w, h,
				1,
				2.0,
				block_size);

	dst_rgb_f32_x2.convertTo(dst_rgb_x2, CV_8U, 255.0);
	cv::imwrite("test_rgb_f32.png", dst_rgb_x2);

	r = w2xconv_convert_yuv(conv,
				dst_yuv_x2.data, dst_yuv_x2.step[0],
				src_yuv.data, src_yuv.step[0],
				w, h,
				1,
				2.0,
				block_size);

	if (r < 0) {
		char *e = w2xconv_strerror(&conv->last_error);
		puts(e);
		w2xconv_free(e);
	} else {
		cv::cvtColor(dst_yuv_x2, dst_yuv_x2, cv::COLOR_YUV2RGB);
		dst_yuv_x2.convertTo(dst_rgb_x2, CV_8U, 255.0);
		cv::imwrite("test_yuv.png", dst_rgb_x2);
	}

	std::vector<cv::Mat> imageSplit;
	cv::split(src_yuv, imageSplit);
	cv::Mat split_src = imageSplit[0].clone();
	cv::Mat split_dst, dst_rgb;
	cv::Mat split_dsty(h, w, CV_32F);

	r = w2xconv_apply_filter_y(conv,
				   W2XCONV_FILTER_DENOISE1,
				   split_dsty.data, split_dsty.step[0],
				   split_src.data, split_src.step[0],
				   w, h, block_size);
	if (r < 0) {
		char *e = w2xconv_strerror(&conv->last_error);
		puts(e);
		w2xconv_free(e);
	} else {
		imageSplit[0] = split_dsty.clone();

		cv::merge(imageSplit, split_dst);

		cv::cvtColor(split_dst, split_dst, cv::COLOR_YUV2RGB);
		split_dst.convertTo(dst_rgb, CV_8U, 255.0);

		cv::imwrite("test_apply.png", dst_rgb);
	}

	return 0;
}
#endif
