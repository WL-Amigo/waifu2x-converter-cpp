#define W2XCONV_IMPL

#include <thread>
#ifdef X86OPT
//#if (defined __GNUC__) || (defined __clang__)
#ifndef _WIN32
#include <cpuid.h>
#endif
#endif // X86OPT
#include "w2xconv.h"
#include "sec.hpp"
#include "Buffer.hpp"
#include "modelHandler.hpp"
#include "convertRoutine.hpp"

struct W2XConvImpl {
	std::string dev_name;

	ComputeEnv env;

	std::vector<std::unique_ptr<w2xc::Model> > noise1_models;
	std::vector<std::unique_ptr<w2xc::Model> > noise2_models;
	std::vector<std::unique_ptr<w2xc::Model> > scale2_models;
};

W2XConv *
w2xconv_init(enum W2XConvGPUMode gpu,
             int nJob,
	     int enable_log)
{
	struct W2XConv *c = new struct W2XConv;
	struct W2XConvImpl *impl = new W2XConvImpl;

	c->impl = impl;
	c->enable_log = enable_log;

	if (nJob == 0) {
		nJob = std::thread::hardware_concurrency();
	}

#if defined(_WIN32) || defined(__linux)
	impl->env.tpool = w2xc::initThreadPool(nJob);
#endif

	if (gpu == W2XCONV_GPU_DISABLE) {
		/* disable */
	} else {
		w2xc::initOpenCL(&impl->env, gpu);
		if (gpu != W2XCONV_GPU_FORCE_OPENCL) {
			w2xc::initCUDA(&impl->env);
		}
	}

	c->last_error.code = W2XCONV_NOERROR;
	c->flops.flop = 0;
	c->flops.filter_sec = 0;
	c->flops.process_sec = 0;

	if (impl->env.num_cuda_dev != 0) {
		c->target_processor.type = W2XCONV_PROC_CUDA;
		c->target_processor.devid = 0;
		impl->dev_name = impl->env.cuda_dev_list[0].name.c_str();
	} else if (impl->env.num_cl_dev != 0) {
		c->target_processor.type = W2XCONV_PROC_OPENCL;
		c->target_processor.devid = 0;
		impl->dev_name = impl->env.cl_dev_list[0].name.c_str();
	} else {
#ifdef X86OPT
		{

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

				impl->dev_name = (char*)data;
			} else {
				x_cpuid(data, 0x0);
				data[4] = 0;
				impl->dev_name = (char*)(data + 1);
			}

			c->target_processor.type = W2XCONV_PROC_HOST;
		}
#endif // X86OPT
	}

	c->target_processor.dev_name = impl->dev_name.c_str();
	impl->env.target_processor = c->target_processor;

	return c;
}

static void
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

	if (denoise_level == 1) {
		w2xc::convertWithModels(env, *input, *output,
					impl->noise1_models,
					&conv->flops, blockSize, fmt, conv->enable_log);
	} else {
		w2xc::convertWithModels(env, *input, *output,
					impl->noise2_models,
					&conv->flops, blockSize, fmt, conv->enable_log);
	}

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

		if(!w2xc::convertWithModels(env,
					    *input,
					    *output,
					    impl->scale2_models,
					    &conv->flops, blockSize, fmt,
					    conv->enable_log))
		{
			std::cerr << "w2xc::convertWithModels : something error has occured.\n"
				"stop." << std::endl;
			std::exit(1);
		}

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

	cv::Mat dsti(src_h, src_w, CV_32F, dst, dst_step_byte);
	cv::Mat srci(src_h, src_w, CV_32F, src, src_step_byte);

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

	cv::Mat result;
	w2xc::convertWithModels(env, srci, result,
				*mp,
				&conv->flops, blockSize, w2xc::IMAGE_Y, conv->enable_log);

	result.copyTo(dsti);

	return 0;
}


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
