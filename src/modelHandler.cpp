/*
 * modelHandler.cpp
 *   (ここにファイルの簡易説明を記入)
 *
 *  Created on: 2015/05/24
 *      Author: wlamigo
 * 
 *   (ここにファイルの説明を記入)
 */

#include "modelHandler.hpp"
// #include <iostream> in modelHandler.hpp
#include "cvwrap.hpp"
#include <fstream>
#include <thread>
#include "sec.hpp"
//#include "threadPool.hpp"
#include "common.hpp"
#include "filters.hpp"
#include "params.h"

namespace w2xc {

int Model::getNInputPlanes() {
	return nInputPlanes;
}

int Model::getNOutputPlanes() {
	return nOutputPlanes;
}

bool
Model::filter_CV(ComputeEnv *env,
		 Buffer *packed_input_buf,
		 Buffer *packed_output_buf,
		 const W2Size &size)
{
#ifdef HAVE_OPENCV
	size_t in_size = sizeof(float) * size.width * size.height * nInputPlanes;

	const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env, in_size);
	float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

	std::vector<cv::Mat> outputPlanes;
	std::vector<cv::Mat> inputPlanes;

	for (int i = 0; i < nInputPlanes; i++) {
		inputPlanes.push_back(cv::Mat::zeros(cvSize_from_w2(size), CV_32FC1));
	}
	std::vector<W2Mat> inputPlanes_2(extract_viewlist_from_cvmat(inputPlanes));
	unpack_mat(inputPlanes_2, packed_input, size.width, size.height, nInputPlanes);

	outputPlanes.clear();
	for (int i = 0; i < nOutputPlanes; i++) {
		outputPlanes.push_back(cv::Mat::zeros(cvSize_from_w2(size), CV_32FC1));
	}

	int nJob = modelUtility::getInstance().getNumberOfJobs();

	// filter job issuing
	std::vector<std::thread> workerThreads;
	int worksPerThread = nOutputPlanes / nJob;

	std::vector<W2Mat> inputPlanes_w2 = extract_viewlist_from_cvmat(inputPlanes);
	std::vector<W2Mat> outputPlanes_w2 = extract_viewlist_from_cvmat(outputPlanes);

	for (int idx = 0; idx < nJob; idx++) {
		if (!(idx == (nJob - 1) && worksPerThread * nJob != nOutputPlanes)) {
			workerThreads.push_back(
					std::thread(&Model::filterWorker, this,
						    std::ref(inputPlanes_w2), std::ref(weights),
						    std::ref(outputPlanes_w2),
						    static_cast<unsigned int>(worksPerThread * idx),
						    static_cast<unsigned int>(worksPerThread)));
		} else {
			// worksPerThread * nJob != nOutputPlanes
			workerThreads.push_back(
					std::thread(&Model::filterWorker, this,
						    std::ref(inputPlanes_w2), std::ref(weights),
						    std::ref(outputPlanes_w2),
						    static_cast<unsigned int>(worksPerThread * idx),
						    static_cast<unsigned int>(nOutputPlanes
									      - worksPerThread * idx)));
		}
	}
	// wait for finishing jobs
	for (auto& th : workerThreads) {
		th.join();
	}

	std::vector<W2Mat> outputPlanes_2(extract_viewlist_from_cvmat(outputPlanes));
	pack_mat(packed_output, outputPlanes_2, size.width, size.height, nOutputPlanes);

	return true;
#else
	abort();

#endif
}

//#define COMPARE_RESULT

bool Model::filter_AVX_OpenCL(W2XConv *conv,
			      ComputeEnv *env,
			      Buffer *packed_input_buf,
			      Buffer *packed_output_buf,
			      const W2Size &size)
{
	int vec_width;
	int weight_step;
	int nJob = modelUtility::getInstance().getNumberOfJobs();
	const struct W2XConvProcessor *proc = conv->target_processor;

	bool gpu = (proc->type == W2XCONV_PROC_OPENCL) || (proc->type == W2XCONV_PROC_CUDA);

	if (gpu) {
		weight_step = GPU_VEC_WIDTH;
		vec_width = GPU_VEC_WIDTH;
	} else {
		weight_step = nOutputPlanes;
		vec_width = VEC_WIDTH;
	}

	float *weight_flat = (float*)w2xc_aligned_malloc(sizeof(float)*nInputPlanes*weight_step*3*3, 64);
	float *fbiases_flat = (float*)w2xc_aligned_malloc(sizeof(float) * biases.size(), 64);

	for (int i=0; i<(int)biases.size(); i++) {
		fbiases_flat[i] = biases[i];
	}

	if (nOutputPlanes == 1) {
		if (gpu) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				W2Mat &wm = weights[ii];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				float *dst = weight_flat + ii * 9;
				dst[0] = src0[0];
				dst[1] = src0[1];
				dst[2] = src0[2];

				dst[3] = src1[0];
				dst[4] = src1[1];
				dst[5] = src1[2];

				dst[6] = src2[0];
				dst[7] = src2[1];
				dst[8] = src2[2];

			}
		} else {
			for (int ii=0; ii<nInputPlanes; ii++) {
				W2Mat &wm = weights[ii];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				int ii_0 = ii % vec_width;
				int ii_1 = (ii / vec_width) * vec_width;

				float *dst = weight_flat + ii_1 * 9  + ii_0;
				dst[0 * vec_width] = src0[0];
				dst[1 * vec_width] = src0[1];
				dst[2 * vec_width] = src0[2];

				dst[3 * vec_width] = src1[0];
				dst[4 * vec_width] = src1[1];
				dst[5 * vec_width] = src1[2];

				dst[6 * vec_width] = src2[0];
				dst[7 * vec_width] = src2[1];
				dst[8 * vec_width] = src2[2];
			}
		}
	} else if (gpu && nInputPlanes == 1) {
		for (int oi=0; oi<nOutputPlanes; oi++) {
			W2Mat &wm = weights[oi];
			const float *src0 = wm.ptr<float>(0);
			const float *src1 = wm.ptr<float>(1);
			const float *src2 = wm.ptr<float>(2);

			float *dst = weight_flat + oi * 9;
			dst[0] = src0[0];
			dst[1] = src0[1];
			dst[2] = src0[2];

			dst[3] = src1[0];
			dst[4] = src1[1];
			dst[5] = src1[2];

			dst[6] = src2[0];
			dst[7] = src2[1];
			dst[8] = src2[2];
		}
	} else if (nOutputPlanes == 3) {
		/* |       o0        |       o1        | o2 ... |
		 * |i0 i1 i2 ... i127|i0 i1 i2 ... i127| ...    |*/

		for (int oi=0; oi<nOutputPlanes; oi++) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				int mi = oi*nInputPlanes+ii;
				W2Mat &wm = weights[mi];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				float *dst = weight_flat + (oi * nInputPlanes * 9) + ii;
				dst[0*nInputPlanes] = src0[0];
				dst[1*nInputPlanes] = src0[1];
				dst[2*nInputPlanes] = src0[2];

				dst[3*nInputPlanes] = src1[0];
				dst[4*nInputPlanes] = src1[1];
				dst[5*nInputPlanes] = src1[2];

				dst[6*nInputPlanes] = src2[0];
				dst[7*nInputPlanes] = src2[1];
				dst[8*nInputPlanes] = src2[2];
			}
		}
	} else if (gpu && (nInputPlanes == 3) && (nOutputPlanes == 32)) {
		/* | i0             | i1        | i2 .. iN-1|
		 * |o0 o1 o2 o3..o31|o0 .... o32| ....      |
		 * |<-            ->|
		 * |    32          |
		 * |   x  9         |
		 */

		for (int oi=0; oi<nOutputPlanes; oi++) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				int mi = oi*nInputPlanes+ii;
				W2Mat &wm = weights[mi];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				float *dst = weight_flat + (ii * nOutputPlanes * 9) + oi;
				dst[0*nOutputPlanes] = src0[0];
				dst[1*nOutputPlanes] = src0[1];
				dst[2*nOutputPlanes] = src0[2];

				dst[3*nOutputPlanes] = src1[0];
				dst[4*nOutputPlanes] = src1[1];
				dst[5*nOutputPlanes] = src1[2];

				dst[6*nOutputPlanes] = src2[0];
				dst[7*nOutputPlanes] = src2[1];
				dst[8*nOutputPlanes] = src2[2];
			}
		}
	} else {
		bool simd_available = false;
		int simd_vec_width = 0;
		if (proc->type == W2XCONV_PROC_HOST) {
			switch (proc->sub_type) {
			case W2XCONV_PROC_HOST_SSE3:
			case W2XCONV_PROC_HOST_NEON:
				simd_vec_width = 4;
				simd_available = true;
				break;


			case W2XCONV_PROC_HOST_AVX:
			case W2XCONV_PROC_HOST_FMA:
				simd_vec_width = 8;
				simd_available = true;
				break;
			}
		}

		simd_available = simd_available && (nInputPlanes%(simd_vec_width*4) == 0) && (nOutputPlanes%(simd_vec_width*2) == 0);

		if (simd_available) {
			/* 
			 * weight_chunk (16x32x3x4 = 6144[Byte])
			 * (where op_block_size=16, ip_block_size=32)
			 *
			 * 111                                            oplane x16
			 * 16 16 .. (x16)  ..16                           iplane x32
			 *            \               |               /   horiz  x3
			 *                                                oplane xnOutputPlane_block
			 *                                                iplane xnInputPlane_block
			 *                                                vert   x3
			 */
			int ip_block_size = (simd_vec_width*4);
			int op_block_size = (simd_vec_width*2);
			int nInputPlane_block = nInputPlanes/ip_block_size;
			int nOutputPlane_block = nOutputPlanes/op_block_size;

			float *dst = weight_flat;

			for (int dposy=0; dposy<3; dposy++) {
				for (int ii0=0; ii0<nInputPlane_block; ii0++) {
					for (int oi0=0; oi0<nOutputPlane_block; oi0++) {
						for (int dposx=0; dposx<3; dposx++) {
							for (int ii1=0; ii1<ip_block_size; ii1++) {
								for (int oi1=0; oi1<op_block_size; oi1++) {
									int ii = ii0*ip_block_size + ii1;
									int oi = oi0*op_block_size + oi1;
									int mi = oi*nInputPlanes + ii;

									W2Mat &wm = weights[mi];
									float &src = wm.at<float>(dposy, dposx);
									*dst = src;

									dst++;
								}
							}
						}
					}
				}
			}

			//for (int i=0; i<nInputPlanes*nOutputPlanes; i++) {
			//	W2Mat &wm = weights[i];
			//
			//	for (int yi=0; yi<3; yi++) {
			//		for (int xi=0; xi<3; xi++) {
			//			float v = wm.at<float>(yi,xi);
			//			printf("%f\n", v);
			//		}
			//	}
			//}
		} else {
			/* | i0        | i1        | i2 .. iN-1|   i0      | i1        | ..
			 * |o0 o1 o2 o3|o0 o1 o2 o3| ....      |o4 o5 o6 o7|o4 o5 o6 o7| ..
			 * |<-       ->|
			 * | VEC_WIDTH |
			 * |   x  9    |
			 */

			for (int oi=0; oi<nOutputPlanes; oi++) {
				for (int ii=0; ii<nInputPlanes; ii++) {
					int mi = oi*nInputPlanes+ii;
					W2Mat &wm = weights[mi];
					const float *src0 = wm.ptr<float>(0);
					const float *src1 = wm.ptr<float>(1);
					const float *src2 = wm.ptr<float>(2);

					int oi_0 = oi % vec_width;
					int oi_1 = (oi / vec_width) * vec_width;

					float *dst = weight_flat + ((ii*weight_step + oi_1) * 9) + oi_0;
					dst[0*vec_width] = src0[0];
					dst[1*vec_width] = src0[1];
					dst[2*vec_width] = src0[2];

					dst[3*vec_width] = src1[0];
					dst[4*vec_width] = src1[1];
					dst[5*vec_width] = src1[2];

					dst[6*vec_width] = src2[0];
					dst[7*vec_width] = src2[1];
					dst[8*vec_width] = src2[2];
				}
			}
		}
	}

	bool compare_result = false;

#ifdef COMPARE_RESULT
	compare_result = true;
#endif

	size_t in_size = size.width * size.height * sizeof(float) * nInputPlanes;
	size_t out_size = size.width * size.height * sizeof(float) * nOutputPlanes;

	if (compare_result) {
		Buffer *packed_output_cv_buf = new Buffer(env, sizeof(float) * size.width * size.height * nOutputPlanes);

		double t0 = getsec();
		filter_CV(env, packed_input_buf, packed_output_cv_buf, size);
		//filter_FMA_impl(packed_input, packed_output_cv,
		//		nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		double t1 = getsec();

		/* 3x3 = 9 fma */
		double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;

		if (proc->type == W2XCONV_PROC_OPENCL) {
			filter_OpenCL_impl(env, packed_input_buf, packed_output_buf,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					   size.width, size.height, nJob);
		} else if (proc->type == W2XCONV_PROC_CUDA) {
			filter_CUDA_impl(env, packed_input_buf, packed_output_buf,
					 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					 size.width, size.height, nJob);
		} else {
			const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env, in_size);
			float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

			switch (proc->sub_type) {
#ifdef X86OPT
			case W2XCONV_PROC_HOST_FMA:
				filter_FMA_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_AVX:
				filter_AVX_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_SSE3:
				filter_SSE_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;
#endif
#ifdef ARMOPT
			case W2XCONV_PROC_HOST_NEON:
				filter_NEON_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;
#endif

			default:
				filter_CV(env, packed_input_buf, packed_output_buf, size);
				break;
			}
		}

		double t2 = getsec();

		printf("(w=%d,h=%d) (ip=%d,op=%d) %f %f %f[gflops]\n", size.width, size.height, nInputPlanes, nOutputPlanes, t1-t0, t2-t1, ops/(1000*1000*1000));
		printf("ver2 : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1));
		printf("orig : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t1-t0));
		int error_count = 0;

		float *packed_output_cv = (float*)packed_output_cv_buf->get_read_ptr_host(env, out_size);
		float *packed_output = (float*)packed_output_buf->get_read_ptr_host(env, out_size);

		for (int i=0; i<size.width * size.height * nOutputPlanes; i++) {
			float v0 = packed_output_cv[i];
			float v1 = packed_output[i];
			float d = fabs(v0 - v1);

			float r0 = d/fabs(v0);
			float r1 = d/fabs(v1);

			float r = (std::max)(r0, r1);

			if (r > 0.1f && d > 0.000001f) {
				int plane = i % nOutputPlanes;
				int pixpos = i / nOutputPlanes;
				int xpos = pixpos % size.width;
				int ypos = pixpos / size.width;

				printf("d=%.20f %.20f %.20f @ (%d,%d,%d,%d) \n",r, v0, v1, xpos, ypos, plane, i);
				error_count++;

				if (error_count >= 256) {
					exit(1);
				}
			}
		}

		if (error_count != 0) {
			exit(1);
		}

		delete packed_output_cv_buf;
	} else {
		if (proc->type == W2XCONV_PROC_OPENCL) {
			filter_OpenCL_impl(env, packed_input_buf, packed_output_buf,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					   size.width, size.height, nJob);
		} else if (proc->type == W2XCONV_PROC_CUDA) {
			filter_CUDA_impl(env, packed_input_buf, packed_output_buf,
					 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					 size.width, size.height, nJob);
		} else {
			const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env, in_size);
			float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

			switch (proc->sub_type) {
#ifdef X86OPT
			case W2XCONV_PROC_HOST_FMA:
				filter_FMA_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_AVX:
				filter_AVX_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_SSE3:
				filter_SSE_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;
#endif
#ifdef ARMOPT
			case W2XCONV_PROC_HOST_NEON:
				filter_NEON_impl(env, packed_input, packed_output,
						 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						 size.width, size.height, nJob);
				break;
#endif
			default:
				filter_CV(env, packed_input_buf, packed_output_buf, size);
				break;
			}
		}
	}

	w2xc_aligned_free(fbiases_flat);
	w2xc_aligned_free(weight_flat);

	return true;

}

bool Model::filter(W2XConv *conv,
		   ComputeEnv *env,
		   Buffer *packed_input_buf,
		   Buffer *packed_output_buf,
		   W2Size const &size)
{
	bool ret;

	bool avx_available = true;
	bool cl_available = true;
	bool cuda_available = true;

	if (nOutputPlanes > GPU_VEC_WIDTH) {
		cl_available = false;
		cuda_available = false;
	}

	if (nOutputPlanes == 32 && nInputPlanes == 1) {
		/* i1 o32 filter */
	} else if (nOutputPlanes == 1 && nInputPlanes == 128) {
		/* i128 o32 filter */
	} else if (nOutputPlanes == 32 && nInputPlanes == 3) {
		/* i3 o32 filter */
	} else if (nOutputPlanes == 3 && nInputPlanes == 128) {
		/* i128 o3 filter */
	} else {
		if (nInputPlanes & 1) {
			cl_available = false;
			cuda_available = false;
			avx_available = false;
		}

		if (nOutputPlanes & 31) {
			cl_available = false;
			cuda_available = false;
			avx_available = false;
		}

		if (nInputPlanes == 32 || nInputPlanes == 64 || nInputPlanes == 128) {
			/* ok */
		} else {
			cuda_available = false;
		}
	}

	//printf("%d %d %d\n",
	//       (int)cuda_available,
	//       (int)cl_available,
	//       (int)avx_available);

	const struct W2XConvProcessor *proc = conv->target_processor;
	if ((cl_available && proc->type == W2XCONV_PROC_OPENCL) ||
	    (cuda_available && proc->type == W2XCONV_PROC_CUDA) ||
	    (avx_available && proc->type == W2XCONV_PROC_HOST))
	{
		ret = filter_AVX_OpenCL(conv, env, packed_input_buf, packed_output_buf, size);
	} else {
		ret = filter_CV(env, packed_input_buf, packed_output_buf, size);
	}

	return ret;
}

bool Model::loadModelFromJSONObject(picojson::object &jsonObj) {

	// nInputPlanes,nOutputPlanes,kernelSize have already set.
	int matProgress = 0;
	picojson::array &wOutputPlane = jsonObj["weight"].get<picojson::array>();

	// setting weight matrices
	for (auto&& wInputPlaneV : wOutputPlane) {
		picojson::array &wInputPlane = wInputPlaneV.get<picojson::array>();

		for (auto&& weightMatV : wInputPlane) {
			picojson::array &weightMat = weightMatV.get<picojson::array>();
			W2Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);

			for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
				auto& weightMatRowV = weightMat.at(writingRow);
				picojson::array &weightMatRow = weightMatRowV.get<
						picojson::array>();

				for (int index = 0; index < kernelSize; index++) {
					writeMatrix.ptr<float>(writingRow)[index] = weightMatRow[index].get<double>();
				} // for(weightMatRow) (writing 1 row finished)

			} // for(weightMat) (writing 1 matrix finished)

			weights.push_back(std::move(writeMatrix));
			matProgress++;
		} // for(wInputPlane) (writing matrices in set of wInputPlane finished)

	} //for(wOutputPlane) (writing all matrices finished)

	// setting biases
	picojson::array biasesData = jsonObj["bias"].get<picojson::array>();
	for (int index = 0; index < nOutputPlanes; index++) {
		biases[index] = biasesData[index].get<double>();
	}

	return true;
}

#ifdef HAVE_OPENCV
bool Model::filterWorker(std::vector<W2Mat> &inputPlanes_w2,
			 std::vector<W2Mat> &weightMatrices_w2,
			 std::vector<W2Mat> &outputPlanes_w2, unsigned int beginningIndex,
			 unsigned int nWorks) {

	std::vector<cv::Mat> inputPlanes = extract_viewlist_to_cvmat(inputPlanes_w2);
	std::vector<cv::Mat> weightMatrices = extract_viewlist_to_cvmat(weightMatrices_w2);
	std::vector<cv::Mat> outputPlanes = extract_viewlist_to_cvmat(outputPlanes_w2);
			 
	cv::Size ipSize = inputPlanes[0].size();
	// filter processing
	// input : inputPlanes
	// kernel : weightMatrices

	for (int opIndex = beginningIndex;
	     opIndex < (int)(beginningIndex + nWorks);
	     opIndex++) {
		int wMatIndex = nInputPlanes * opIndex;
		cv::Mat outputPlane = cv::Mat::zeros(ipSize, CV_32FC1);
		cv::Mat &uIntermediatePlane = outputPlane; // all zero matrix

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
			cv::Mat &uInputPlane = inputPlanes[ipIndex];
			cv::Mat &weightMatrix = weightMatrices[wMatIndex + ipIndex];
			cv::Mat filterOutput = cv::Mat::zeros(ipSize, CV_32FC1);

			cv::filter2D(uInputPlane, filterOutput, -1, weightMatrix,
				     cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

			cv::add(uIntermediatePlane, filterOutput, uIntermediatePlane);
		}

		cv::add(uIntermediatePlane, biases[opIndex], uIntermediatePlane);
		cv::Mat moreThanZero = cv::Mat(ipSize,CV_32FC1,0.0);
		cv::Mat lessThanZero = cv::Mat(ipSize,CV_32FC1,0.0);
		(cv::max)(uIntermediatePlane, 0.0, moreThanZero);
		(cv::min)(uIntermediatePlane, 0.0, lessThanZero);
		cv::scaleAdd(lessThanZero, 0.1, moreThanZero, uIntermediatePlane);
		uIntermediatePlane.copyTo(outputPlanes[opIndex]);

	} // for index

	return true;
}
#endif

modelUtility * modelUtility::instance = nullptr;

modelUtility& modelUtility::getInstance(){
	if(instance == nullptr){
		instance = new modelUtility();
	}
	return *instance;
}

Model::Model(FILE *binfp)
{
	uint32_t nInputPlanes, nOutputPlanes;

	fread(&nInputPlanes, 4, 1, binfp);
	fread(&nOutputPlanes, 4, 1, binfp);

	this->nInputPlanes = nInputPlanes;
	this->nOutputPlanes = nOutputPlanes;
	this->kernelSize = 3;
	this->weights.clear();
	this->biases.clear();

	// setting weight matrices
	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		for (uint32_t ii=0; ii<nInputPlanes; ii++) {
			W2Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);
			for (int yi=0; yi<3; yi++) {
				for (int xi=0; xi<3; xi++) {
					double v;
					fread(&v, 8, 1, binfp);
					writeMatrix.at<float>(yi, xi) = v;
				}
			}
			this->weights.emplace_back(std::move(writeMatrix));
		}
	}

	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		double v;
		fread(&v, 8, 1, binfp);
		biases.push_back(v);
	}
}

Model::Model(int nInputPlane,
	     int nOutputPlane,
	     const float *coef_list,
	     const float *bias)
{
	this->nInputPlanes = nInputPlane;
	this->nOutputPlanes = nOutputPlane;
	this->kernelSize = 3;
	this->weights.clear();
	this->biases.clear();

	int cur = 0;
	// setting weight matrices
	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		for (uint32_t ii=0; ii<nInputPlanes; ii++) {
			W2Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);
			for (int yi=0; yi<3; yi++) {
				for (int xi=0; xi<3; xi++) {
					double v = coef_list[cur++];
					writeMatrix.at<float>(yi, xi) = v;
				}
			}
			this->weights.emplace_back(std::move(writeMatrix));
		}
	}

	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		double v = bias[oi];
		biases.push_back(v);
	}
}



bool modelUtility::generateModelFromJSON(const std::string &fileName,
		std::vector<std::unique_ptr<Model> > &models) {

	std::string binpath = fileName + ".bin";
	FILE *binfp = fopen(binpath.c_str(), "rb");

	if (binfp) {
		uint32_t nModel;

		fread(&nModel, 4, 1, binfp);

		for (uint32_t i=0; i<nModel; i++) {
			std::unique_ptr<Model> m = std::unique_ptr<Model>(
				new Model(binfp));
			models.push_back(std::move(m));
		}

		fclose(binfp);
	} else {
		std::ifstream jsonFile;

		jsonFile.open(fileName);
		if (!jsonFile.is_open()) {
			std::cerr << "Error : couldn't open " << fileName << std::endl;
			return false;
		}

		picojson::value jsonValue;
		jsonFile >> jsonValue;

		std::string errMsg = picojson::get_last_error();
		if (!errMsg.empty()) {
			std::cerr << "Error : PicoJSON Error : " << errMsg << std::endl;
			return false;
		}

		picojson::array& objectArray = jsonValue.get<picojson::array>();
		for (auto&& obj : objectArray) {
			std::unique_ptr<Model> m = std::unique_ptr<Model>(
				new Model(obj.get<picojson::object>()));
			models.push_back(std::move(m));
		}

		binfp = fopen(binpath.c_str(), "wb");
		if (binfp) {
			uint32_t nModel = objectArray.size();

			fwrite(&nModel, 4, 1, binfp);
			for (auto&& m : models) {
				uint32_t nInputPlanes = m->getNInputPlanes();
				uint32_t nOutputPlanes = m->getNOutputPlanes();

				fwrite(&nInputPlanes, 4, 1, binfp);
				fwrite(&nOutputPlanes, 4, 1, binfp);

				std::vector<W2Mat> &weights = m->getWeigts();

				int nw = weights.size();
				for (int wi=0; wi<nw; wi++) {
					W2Mat &wm = weights[wi];
					double v;
					v = wm.at<float>(0,0);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(0,1);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(0,2);
					fwrite(&v, 1, 8, binfp);

					v = wm.at<float>(1,0);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(1,1);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(1,2);
					fwrite(&v, 1, 8, binfp);

					v = wm.at<float>(2,0);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(2,1);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(2,2);
					fwrite(&v, 1, 8, binfp);
				}

				std::vector<double> &b = m->getBiases();
				fwrite(&b[0], 8, b.size(), binfp);
			}

			fclose(binfp);
		}
	}

	return true;
}

void
modelUtility::generateModelFromMEM(int layer_depth,
				   int num_input_plane,
				   const int *num_map, // num_map[layer_depth]
				   const float *coef_list, // coef_list[layer_depth][num_map][3x3]
				   const float *bias, // bias[layer_depth][num_map]
				   std::vector<std::unique_ptr<Model> > &models
	)
{
	int cur = 0;
	models.resize(layer_depth);

	models[0] = std::unique_ptr<Model>(new Model(num_input_plane,
						     num_map[0],
						     &coef_list[0],
						     &bias[0]));

	cur += num_map[0];

	for (int li=1; li<layer_depth; li++) {
		models[li] = std::unique_ptr<Model>(new Model(num_map[li-1],
							      num_map[li],
							      &coef_list[cur * 3 * 3],
							      &bias[cur]));

		cur += num_map[li];
	}
}



bool modelUtility::setNumberOfJobs(int setNJob){
	if(setNJob < 1)return false;
	nJob = setNJob;

	return true;
};

int modelUtility::getNumberOfJobs(){
	return nJob;
}

// for debugging

void Model::printWeightMatrix() {

	for (auto&& weightMatrix : weights) {
		//std::cout << weightMatrix << std::endl;
	}

}

void Model::printBiases() {

	for (auto&& bias : biases) {
		std::cout << bias << std::endl;
	}
}

}
