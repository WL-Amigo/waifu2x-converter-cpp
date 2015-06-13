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
#include <fstream>
#include <thread>
#ifdef __GNUC__
#include <cpuid.h>
#else
#include <intrin.h>
#endif
#include "sec.hpp"
#include "threadPool.hpp"
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
		 cv::Size size)
{
	const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env);
	float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

	std::vector<cv::Mat> outputPlanes;
	std::vector<cv::Mat> inputPlanes;

	for (int i = 0; i < nInputPlanes; i++) {
		inputPlanes.push_back(cv::Mat::zeros(size, CV_32FC1));
	}
	unpack_mat(inputPlanes, packed_input, size.width, size.height, nInputPlanes);

	outputPlanes.clear();
	for (int i = 0; i < nOutputPlanes; i++) {
		outputPlanes.push_back(cv::Mat::zeros(size, CV_32FC1));
	}

	int nJob = modelUtility::getInstance().getNumberOfJobs();

	// filter job issuing
	std::vector<std::thread> workerThreads;
	int worksPerThread = nOutputPlanes / nJob;
	for (int idx = 0; idx < nJob; idx++) {
		if (!(idx == (nJob - 1) && worksPerThread * nJob != nOutputPlanes)) {
			workerThreads.push_back(
					std::thread(&Model::filterWorker, this,
							std::ref(inputPlanes), std::ref(weights),
							std::ref(outputPlanes),
							static_cast<unsigned int>(worksPerThread * idx),
							static_cast<unsigned int>(worksPerThread)));
		} else {
			// worksPerThread * nJob != nOutputPlanes
			workerThreads.push_back(
					std::thread(&Model::filterWorker, this,
							std::ref(inputPlanes), std::ref(weights),
							std::ref(outputPlanes),
							static_cast<unsigned int>(worksPerThread * idx),
							static_cast<unsigned int>(nOutputPlanes
									- worksPerThread * idx)));
		}
	}
	// wait for finishing jobs
	for (auto& th : workerThreads) {
		th.join();
	}

	pack_mat(packed_output, outputPlanes, size.width, size.height, nOutputPlanes);

	return true;
}

#define COMPARE_RESULT

bool Model::filter_AVX_OpenCL(ComputeEnv *env,
			      Buffer *packed_input_buf,
			      Buffer *packed_output_buf,
			      cv::Size size,
			      enum runtype rt)
{
	int vec_width;
	int weight_step;
	unsigned int eax=0, ebx=0, ecx=0, edx=0;
	bool have_fma = false, have_avx = false;
	int nJob = modelUtility::getInstance().getNumberOfJobs();

#ifdef __GNUC__
	__get_cpuid(1, &eax, &ebx, &ecx, &edx);
#else
	int cpuInfo[4];
	__cpuid(cpuInfo, 1);
	eax = cpuInfo[0];
	ebx = cpuInfo[1];
	ecx = cpuInfo[2];
	edx = cpuInfo[3];
#endif
	if ((ecx & 0x18000000) == 0x18000000) {
		have_avx = true;
	}

	if (ecx & (1<<12)) {
		have_fma = true;
	}

	bool gpu = (rt == RUN_OPENCL) || (rt == RUN_CUDA);

	if (gpu) {
		weight_step = GPU_VEC_WIDTH;
		vec_width = GPU_VEC_WIDTH;
	} else {
		weight_step = nOutputPlanes;
		vec_width = VEC_WIDTH;
	}

	float *weight_flat = (float*)_mm_malloc(sizeof(float)*nInputPlanes*weight_step*3*3, 64);
	float *fbiases_flat = (float*)_mm_malloc(sizeof(float) * biases.size(), 64);

	for (int i=0; i<(int)biases.size(); i++) {
		fbiases_flat[i] = biases[i];
	}

	if (nOutputPlanes == 1) {
		if (gpu) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				cv::Mat &wm = weights[ii];
				const float *src0 = (float*)wm.ptr(0);
				const float *src1 = (float*)wm.ptr(1);
				const float *src2 = (float*)wm.ptr(2);

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
				cv::Mat &wm = weights[ii];
				const float *src0 = (float*)wm.ptr(0);
				const float *src1 = (float*)wm.ptr(1);
				const float *src2 = (float*)wm.ptr(2);

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
			cv::Mat &wm = weights[oi];
			const float *src0 = (float*)wm.ptr(0);
			const float *src1 = (float*)wm.ptr(1);
			const float *src2 = (float*)wm.ptr(2);

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
	} else {
		for (int oi=0; oi<nOutputPlanes; oi++) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				int mi = oi*nInputPlanes+ii;
				cv::Mat &wm = weights[mi];
				const float *src0 = (float*)wm.ptr(0);
				const float *src1 = (float*)wm.ptr(1);
				const float *src2 = (float*)wm.ptr(2);

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

	bool compare_result = false;

#ifdef COMPARE_RESULT
	compare_result = true;
#endif

	if (compare_result) {
		Buffer *packed_output_cv_buf = new Buffer(env, sizeof(float) * size.width * size.height * nOutputPlanes);

		double t0 = getsec();
		filter_CV(env, packed_input_buf, packed_output_cv_buf, size);
		//filter_FMA_impl(packed_input, packed_output_cv,
		//		nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		double t1 = getsec();

		/* 3x3 = 9 fma */
		double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
		std::vector<cv::Mat> output2;
		if (rt == RUN_OPENCL) {
			filter_OpenCL_impl(env, packed_input_buf, packed_output_buf,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					   size.width, size.height, nJob);
		} else if (rt == RUN_CUDA) {
			filter_CUDA_impl(env, packed_input_buf, packed_output_buf,
					 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					 size.width, size.height, nJob);
		} else {
			const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env);
			float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

			if (have_fma) {
				filter_FMA_impl(packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
			} else {
				filter_AVX_impl(packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
			}
		}

		double t2 = getsec();

		printf("%d %d %d %d %f %f %f[gops]\n", size.width, size.height, nInputPlanes, nOutputPlanes, t1-t0, t2-t1, ops/(1000*1000*1000));
		printf("ver2 : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1));
		printf("orig : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t1-t0));
		int error_count = 0;

		float *packed_output_cv = (float*)packed_output_cv_buf->get_read_ptr_host(env);
		float *packed_output = (float*)packed_output_buf->get_read_ptr_host(env);

		for (int i=0; i<size.width * size.height * nOutputPlanes; i++) {
			float v0 = packed_output_cv[i];
			float v1 = packed_output[i];
			float d = fabs(v0 - v1);

			float r0 = d/fabs(v0);
			float r1 = d/fabs(v1);

			float r = (std::max)(r0, r1);

			if (r > 0.1f && d > 0.0000001f) {
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
		if (rt == RUN_OPENCL) {
			filter_OpenCL_impl(env,
					   packed_input_buf, packed_output_buf,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					   size.width, size.height, nJob);
		} else if (rt == RUN_CUDA) {
			filter_CUDA_impl(env,
					 packed_input_buf, packed_output_buf,
					 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					 size.width, size.height, nJob);
		} else {
			if (!have_avx) {
				filter_CV(env, packed_input_buf, packed_output_buf, size);
			} else {
				const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env);
				float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

				if (have_fma) {
					filter_FMA_impl(packed_input, packed_output,
							nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
							size.width, size.height, nJob);
				} else if (have_avx) {
					filter_AVX_impl(packed_input, packed_output,
							nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
							size.width, size.height, nJob);
				}
			}
		}
	}

	_mm_free(fbiases_flat);
	_mm_free(weight_flat);

	return true;

}

bool Model::filter(ComputeEnv *env,
		   Buffer *packed_input_buf,
		   Buffer *packed_output_buf,
		   cv::Size size)
{
	bool ret;

	bool avx_available = true;
	bool cl_available = env->num_cl_dev > 0;
	bool cuda_available = env->num_cuda_dev > 0;

	//printf("%d->%d\n", nInputPlanes, nOutputPlanes);

	if (nOutputPlanes > GPU_VEC_WIDTH && nOutputPlanes % GPU_VEC_WIDTH) {
		cl_available = false;
		cuda_available = false;
	}

	if (nOutputPlanes == 1 || (nInputPlanes & 1)) {
		if (nOutputPlanes == 32 && nInputPlanes == 1) {
			/* in1 filter */
		} else if (nOutputPlanes == 1 && nInputPlanes == 128) {
			/* out1 filter */
		} else {
			cl_available = false;
		}

		cuda_available = false;
	}

	if (nOutputPlanes % (VEC_WIDTH*UNROLL)) {
		if (nOutputPlanes == 1) {
			if (nInputPlanes % VEC_WIDTH) {
				avx_available = false;
			}
		} else {
			avx_available = false;
		}
	}

	if (size.width&1) {
		avx_available = false;
	}

	if (cuda_available) {
		ret = filter_AVX_OpenCL(env, packed_input_buf, packed_output_buf, size, RUN_CUDA);
	} else if (cl_available) {
		ret = filter_AVX_OpenCL(env, packed_input_buf, packed_output_buf, size, RUN_OPENCL);
	} else if (avx_available) {
		ret = filter_AVX_OpenCL(env, packed_input_buf, packed_output_buf, size, RUN_CPU);
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
			cv::Mat writeMatrix = cv::Mat::zeros(kernelSize, kernelSize,
			CV_32FC1);

			for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
				auto& weightMatRowV = weightMat.at(writingRow);
				picojson::array &weightMatRow = weightMatRowV.get<
						picojson::array>();

				for (int index = 0; index < kernelSize; index++) {
					writeMatrix.at<float>(writingRow, index) =
							weightMatRow[index].get<double>();
				} // for(weightMatRow) (writing 1 row finished)

			} // for(weightMat) (writing 1 matrix finished)

			weights.at(matProgress) = std::move(writeMatrix);
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

bool Model::filterWorker(std::vector<cv::Mat> &inputPlanes,
		std::vector<cv::Mat> &weightMatrices,
		std::vector<cv::Mat> &outputPlanes, unsigned int beginningIndex,
		unsigned int nWorks) {

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
			cv::Mat filterOutput = cv::Mat(ipSize, CV_32FC1);

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
			cv::Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);
			for (int yi=0; yi<3; yi++) {
				for (int xi=0; xi<3; xi++) {
					double v;
					fread(&v, 8, 1, binfp);
					writeMatrix.at<float>(yi, xi) = v;
				}
			}
			this->weights.push_back(std::move(writeMatrix));
		}
	}

	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		double v;
		fread(&v, 8, 1, binfp);
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

				int kernelSize = 3;
				std::vector<cv::Mat> &weights = m->getWeigts();

				int nw = weights.size();
				for (int wi=0; wi<nw; wi++) {
					cv::Mat &wm = weights[wi];
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

bool modelUtility::setNumberOfJobs(int setNJob){
	if(setNJob < 1)return false;
	nJob = setNJob;

	initThreadPool(nJob);

	return true;
};

int modelUtility::getNumberOfJobs(){
	return nJob;
}

bool modelUtility::setBlockSize(cv::Size size){
	if(size.width < 0 || size.height < 0)return false;
	blockSplittingSize = size;
	return true;
}

bool modelUtility::setBlockSizeExp2Square(int exp){
	if(exp < 0)return false;
	int length = std::pow(2, exp);
	blockSplittingSize = cv::Size(length, length);
	return true;
}

cv::Size modelUtility::getBlockSize(){
	return blockSplittingSize;
}


// for debugging

void Model::printWeightMatrix() {

	for (auto&& weightMatrix : weights) {
		std::cout << weightMatrix << std::endl;
	}

}

void Model::printBiases() {

	for (auto&& bias : biases) {
		std::cout << bias << std::endl;
	}
}

}
