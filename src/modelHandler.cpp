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
#include "sec.hpp"
#include "common.hpp"

namespace w2xc {

int Model::getNInputPlanes() {
	return nInputPlanes;
}

int Model::getNOutputPlanes() {
	return nOutputPlanes;
}

bool
Model::filter_CV(const float *packed_input,
		 float *packed_output,
		 cv::Size size)
{
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

//#define COMPARE_RESULT

bool Model::filter_AVX_OpenCL(const float *packed_input,
			      float *packed_output,
			      cv::Size size,
			      bool OpenCL)
{
	int vec_width;

	if (have_OpenCL) {
		vec_width = GPU_VEC_WIDTH;
	} else {
		vec_width = VEC_WIDTH;
	}

	float *weight_flat = (float*)malloc(sizeof(float)*nInputPlanes*nOutputPlanes*3*3);
	float *fbiases_flat = (float*)malloc(sizeof(float) * biases.size());

	for (int i=0; i<(int)biases.size(); i++) {
		fbiases_flat[i] = biases[i];
	}

	if (nOutputPlanes == 1) {
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

				float *dst = weight_flat + ((ii*nOutputPlanes + oi_1) * 9) + oi_0;
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
	if (nOutputPlanes == 1) {
		compare_result = true;
	}
#endif

	if (compare_result) {
		float *packed_output_cv = (float*)malloc(sizeof(float) * size.width * size.height * nOutputPlanes);

		double t0 = getsec();
		filter_CV(packed_input, packed_output_cv, size);
		double t1 = getsec();

		/* 3x3 = 9 fma */
		double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
		std::vector<cv::Mat> output2;
		if (OpenCL) {
			filter_OpenCL_impl(packed_input, packed_output,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		} else {
			filter_AVX_impl(packed_input, packed_output,
					nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		}

		double t2 = getsec();

		printf("%d %d %f %f\n", nInputPlanes, nOutputPlanes, t1-t0, t2-t1);
		printf("ver2 : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1));
		printf("orig : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t1-t0));
		int error_count = 0;

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

				if (error_count >= 4) {
					exit(1);
				}
			}

		}

		if (error_count != 0) {
			exit(1);
		}
	} else {
		static double sum = 0;
		double t1 = getsec();
		if (OpenCL) {
			filter_OpenCL_impl(packed_input, packed_output,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		} else {
			filter_AVX_impl(packed_input, packed_output,
					nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		}
		double t2 = getsec();
		sum += t2 - t1;
		double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
		printf("ver2 : %f [Gflops], %f[msec], total= %f[msec]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1), (t2-t1)*1000, sum);
	}

	free(fbiases_flat);
	free(weight_flat);

	return true;

}

bool Model::filter(float *packed_input,
		   float *packed_output,
		   cv::Size size)
{
	bool ret;
	int vec_width;
	int unroll;

	if (have_OpenCL) {
		vec_width = GPU_VEC_WIDTH;
		unroll = 1;
	} else {
		vec_width = VEC_WIDTH;
		unroll = UNROLL;
	}

	bool avx_available = true;

	if (nOutputPlanes % (vec_width*unroll)) {
		if (nOutputPlanes == 1) {
			if (nInputPlanes % vec_width) {
				avx_available = false;
			}
		} else {
			avx_available = false;
		}
	}

	if (size.width&1) {
		avx_available = false;
	}

	if (avx_available) {
		ret = filter_AVX_OpenCL(packed_input, packed_output, size, have_OpenCL);
	} else {
		ret = filter_CV(packed_input, packed_output, size);
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
		cv::ocl::setUseOpenCL(false); // disable OpenCL Support(temporary)

		int wMatIndex = nInputPlanes * opIndex;
		cv::Mat outputPlane = cv::Mat::zeros(ipSize, CV_32FC1);
		cv::UMat uIntermediatePlane = outputPlane.getUMat(cv::ACCESS_WRITE); // all zero matrix

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
			cv::UMat uInputPlane = inputPlanes[ipIndex].getUMat(
					cv::ACCESS_READ);
			cv::UMat weightMatrix = weightMatrices[wMatIndex + ipIndex].getUMat(
					cv::ACCESS_READ);
			cv::UMat filterOutput = cv::UMat(ipSize, CV_32FC1);

			cv::filter2D(uInputPlane, filterOutput, -1, weightMatrix,
					cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

			cv::add(uIntermediatePlane, filterOutput, uIntermediatePlane);
		}

		cv::add(uIntermediatePlane, biases[opIndex], uIntermediatePlane);
		cv::UMat moreThanZero = cv::UMat(ipSize,CV_32FC1,0.0);
		cv::UMat lessThanZero = cv::UMat(ipSize,CV_32FC1,0.0);
		(cv::max)(uIntermediatePlane, 0.0, moreThanZero);
		(cv::min)(uIntermediatePlane, 0.0, lessThanZero);
		cv::scaleAdd(lessThanZero, 0.1, moreThanZero, uIntermediatePlane);
		outputPlane = uIntermediatePlane.getMat(cv::ACCESS_READ);
		outputPlane.copyTo(outputPlanes[opIndex]);

	} // for index

	return true;
}

void Model::setNumberOfJobs(int setNJob) {
	nJob = setNJob;
}

bool modelUtility::generateModelFromJSON(const std::string &fileName,
		std::vector<std::unique_ptr<Model> > &models) {

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

	return true;
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
