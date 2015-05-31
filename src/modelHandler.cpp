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

#define VEC_WIDTH 8U
#define UNROLL 2U

namespace w2xc {

int Model::getNInputPlanes() {
	return nInputPlanes;
}

int Model::getNOutputPlanes() {
	return nOutputPlanes;
}

template <bool border> inline
float
get_data(const float *p, int hsz, int wsz, int step, int yi, int xi, int num_plane, int plane)
{
	if (border) {
		yi = std::min(hsz-1, yi);
		yi = std::max(0, yi);

		xi = std::min(wsz-1, xi);
		xi = std::max(0, xi);

		char *p1 = (char*)p;
		return ((float*)(p1 + yi*step))[xi*num_plane + plane];
	} else {
		char *p1 = (char*)p;
		return ((float*)(p1 + yi*step))[xi*num_plane + plane];
	}
}

template <bool border> void
filter_1elem(const float *packed_input,
	     int nInputPlanes,
	     float *packed_output,
	     int nOutputPlanes,
	     float *biases,
	     unsigned long hsz,
	     unsigned long wsz,
	     unsigned long yi,
	     unsigned long xi,
	     float *weight,
	     float *intermediate)
{
	const float *in = packed_input;
	size_t in_step = wsz * sizeof(float) * nInputPlanes;

	for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {

#if 0
		float i00 = get_data<border>(in, hsz, wsz, in_step, yi-1, xi-1, ipIndex);
		float i01 = get_data<border>(in, hsz, wsz, in_step, yi-1, xi  , ipIndex);
		float i02 = get_data<border>(in, hsz, wsz, in_step, yi-1, xi+1, ipIndex);

		float i10 = get_data<border>(in, hsz, wsz, in_step, yi  , xi-1, ipIndex);
		float i11 = get_data<border>(in, hsz, wsz, in_step, yi  , xi  , ipIndex);
		float i12 = get_data<border>(in, hsz, wsz, in_step, yi  , xi+1, ipIndex);

		float i20 = get_data<border>(in, hsz, wsz, in_step, yi+1, xi-1, ipIndex);
		float i21 = get_data<border>(in, hsz, wsz, in_step, yi+1, xi  , ipIndex);
		float i22 = get_data<border>(in, hsz, wsz, in_step, yi+1, xi+1, ipIndex);

		float *w_base = weight + (ipIndex * nOutputPlanes) * 9;

		for (unsigned int opIndex = 0;
		     opIndex < (unsigned int)nOutputPlanes;
		     opIndex ++)
		{
			int oi_0 = opIndex % VEC_WIDTH;
			int oi_1 = (opIndex / VEC_WIDTH) * VEC_WIDTH;

			float *w = w_base + oi_1*9 + oi_0;
			float v = 0;

			v += w[0*VEC_WIDTH] * i00;
			v += w[1*VEC_WIDTH] * i01;
			v += w[2*VEC_WIDTH] * i02;

			v += w[3*VEC_WIDTH] * i10;
			v += w[4*VEC_WIDTH] * i11;
			v += w[5*VEC_WIDTH] * i12;

			v += w[6*VEC_WIDTH] * i20;
			v += w[7*VEC_WIDTH] * i21;
			v += w[8*VEC_WIDTH] * i22;

			if (ipIndex == 0) {
				intermediate[opIndex] = v;
			} else {
				intermediate[opIndex] += v;
			}
		}

#else
		__m256 i00 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi-1, xi-1, nInputPlanes, ipIndex));
		__m256 i01 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi-1, xi  , nInputPlanes, ipIndex));
		__m256 i02 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi-1, xi+1, nInputPlanes, ipIndex));

		__m256 i10 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi  , xi-1, nInputPlanes, ipIndex));
		__m256 i11 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi  , xi  , nInputPlanes, ipIndex));
		__m256 i12 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi  , xi+1, nInputPlanes, ipIndex));

		__m256 i20 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi+1, xi-1, nInputPlanes, ipIndex));
		__m256 i21 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi+1, xi  , nInputPlanes, ipIndex));
		__m256 i22 = _mm256_set1_ps(get_data<border>(in, hsz, wsz, in_step, yi+1, xi+1, nInputPlanes, ipIndex));

		float *w = weight + (ipIndex * nOutputPlanes) * 9;

		for (unsigned int opIndex = 0;
		     opIndex < (unsigned int)nOutputPlanes;
		     opIndex += VEC_WIDTH*UNROLL)
		{
#define APPLY_FILTER(e,off)						\
			__m256 v##e;					\
			v##e = _mm256_setzero_ps();		\
									\
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[0*VEC_WIDTH]), i00, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[1*VEC_WIDTH]), i01, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[2*VEC_WIDTH]), i02, v##e); \
									\
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[3*VEC_WIDTH]), i10, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[4*VEC_WIDTH]), i11, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[5*VEC_WIDTH]), i12, v##e); \
									\
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[6*VEC_WIDTH]), i20, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[7*VEC_WIDTH]), i21, v##e); \
			v##e = _mm256_fmadd_ps(_mm256_loadu_ps(&w[8*VEC_WIDTH]), i22, v##e); \
									\
			w += 9 * VEC_WIDTH;				\
									\

			APPLY_FILTER(0, 0);
			APPLY_FILTER(1, 8);

			if (ipIndex == 0) {
				_mm256_storeu_ps(&intermediate[opIndex+0], v0);
				_mm256_storeu_ps(&intermediate[opIndex+8], v1);
			} else {					\
				__m256 prev0 = _mm256_loadu_ps(&intermediate[opIndex+0]);
				__m256 prev1 = _mm256_loadu_ps(&intermediate[opIndex+8]);
				_mm256_storeu_ps(&intermediate[opIndex+0], _mm256_add_ps(prev0,v0));
				_mm256_storeu_ps(&intermediate[opIndex+8], _mm256_add_ps(prev1,v1));
			}
		}
#endif
	}

	float *out = packed_output + (yi*wsz + xi)*nOutputPlanes;
	for (int opIndex = 0; opIndex < nOutputPlanes; opIndex+=VEC_WIDTH) {
		__m256 bv = _mm256_loadu_ps(&biases[opIndex]);
		__m256 v = _mm256_loadu_ps(&intermediate[opIndex]);
		v = _mm256_add_ps(v, bv);
		__m256 mtz = _mm256_max_ps(v, _mm256_setzero_ps());
		__m256 ltz = _mm256_min_ps(v, _mm256_setzero_ps());

		v = _mm256_add_ps(_mm256_mul_ps(ltz, _mm256_set1_ps(0.1f)), mtz);

		_mm256_storeu_ps(&out[opIndex], v);
	}

}

static void
filter_AVX_impl(const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		std::vector<double> &biases,
		std::vector<cv::Mat> &weightMatrices,
		cv::Size ipSize,
		int nJob)
{
	int wsz = ipSize.width;
	int hsz = ipSize.height;

	// filter processing
	// input : inputPlanes
	// kernel : weightMatrices

	float *weight = (float*)malloc(sizeof(float)*nInputPlanes*nOutputPlanes*3*3);
	float *fbiases = (float*)malloc(sizeof(float) * biases.size());

	for (int i=0; i<(int)biases.size(); i++) {
		fbiases[i] = biases[i];
	}

	for (int oi=0; oi<nOutputPlanes; oi++) {
		for (int ii=0; ii<nInputPlanes; ii++) {
			int mi = oi*nInputPlanes+ii;
			cv::Mat &wm = weightMatrices[mi];
			const float *src0 = (float*)wm.ptr(0);
			const float *src1 = (float*)wm.ptr(1);
			const float *src2 = (float*)wm.ptr(2);

			int oi_0 = oi % VEC_WIDTH;
			int oi_1 = (oi / VEC_WIDTH) * VEC_WIDTH;

			float *dst = weight + ((ii*nOutputPlanes + oi_1) * 9) + oi_0;
			dst[0*VEC_WIDTH] = src0[0];
			dst[1*VEC_WIDTH] = src0[1];
			dst[2*VEC_WIDTH] = src0[2];

			dst[3*VEC_WIDTH] = src1[0];
			dst[4*VEC_WIDTH] = src1[1];
			dst[5*VEC_WIDTH] = src1[2];

			dst[6*VEC_WIDTH] = src2[0];
			dst[7*VEC_WIDTH] = src2[1];
			dst[8*VEC_WIDTH] = src2[2];
		}
	}

	int per_job = hsz / nJob;

	std::vector<std::thread> workerThreads;

	for (int ji=0; ji<nJob; ji++) {
		auto t = std::thread([&](int ji) {
				float *intermediate = (float*)malloc(sizeof(float)*nOutputPlanes);

				int start = per_job * ji, end;

				if (ji == nJob-1) {
					end = hsz;
				} else {
					end = per_job * (ji+1);
				}

				for (int yi=start; yi<end; yi++) {
					for (int xi=0; xi<wsz; xi++) {
						if (yi == 0 || xi ==0 || yi == (hsz-1) || xi == (wsz-1)) {
							filter_1elem<true>(packed_input, nInputPlanes,
									   packed_output, nOutputPlanes,
									   fbiases, hsz, wsz, yi, xi, weight, intermediate);
						} else {
							filter_1elem<false>(packed_input, nInputPlanes,
									    packed_output, nOutputPlanes,
									    fbiases, hsz, wsz, yi, xi, weight, intermediate);
						}
					}
				}

				free(intermediate);
			}, ji
			);

		workerThreads.push_back(std::move(t));
	}

	for (auto& th : workerThreads) {
		th.join();
	}

	free(fbiases);
	free(weight);
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

bool Model::filter_AVX(const float *packed_input,
		       float *packed_output,
		       cv::Size size)
{
#ifdef COMPARE_RESULT
	float *packed_output_cv = (float*)malloc(sizeof(float) * size.width * size.height * nOutputPlanes);

	double t0 = getsec();
	filter_CV(packed_input, packed_output_cv, size);
	double t1 = getsec();

	/* 3x3 = 9 fma */
	double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
	std::vector<cv::Mat> output2;
	filter_AVX_impl(packed_input, packed_output,
			nInputPlanes, nOutputPlanes, biases, weights, size, nJob);
	double t2 = getsec();

	printf("%d %d %f %f\n", nInputPlanes, nOutputPlanes, t1-t0, t2-t1);
	printf("ver2 : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1));
	printf("orig : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t1-t0));

	for (int i=0; i<size.width * size.height * nOutputPlanes; i++) {
		float v0 = packed_output_cv[i];
		float v1 = packed_output[i];
		float d = fabs(v0 - v1);

		float r0 = d/fabs(v0);
		float r1 = d/fabs(v1);

		float r = std::max(r0, r1);

		if (r > 0.1f && d > 0.0000001f) {
			printf("d=%.20f %.20f %.20f @ \n",r, v0, v1, i);
			exit(1);
		}

	}
#else
	double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;

	double t1 = getsec();
	filter_AVX_impl(packed_input, packed_output,
			nInputPlanes, nOutputPlanes, biases, weights, size, nJob);
	double t2 = getsec();

	printf("ver2 : %f [Gflops], %f[msec]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1), (t2-t1)*1000);
#endif

	return true;

}


bool Model::filter(float *packed_input,
		   float *packed_output,
		   cv::Size size)
{
	int ninput = nInputPlanes;
	int noutput = nOutputPlanes;

	bool ret;

	if (nOutputPlanes % (VEC_WIDTH*UNROLL)) {
		ret = filter_CV(packed_input, packed_output, size);
	} else {
		ret = filter_AVX(packed_input, packed_output, size);
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
		cv::max(uIntermediatePlane, 0.0, moreThanZero);
		cv::min(uIntermediatePlane, 0.0, lessThanZero);
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
