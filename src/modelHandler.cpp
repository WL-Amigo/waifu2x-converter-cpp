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
#include <time.h>

static double
getsec()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return ts.tv_sec + ts.tv_nsec/1000000000.0;
}

namespace w2xc {

int Model::getNInputPlanes() {
	return nInputPlanes;
}

int Model::getNOutputPlanes() {
	return nOutputPlanes;
}

static void
filter2(std::vector<cv::Mat> &inputPlanes,
	std::vector<cv::Mat> &outputPlanes,
	int nOutputPlanes,
	std::vector<double> &biases,
	std::vector<cv::Mat> &weightMatrices)
{
	cv::ocl::setUseOpenCL(false); // disable OpenCL Support(temporary)

	int nInputPlanes = inputPlanes.size();
	outputPlanes.clear();
	for (int i = 0; i < nOutputPlanes; i++) {
		outputPlanes.push_back(cv::Mat::zeros(inputPlanes[0].size(), CV_32FC1));
	}

	cv::Size ipSize = inputPlanes[0].size();
	// filter processing
	// input : inputPlanes
	// kernel : weightMatrices
	for (int opIndex = 0;
	     opIndex < nOutputPlanes;
	     opIndex++)
	{
		int wMatIndex = nInputPlanes * opIndex;
		cv::Mat &uIntermediatePlane = outputPlanes[opIndex];

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
			cv::Mat &uInputPlane = inputPlanes[ipIndex];
			cv::Mat &weightMatrix = weightMatrices[wMatIndex + ipIndex];
			cv::Mat filterOutput = cv::Mat(ipSize, CV_32FC1);

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
	} // for index
}

bool Model::filter(std::vector<cv::Mat> &inputPlanes,
		std::vector<cv::Mat> &outputPlanes) {

	if (inputPlanes.size() != nInputPlanes) {
		std::cerr << "Error : Model-filter : \n"
				"number of input planes mismatch." << std::endl;
		std::cerr << inputPlanes.size() << ","
				<< nInputPlanes << std::endl;
		return false;
	}

	outputPlanes.clear();
	for (int i = 0; i < nOutputPlanes; i++) {
		outputPlanes.push_back(cv::Mat::zeros(inputPlanes[0].size(), CV_32FC1));
	}

	double t0 = getsec();
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
	double t1 = getsec();

	std::vector<cv::Mat> output2;
	filter2(inputPlanes, output2, nOutputPlanes, biases, weights);
	double t2 = getsec();

	printf("%d %d %f %f\n", nInputPlanes, nOutputPlanes, t1-t0, t2-t1);

	cv::Size ipSize = inputPlanes[0].size();
	/* 3x3 = 9 fma */
	double ops = ipSize.width * ipSize.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
	printf("orig : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t1-t0));
	printf("ver2 : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1));

	for (int i = 0; i < nOutputPlanes; i++) {
		cv::Mat &m0 = outputPlanes[i];
		cv::Mat &m1 = output2[i];

		for (int my=0; my<m0.size[1]; my++) {
			const float *p0 = (float*)m0.ptr(my);
			const float *p1 = (float*)m1.ptr(my);

			for (int mx=0; mx<m0.size[0]; mx++) {
				if (p0[mx] != p1[mx]) {
					printf("%f %f @ %d-(%d,%d)\n",p0[mx], p1[mx], i, mx, my);
					exit(1);
				}
			}
		}
	}

	return true;
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
	for (int opIndex = beginningIndex; opIndex < (beginningIndex + nWorks);
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
