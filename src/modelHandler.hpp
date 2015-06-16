/*
 * modelHandler.hpp
 *   (ここにファイルの簡易説明を記入)
 *
 *  Created on: 2015/05/24
 *      Author: wlamigo
 * 
 *   (ここにファイルの説明を記入)
 */

#ifndef MODEL_HANDLER_HPP_
#define MODEL_HANDLER_HPP_

#include <opencv2/opencv.hpp>
#include "picojson.h"
#include "Buffer.hpp"
#include "filters.hpp"
#include <iostream>
#include <memory>
#include <cstdint>
#include <cstdlib>

namespace w2xc {

class Model {

private:
	int nInputPlanes;
	int nOutputPlanes;
	std::vector<cv::Mat> weights;
	std::vector<double> biases;
	int kernelSize;

	Model() {
	}
	; // cannot use no-argument constructor

	// class inside operation function
	bool loadModelFromJSONObject(picojson::object& jsonObj);

	// thread worker function
	bool filterWorker(std::vector<cv::Mat> &inputPlanes,
			std::vector<cv::Mat> &weightMatrices,
			std::vector<cv::Mat> &outputPlanes, unsigned int beginningIndex,
			unsigned int nWorks);

	bool filter_CV(ComputeEnv *env,
		       Buffer *packed_input,
		       Buffer *packed_output,
		       cv::Size size);
	enum runtype {
		RUN_CUDA,
		RUN_OPENCL,
		RUN_CPU
	};

	bool filter_AVX_OpenCL(ComputeEnv *env,
			       Buffer *packed_input,
                               Buffer *packed_output,
                               cv::Size size,
			       enum runtype rt);

public:
	// ctor and dtor
	Model(picojson::object &jsonObj) {
		// preload nInputPlanes,nOutputPlanes, and preserve required size vector
		nInputPlanes = static_cast<int>(jsonObj["nInputPlane"].get<double>());
		nOutputPlanes = static_cast<int>(jsonObj["nOutputPlane"].get<double>());
		if ((kernelSize = static_cast<int>(jsonObj["kW"].get<double>()))
				!= static_cast<int>(jsonObj["kH"].get<double>())) {
			std::cerr << "Error : Model-Constructor : \n"
					"kernel in model is not square.\n"
					"stop." << std::endl;
			std::exit(-1);
		} // kH == kW

		weights = std::vector<cv::Mat>(nInputPlanes * nOutputPlanes,
				cv::Mat(kernelSize, kernelSize, CV_32FC1));
		biases = std::vector<double>(nOutputPlanes, 0.0);

		if (!loadModelFromJSONObject(jsonObj)) {
			std::cerr
					<< "Error : Model-Constructor : \n"
							"something error has been occured in loading model from JSON-Object.\n"
							"stop." << std::endl;
			std::exit(-1);
		}
	}
	Model(FILE *binfp);

	;
	~Model() {
	}

	// for debugging
	void printWeightMatrix();
	void printBiases();

	// getter function
	int getNInputPlanes();
	int getNOutputPlanes();

	std::vector<cv::Mat> &getWeigts() {
		return weights;
	}
	std::vector<double> &getBiases() {
		return biases;
	}
	// setter function

	// public operation function
	bool filter(ComputeEnv *env,
		    Buffer *packed_input,
		    Buffer *packed_output,
		    cv::Size size);


};

class modelUtility {

private:
	static modelUtility* instance;
	int nJob;
	modelUtility() :
		nJob(4) {
	}
	;

public:
	static bool generateModelFromJSON(const std::string &fileName,
			std::vector<std::unique_ptr<Model> > &models);
	static modelUtility& getInstance();
	bool setNumberOfJobs(int setNJob);
	int getNumberOfJobs();
};

}

#endif /* MODEL_HANDLER_HPP_ */
