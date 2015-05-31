/*
 * convertRoutine.cpp
 *   (ここにファイルの簡易説明を記入)
 *
 *  Created on: 2015/05/31
 *      Author: wlamigo
 * 
 *   (ここにファイルの説明を記入)
 */

#include "convertRoutine.hpp"

namespace w2xc {

bool convertWithModels(cv::Mat &inputPlane, cv::Mat &outputPlane,
		std::vector<std::unique_ptr<Model> > &models) {

	std::unique_ptr<std::vector<cv::Mat> > inputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>());
	std::unique_ptr<std::vector<cv::Mat> > outputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>());

	//insert padding to inputPlane
	cv::Mat tempMat;
	int nModel = models.size();
	cv::Size outputSize = inputPlane.size();
	cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel,
			nModel, cv::BORDER_REPLICATE);

	inputPlanes->clear();
	inputPlanes->push_back(tempMat);

	for (int index = 0; index < models.size(); index++) {
		std::cout << "Iteration #" << (index + 1) << "..." << std::endl;
		if (!models[index]->filter(*inputPlanes, *outputPlanes)) {
			std::exit(-1);
		}
		if (index != models.size() - 1) {
			inputPlanes = std::move(outputPlanes);
			outputPlanes = std::unique_ptr<std::vector<cv::Mat> >(
					new std::vector<cv::Mat>());
		}
	}

	tempMat = outputPlanes->at(0)(cv::Range(nModel,outputSize.height + nModel),
			cv::Range(nModel,outputSize.width + nModel));
	assert(tempMat.size().width == outputSize.width && tempMat.size().height == outputSize.height);

	tempMat.copyTo(outputPlane);

	return true;

}

}

