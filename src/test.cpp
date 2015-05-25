/*
 * test.cpp
 *   (ここにファイルの簡易説明を記入)
 *
 *  Created on: 2015/05/24
 *      Author: wlamigo
 * 
 *   (ここにファイルの説明を記入)
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "picojson.h"

#include "modelHandler.hpp"

using namespace cv;

int main(int argc, char** argv) {

	std::string fileName(argv[1]);
	std::vector<std::unique_ptr<w2xc::Model> > models;

	w2xc::modelUtility::generateModelFromJSON(fileName, models);

	std::cout << "reading model data seems to be succeed." << std::endl;

	cv::Mat image = cv::imread("/home/wlamigo-data/Picture/HNI_0049(noise).png",
			cv::IMREAD_COLOR);
	cv::Mat image2x = cv::Mat(image.size().height * 2,image.size().width * 2,CV_32FC3);
	cv::resize(image,image2x,image2x.size(),0,0,INTER_NEAREST);
	cv::imwrite("/home/wlamigo-data/Picture/test-cv-nnresize.png",image2x);
	cv::Mat imageYUV;
	image2x.convertTo(imageYUV, CV_32F, 1.0 / 255.0);
	cv::cvtColor(imageYUV, imageYUV, COLOR_RGB2YUV);
	std::vector<cv::Mat> imageSprit;
	cv::Mat imageY;
	cv::split(imageYUV, imageSprit);
	imageSprit[0].copyTo(imageY);

	/*
	imageSprit.clear();
	cv::Mat image2xBicubic;
	cv::resize(image,image2xBicubic,image2x.size(),0,0,INTER_CUBIC);
	cv::imwrite("/home/wlamigo-data/Picture/test-cv-bcresize.png",image2xBicubic);
	image2xBicubic.convertTo(imageYUV,CV_32F, 1.0 / 255.0);
	cv::cvtColor(imageYUV, imageYUV, COLOR_RGB2YUV);
	cv::split(imageYUV, imageSprit);
	*/

	std::unique_ptr<std::vector<cv::Mat> > inputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>(1));
	std::unique_ptr<std::vector<cv::Mat> > outputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>(32));

	inputPlanes->clear();
	inputPlanes->push_back(imageY);

	/*
	cv::Mat test;
	imageY.copyTo(imageSprit[1]);
	imageY.copyTo(imageSprit[2]);
	cv::merge(imageSprit,test);
	test.convertTo(test,CV_8U,255.0);
	cv::imwrite("/home/wlamigo-data/Picture/test-cv.png",test);
	std::exit(0);
	*/

	for (int index = 0; index < models.size(); index++) {
		std::cout << "Iteration #" << (index + 1) << "..." << std::endl;
//		std::cout << models[index]->getNInputPlanes() << ","
//				<< models[index]->getNOutputPlanes() << std::endl;
		if(!models[index]->filter(*inputPlanes, *outputPlanes)){
			std::exit(-1);
		}
		std::cout << outputPlanes->size() << std::endl;
		if (index != models.size() - 1) {
			inputPlanes = std::move(outputPlanes);
			outputPlanes = std::unique_ptr<std::vector<cv::Mat> >(
					new std::vector<cv::Mat>(models[index + 1]->getNOutputPlanes()));
		}
	}

	outputPlanes->at(0).copyTo(imageSprit[0]);
	cv::Mat result;
	cv::merge(imageSprit,result);
	cv::cvtColor(result,result,COLOR_YUV2RGB);
	result.convertTo(result,CV_8U,255.0);
	cv::imwrite("/home/wlamigo-data/Picture/test-cv.png",result);

	return 0;
}

