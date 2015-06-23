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
#include "common.hpp"
#include "Buffer.hpp"
#include "sec.hpp"

namespace w2xc {

// converting process inside program
static bool convertWithModelsBasic(ComputeEnv *env,
				   cv::Mat &inputPlane, cv::Mat &outputPlane,
				   Buffer *input_buf, Buffer *output_buf,
				   std::vector<std::unique_ptr<Model> > &models,
				   W2XConvFlopsCounter *flops,
				   bool is_rgb,
				   bool enableLog);
static bool convertWithModelsBlockSplit(ComputeEnv *env,
					cv::Mat &inputPlane,
					cv::Mat &outputPlane, std::vector<std::unique_ptr<Model> > &models,
					W2XConvFlopsCounter *flops,
					cv::Size blockSize,
					bool is_rgb,
					bool enableLog);

bool convertWithModels(ComputeEnv *env,
		       cv::Mat &inputPlane, cv::Mat &outputPlane,
		       std::vector<std::unique_ptr<Model> > &models,
		       W2XConvFlopsCounter *flops,
		       cv::Size blockSize,
		       bool is_rgb,
		       bool enableLog)
{
	bool requireSplitting = (inputPlane.size().width * inputPlane.size().height)
			> blockSize.width * blockSize.height * 3 / 2;
	if (requireSplitting) {
		return convertWithModelsBlockSplit(env, inputPlane, outputPlane, models, flops, blockSize, is_rgb, enableLog);
	} else {
		//insert padding to inputPlane
		cv::Mat tempMat;
		int nModel = models.size();
		cv::Size outputSize = inputPlane.size();
		cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel,
				cv::BORDER_REPLICATE);

		size_t max_size = 0;
		int width = tempMat.size().width;
		int height = tempMat.size().height;

		for (int index = 0; index < (int)models.size(); index++) {
			size_t bufsize = sizeof(float) * width * height *
				models[index]->getNOutputPlanes();

			max_size = (std::max)(max_size, bufsize);
		}

		Buffer *input_buf = new Buffer(env, max_size);
		Buffer *output_buf = new Buffer(env, max_size);

		bool ret = convertWithModelsBasic(env, tempMat, outputPlane,
						  input_buf, output_buf,
						  models, flops, is_rgb, enableLog);

		tempMat = outputPlane(cv::Range(nModel, outputSize.height + nModel),
				cv::Range(nModel, outputSize.width + nModel));
		assert(
				tempMat.size().width == outputSize.width
						&& tempMat.size().height == outputSize.height);

		tempMat.copyTo(outputPlane);

		delete input_buf;
		delete output_buf;

		return ret;
	}

}

static bool convertWithModelsBasic(ComputeEnv *env,
				   cv::Mat &inputPlane, cv::Mat &outputPlane,
				   Buffer *packed_input_buf,
				   Buffer *packed_output_buf,
				   std::vector<std::unique_ptr<Model> > &models, W2XConvFlopsCounter *flops,
				   bool is_rgb,
				   bool enableLog)
{
	// padding is require before calling this function

	std::unique_ptr<std::vector<cv::Mat> > inputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>());
	std::unique_ptr<std::vector<cv::Mat> > outputPlanes = std::unique_ptr<
			std::vector<cv::Mat> >(new std::vector<cv::Mat>());

	inputPlanes->clear();
	inputPlanes->push_back(inputPlane);

	cv::Size filterSize = inputPlane.size();
	int filterWidth = filterSize.width;
	int filterHeight = filterSize.height;

	float *packed_input = (float*)packed_input_buf->get_write_ptr_host(env);

	if (is_rgb) {
		pack_mat_rgb(packed_input, inputPlane, filterWidth, filterHeight);
	} else {
		pack_mat(packed_input, *inputPlanes, filterWidth, filterHeight, 1);
	}

	double t00 = getsec();
	double ops_sum = 0;

	for (int index = 0; index < (int)models.size(); index++) {
		int nOutputPlanes = models[index]->getNOutputPlanes();
		int nInputPlanes = models[index]->getNInputPlanes();

		if (enableLog) {
			std::cout << "Iteration #" << (index + 1) << "(" << nInputPlanes << "->" << nOutputPlanes << ")..." ;
		}
		double t0 = getsec();
		if (!models[index]->filter(env, packed_input_buf, packed_output_buf, filterSize)) {
			std::exit(-1);
		}
		double t1 = getsec();
		double ops = filterSize.width * filterSize.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
		double gflops = (ops/(1000.0*1000.0*1000.0)) / (t1-t0);
		double bytes = filterSize.width * filterSize.height * sizeof(float) * (nOutputPlanes + nInputPlanes);
		double GBs = (bytes/(1000.0*1000.0*1000.0)) / (t1-t0);

		if (enableLog) {
			std::cout << "(" << (t1-t0)*1000 << "[ms], " << gflops << "[GFLOPS], " << GBs << "[GB/s])" << std::endl;
		}
		ops_sum += ops;

		flops->flop += ops;
		flops->filter_sec += t1-t0;

		std::swap(packed_input_buf, packed_output_buf);
	}
	double t01 = getsec();

	if (is_rgb) {
		packed_input = (float*)packed_input_buf->get_read_ptr_host(env, sizeof(float)*filterWidth*filterHeight*3);
	} else {
		packed_input = (float*)packed_input_buf->get_read_ptr_host(env, sizeof(float)*filterWidth*filterHeight);
	}

	if (is_rgb) {
		outputPlane = cv::Mat::zeros(filterSize, CV_8UC3);
		unpack_mat_rgb(outputPlane, packed_input, filterWidth, filterHeight);
	} else {
		outputPlane = cv::Mat::zeros(filterSize, CV_32FC1);
		unpack_mat1(outputPlane, packed_input, filterWidth, filterHeight);
	}

	if (enableLog) {
		double gflops = ops_sum/(1000.0*1000.0*1000.0) / (t01-t00);
		std::cout << "total : " << (t01-t00) << "[sec], " << gflops << "[GFLOPS]" << std::endl;
	}

	return true;

}

static bool convertWithModelsBlockSplit(ComputeEnv *env,
					cv::Mat &inputPlane,
					cv::Mat &outputPlane,
					std::vector<std::unique_ptr<Model> > &models,
					W2XConvFlopsCounter *flops,
					cv::Size blockSize,
					bool is_rgb,
					bool enableLog)
{

	// padding is not required before calling this function

	// initialize local variables
	unsigned int nModel = models.size();

	//insert padding to inputPlane
	cv::Mat tempMat;
	cv::Size outputSize = inputPlane.size();
	cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel,
			cv::BORDER_REPLICATE);

	// calcurate split rows/cols
	unsigned int splitColumns = static_cast<unsigned int>(std::ceil(
			static_cast<float>(outputSize.width)
					/ static_cast<float>(blockSize.width - 2 * nModel)));
	unsigned int splitRows = static_cast<unsigned int>(std::ceil(
			static_cast<float>(outputSize.height)
					/ static_cast<float>(blockSize.height - 2 * nModel)));

	// start to convert
	cv::Mat processRow;
	cv::Mat processBlock;
	cv::Mat processBlockOutput;
	cv::Mat writeMatTo;
	cv::Mat writeMatFrom;

	size_t max_size = 0;
	int width = (std::min)(tempMat.size().width, blockSize.width);
	int height = (std::min)(tempMat.size().height, blockSize.height);

	for (int index = 0; index < (int)models.size(); index++) {
		size_t bufsize = sizeof(float) * width * height *
			models[index]->getNOutputPlanes();

		max_size = (std::max)(max_size, bufsize);
	}

	Buffer *input_buf = new Buffer(env, max_size);
	Buffer *output_buf = new Buffer(env, max_size);

	if (is_rgb) {
		outputPlane = cv::Mat::zeros(outputSize, CV_8UC3);
	} else {
		outputPlane = cv::Mat::zeros(outputSize, CV_32FC1);
	}
	for (unsigned int r = 0; r < splitRows; r++) {
		if (r == splitRows - 1) {
			processRow = tempMat.rowRange(r * (blockSize.height - 2 * nModel),
					tempMat.size().height);
		} else {
			processRow = tempMat.rowRange(r * (blockSize.height - 2 * nModel),
					r * (blockSize.height - 2 * nModel) + blockSize.height);
		}
		for (unsigned int c = 0; c < splitColumns; c++) {
			if (c == splitColumns - 1) {
				processBlock = processRow.colRange(
						c * (blockSize.width - 2 * nModel),
						tempMat.size().width);
			} else {
				processBlock = processRow.colRange(
						c * (blockSize.width - 2 * nModel),
						c * (blockSize.width - 2 * nModel) + blockSize.width);
			}

			if (enableLog) {
				std::cout << "start process block (" << c << "," << r << ") ..."
					  << std::endl;
			}
			if (!convertWithModelsBasic(env,
						    processBlock, processBlockOutput,
						    input_buf, output_buf,
						    models, flops, is_rgb, enableLog)) {
				std::cerr << "w2xc::convertWithModelsBasic()\n"
						"in w2xc::convertWithModelsBlockSplit() : \n"
						"something error has occured. stop." << std::endl;
				return false;
			}

			writeMatFrom = processBlockOutput(
					cv::Range(nModel,
							processBlockOutput.size().height - nModel),
					cv::Range(nModel,
							processBlockOutput.size().width - nModel));
			writeMatTo = outputPlane(
					cv::Range(r * (blockSize.height - 2 * nModel),
							r * (blockSize.height - 2 * nModel)
									+ processBlockOutput.size().height
									- 2 * nModel),
					cv::Range(c * (blockSize.height - 2 * nModel),
							c * (blockSize.height - 2 * nModel)
									+ processBlockOutput.size().width
									- 2 * nModel));
			assert(
					writeMatTo.size().height == writeMatFrom.size().height
							&& writeMatTo.size().width
									== writeMatFrom.size().width);
			writeMatFrom.copyTo(writeMatTo);

		} // end process 1 column

	} // end process all blocks

	delete input_buf;
	delete output_buf;

	return true;

}

}

