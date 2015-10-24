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
static bool convertWithModelsBasic(W2XConv *conv,
				   ComputeEnv *env,
				   W2Mat &inputPlane, W2Mat &outputPlane,
				   Buffer *input_buf, Buffer *output_buf,
				   std::vector<std::unique_ptr<Model> > &models,
				   W2XConvFlopsCounter *flops,
				   enum image_format fmt,
				   bool enableLog);
static bool convertWithModelsBlockSplit(W2XConv *conv,
					ComputeEnv *env,
					W2Mat &inputPlane,
					W2Mat &outputPlane, std::vector<std::unique_ptr<Model> > &models,
					W2XConvFlopsCounter *flops,
					int blockSize,
					enum image_format fmt,
					bool enableLog);

bool convertWithModels(W2XConv *conv,
		       ComputeEnv *env,
		       W2Mat &inputPlane, W2Mat &outputPlane,
		       std::vector<std::unique_ptr<Model> > &models,
		       W2XConvFlopsCounter *flops,
		       int blockSize,
		       enum image_format fmt,
		       bool enableLog)
{
	return convertWithModelsBlockSplit(conv, env,
					   inputPlane, outputPlane,
					   models, flops, blockSize, fmt, enableLog);
}

static bool convertWithModelsBasic(W2XConv *conv,
				   ComputeEnv *env,
				   W2Mat &inputPlane, W2Mat &outputPlane,
				   Buffer *packed_input_buf,
				   Buffer *packed_output_buf,
				   std::vector<std::unique_ptr<Model> > &models, W2XConvFlopsCounter *flops,
				   enum image_format fmt,
				   bool enableLog)
{
	// padding is require before calling this function

	std::vector<W2Mat> inputPlanes;
	inputPlanes.emplace_back(W2Mat::clip_view(inputPlane,0,0,0,0));

	W2Size filterSize(inputPlane.view_width, inputPlane.view_height);
	int filterWidth = filterSize.width;
	int filterHeight = filterSize.height;

	float *packed_input = (float*)packed_input_buf->get_write_ptr_host(env);

	switch (fmt) {
	case IMAGE_BGR:
		pack_mat_bgr(packed_input, inputPlane, filterWidth, filterHeight);
		break;
	case IMAGE_RGB:
		pack_mat_rgb(packed_input, inputPlane, filterWidth, filterHeight);
		break;
	case IMAGE_RGB_F32:
		pack_mat_rgb_f32(packed_input, inputPlane, filterWidth, filterHeight);
		break;
	case IMAGE_Y:
		pack_mat(packed_input, inputPlanes, filterWidth, filterHeight, 1);
		break;
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
		if (!models[index]->filter(conv, env, packed_input_buf, packed_output_buf, filterSize)) {
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

	if (IS_3CHANNEL(fmt)) {
		packed_input = (float*)packed_input_buf->get_read_ptr_host(env, sizeof(float)*filterWidth*filterHeight*3);
	} else {
		packed_input = (float*)packed_input_buf->get_read_ptr_host(env, sizeof(float)*filterWidth*filterHeight);
	}

	switch (fmt) {
	case IMAGE_BGR:
		outputPlane = W2Mat(filterSize.width*3, filterSize.height, 1);
		unpack_mat_bgr(outputPlane, packed_input, filterWidth, filterHeight);
		break;
	case IMAGE_RGB:
		outputPlane = W2Mat(filterSize.width*3, filterSize.height, 1);
		unpack_mat_rgb(outputPlane, packed_input, filterWidth, filterHeight);
		break;
	case IMAGE_RGB_F32:
		outputPlane = W2Mat(filterSize.width*3, filterSize.height, 4);
		unpack_mat_rgb_f32(outputPlane, packed_input, filterWidth, filterHeight);
		break;
	case IMAGE_Y:
		outputPlane = W2Mat(filterSize.width*1, filterSize.height, 4);
		unpack_mat1(outputPlane, packed_input, filterWidth, filterHeight);
		break;
	}

	if (enableLog) {
		double gflops = ops_sum/(1000.0*1000.0*1000.0) / (t01-t00);
		std::cout << "total : " << (t01-t00) << "[sec], " << gflops << "[GFLOPS]" << std::endl;
	}

	return true;

}

static bool convertWithModelsBlockSplit(W2XConv *conv,
					ComputeEnv *env,
					W2Mat &inputPlane_2,
					W2Mat &outputPlane_2,
					std::vector<std::unique_ptr<Model> > &models,
					W2XConvFlopsCounter *flops,
					int blockSize,
					enum image_format fmt,
					bool enableLog)
{
	cv::Mat inputPlane(extract_view_to_cvmat(inputPlane_2));

	// padding is not required before calling this function

	// initialize local variables
	unsigned int nModel = models.size();

	//insert padding to inputPlane
	cv::Mat tempMat;
	cv::Size outputSize = inputPlane.size();
	cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel,
			cv::BORDER_REPLICATE);

	// start to convert
	cv::Mat processRow;
	cv::Mat processBlock;
	W2Mat processBlockOutput;
	cv::Mat writeMatTo;
	cv::Mat writeMatFrom;

	if (blockSize == 0) {
		blockSize = env->pref_block_size;
	}

	Buffer *input_buf, *output_buf;
	int ok_count = 0;

	while (1) {
		long long max_size = 0;

		int width = (std::min)(tempMat.size().width, blockSize);
		int height = (std::min)(tempMat.size().height, blockSize);

		for (int index = 0; index < (int)models.size(); index++) {
			long long bufsize =
				(long long)sizeof(float) *
				(long long)width *
				(long long)height *
				(long long)models[index]->getNOutputPlanes();

			max_size = (std::max)(max_size, (long long)bufsize);
		}

		if ((sizeof(void*)==4) && max_size >= INT_MAX) {
			/* pass */
		} else {
			input_buf = new Buffer(env, max_size);
			output_buf = new Buffer(env, max_size);

			if (input_buf->prealloc(conv, env) &&
			    output_buf->prealloc(conv, env))
			{
				break;
			}

			delete input_buf;
			delete output_buf;
		}

		blockSize /= 2;
		//printf("blockSize = %d\n", blockSize);
		if (blockSize == 0) {
			abort();
		}
	}

	int blockWidth = (std::min)(blockSize, tempMat.size().width);
	int blockHeight = (std::min)(blockSize, tempMat.size().height);

	//printf("blockSize = %d\n", blockSize);

	// calcurate split rows/cols
	unsigned int splitColumns = static_cast<unsigned int>(std::ceil(
			static_cast<float>(outputSize.width)
					/ static_cast<float>(blockWidth - 2 * nModel)));
	unsigned int splitRows = static_cast<unsigned int>(std::ceil(
			static_cast<float>(outputSize.height)
					/ static_cast<float>(blockHeight - 2 * nModel)));

	switch (fmt) {
	case IMAGE_BGR:
	case IMAGE_RGB:
		outputPlane_2 = W2Mat(outputSize.width, outputSize.height, CV_8UC3);
		break;

	case IMAGE_RGB_F32:
		outputPlane_2 = W2Mat(outputSize.width, outputSize.height, CV_32FC3);
		break;

	case IMAGE_Y:
		outputPlane_2 = W2Mat(outputSize.width, outputSize.height, CV_32FC1);
		break;

	default:
		abort();
	}

	for (unsigned int r = 0; r < splitRows; r++) {
		if (r == splitRows - 1) {
			processRow = tempMat.rowRange(r * (blockHeight - 2 * nModel),
					tempMat.size().height);
		} else {
			processRow = tempMat.rowRange(r * (blockHeight - 2 * nModel),
					r * (blockHeight - 2 * nModel) + blockHeight);
		}
		for (unsigned int c = 0; c < splitColumns; c++) {
			if (c == splitColumns - 1) {
				processBlock = processRow.colRange(
						c * (blockWidth- 2 * nModel),
						tempMat.size().width);
			} else {
				processBlock = processRow.colRange(
						c * (blockWidth - 2 * nModel),
						c * (blockWidth - 2 * nModel) + blockWidth);
			}

			int curBlockWidth = processBlock.size().width;
			int curBlockHeight = processBlock.size().height;

			if (enableLog) {
				std::cout << "start process block (" << c << "," << r << ") ..."
					  << std::endl;
			}

			int elemSize = 0;

			switch (fmt) {
			case IMAGE_BGR:
			case IMAGE_RGB:
				elemSize = 3;
				processBlockOutput = W2Mat(curBlockWidth, curBlockHeight, CV_8UC3);
				break;

			case IMAGE_RGB_F32:
				elemSize = 12;
				processBlockOutput = W2Mat(curBlockWidth, curBlockHeight, CV_32FC3);
				break;

			case IMAGE_Y:
				elemSize = 4;
				processBlockOutput = W2Mat(curBlockWidth, curBlockHeight, CV_32FC1);
				break;
			}

			W2Mat processBlock2 = extract_view_from_cvmat(processBlock);

			if (!convertWithModelsBasic(conv, env,
						    processBlock2, processBlockOutput,
						    input_buf, output_buf,
						    models, flops, fmt, enableLog)) {
				std::cerr << "w2xc::convertWithModelsBasic()\n"
					"in w2xc::convertWithModelsBlockSplit() : \n"
					"something error has occured. stop." << std::endl;
				return false;
			}

			int srcStartY = nModel;
			int srcStartX = nModel;

			int dstStartY = r * (blockHeight - 2*nModel);
			int dstStartX = c * (blockWidth - 2*nModel);
			int copyWidth = curBlockWidth - (nModel * 2);
			int copyHeight = curBlockHeight - (nModel * 2);

			for (int yi=0; yi<copyHeight; yi++) {
				char *src = processBlockOutput.ptr<char>(yi + srcStartY);
				char *dst = outputPlane_2.ptr<char>(yi + dstStartY);

				src += srcStartX * elemSize;
				dst += dstStartX * elemSize;

				memcpy(dst, src, copyWidth * elemSize);
			}
		} // end process 1 column

	} // end process all blocks

	delete input_buf;
	delete output_buf;

	return true;

}

}

