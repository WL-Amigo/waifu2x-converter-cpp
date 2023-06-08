/*
* The MIT License (MIT)
* This file is part of waifu2x-converter-cpp
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include <limits.h>
#include "convertRoutine.hpp"
#include "common.hpp"
#include "Buffer.hpp"
#include "sec.hpp"

namespace w2xc
{

	// converting process inside program
	static bool convertWithModelsBasic
	(
		W2XConv *conv,
		ComputeEnv *env,
		W2Mat &inputPlane, W2Mat &outputPlane,
		Buffer *input_buf, Buffer *output_buf,
		std::vector<std::unique_ptr<Model> > &models,
		W2XConvFlopsCounter *flops,
		enum image_format fmt,
		int log_level
	);

	static bool convertWithModelsBlockSplit
	(
		W2XConv *conv,
		ComputeEnv *env,
		W2Mat &inputPlane,
		W2Mat &outputPlane, std::vector<std::unique_ptr<Model> > &models,
		W2XConvFlopsCounter *flops,
		int blockSize,
		enum image_format fmt,
		int log_level
	);

	bool convertWithModels
	(
		W2XConv *conv,
		ComputeEnv *env,
		W2Mat &inputPlane, W2Mat &outputPlane,
		std::vector<std::unique_ptr<Model> > &models,
		W2XConvFlopsCounter *flops,
		int blockSize,
		enum image_format fmt,
		int log_level)
	{
		return convertWithModelsBlockSplit(conv, env, inputPlane, outputPlane, models, flops, blockSize, fmt, log_level);
	}

	static bool convertWithModelsBasic
	(
		W2XConv *conv,
		ComputeEnv *env,
		W2Mat &inputPlane, W2Mat &outputPlane,
		Buffer *packed_input_buf,
		Buffer *packed_output_buf,
		std::vector<std::unique_ptr<Model> > &models, W2XConvFlopsCounter *flops,
		enum image_format fmt,
		int log_level
	)
	{
		// padding is require before calling this function

		std::vector<W2Mat> inputPlanes;
		inputPlanes.emplace_back(W2Mat(inputPlane,0,0,0,0));

		W2Size filterSize(inputPlane.view_width, inputPlane.view_height);
		int filterWidth = filterSize.width;
		int filterHeight = filterSize.height;

		float *packed_input = (float*)packed_input_buf->get_write_ptr_host(env);

		switch (fmt) {
			case IMAGE_BGR:
			{
				pack_mat_bgr(packed_input, inputPlane, filterWidth, filterHeight);
				break;
			}
			case IMAGE_RGB:
			{
				pack_mat_rgb(packed_input, inputPlane, filterWidth, filterHeight);
				break;
			}
			case IMAGE_RGB_F32:
			{
				pack_mat_rgb_f32(packed_input, inputPlane, filterWidth, filterHeight);
				break;
			}
			case IMAGE_Y:
			{
				pack_mat(packed_input, inputPlanes, filterWidth, filterHeight, 1);
				break;
			}
		}

		double t00 = getsec();
		double ops_sum = 0;

		for (int index = 0; index < (int)models.size(); index++)
		{
			int nOutputPlanes = models[index]->getNOutputPlanes();
			int nInputPlanes = models[index]->getNInputPlanes();

			if (log_level >= 4)
			{
				printf("Iteration #%d(%3d->%3d)...", (index + 1), nInputPlanes, nOutputPlanes);
			}
			
			double t0 = getsec();

			if (!models[index]->filter(conv, env, packed_input_buf, packed_output_buf, filterSize))
			{
				std::exit(-1);
			}
			
			double t1 = getsec();
			double ops = filterSize.width * filterSize.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;
			
			if (log_level >= 4)
			{
				double gflops = (ops/(1000.0*1000.0*1000.0)) / (t1-t0);
				double bytes = (double) filterSize.width * filterSize.height * sizeof(float) * (nOutputPlanes + nInputPlanes);
				double gigabytesPerSec = (bytes/(1000.0*1000.0*1000.0)) / (t1-t0);

				printf("(%.5f[s], %7.2f[GFLOPS], %8.3f[GB/s])\n", t1-t0, gflops, gigabytesPerSec);
			}
			
			ops_sum += ops;

			flops->flop += ops;
			flops->filter_sec += t1-t0;

			std::swap(packed_input_buf, packed_output_buf);
		}
		double t01 = getsec();

		if (IS_3CHANNEL(fmt))
		{
			packed_input = (float*)packed_input_buf->get_read_ptr_host(env, sizeof(float)*filterWidth*filterHeight*3);
		}
		else
		{
			packed_input = (float*)packed_input_buf->get_read_ptr_host(env, sizeof(float)*filterWidth*filterHeight);
		}

		switch (fmt) {
			case IMAGE_BGR:
			{
				outputPlane = W2Mat(filterSize.width*3, filterSize.height, 1);
				unpack_mat_bgr(outputPlane, packed_input, filterWidth, filterHeight);
				break;
			}
			case IMAGE_RGB:
			{
				outputPlane = W2Mat(filterSize.width*3, filterSize.height, 1);
				unpack_mat_rgb(outputPlane, packed_input, filterWidth, filterHeight);
				break;
			}
			case IMAGE_RGB_F32:
			{
				outputPlane = W2Mat(filterSize.width*3, filterSize.height, 4);
				unpack_mat_rgb_f32(outputPlane, packed_input, filterWidth, filterHeight);
				break;
			}
			case IMAGE_Y:
			{
				outputPlane = W2Mat(filterSize.width*1, filterSize.height, 4);
				unpack_mat1(outputPlane, packed_input, filterWidth, filterHeight);
				break;
			}
		}

		if (log_level >= 3)
		{
			double gflops = ops_sum/(1000.0*1000.0*1000.0) / (t01-t00);
			printf("total : %.3f[sec], %07.2f[GFLOPS]\n", t01-t00, gflops);
		}

		return true;
	}

	static bool convertWithModelsBlockSplit
	(
		W2XConv *conv,
		ComputeEnv *env,
		W2Mat &inputPlane_2,
		W2Mat &outputPlane_2,
		std::vector<std::unique_ptr<Model> > &models,
		W2XConvFlopsCounter *flops,
		int blockSize,
		enum image_format fmt,
		int log_level
	)
	{
		// padding is not required before calling this function
		// initialize local variables
		int nModel = (int) models.size();

		//insert padding to inputPlane
		int tempWidth = (int) inputPlane_2.view_width + nModel*2;
		int tempHeight = (int) inputPlane_2.view_height + nModel*2;
		int inputWidth = inputPlane_2.view_width;
		int inputHeight = inputPlane_2.view_height;

		W2Mat tempMat_2(tempWidth, tempHeight, inputPlane_2.type);
		int elem_size = CV_ELEM_SIZE(inputPlane_2.type);

		/* y border */
		for (int bi=0; bi<nModel; bi++)
		{
			char *dst;
			char *src;

			/* top */
			dst = tempMat_2.ptr<char>(bi) + elem_size * nModel;
			src = inputPlane_2.ptr<char>(0);
			memcpy(dst, src, inputWidth * elem_size);

			/* bottom */
			dst = tempMat_2.ptr<char>(inputHeight + nModel + bi) + elem_size * nModel;
			src = inputPlane_2.ptr<char>(inputHeight - 1);
			memcpy(dst, src, inputWidth * elem_size);
		}

		/* body */
		for (int bi=0; bi<inputHeight; bi++)
		{
			char *dst;
			char *src;
			dst = tempMat_2.ptr<char>(bi + nModel) + elem_size * nModel;
			src = inputPlane_2.ptr<char>(bi);
			memcpy(dst, src, inputWidth * elem_size);
		}

		/* x border */
		for (int bi=0; bi<tempHeight; bi++)
		{
			char *left = tempMat_2.ptr<char>(bi);
			char *right = left + elem_size * (nModel + inputWidth);
			uint32_t v32;
			uint32_t v_0, v_1, v_2;

			switch (elem_size)
			{
				case 1:
				{
					memset(left, left[nModel], nModel);
					memset(right, right[-1], nModel);
					break;
				}
				case 3:
				{
					v_0 = ((unsigned char*)left)[nModel*3+0];
					v_1 = ((unsigned char*)left)[nModel*3+1];
					v_2 = ((unsigned char*)left)[nModel*3+2];
					
					for (int xi=0; xi<nModel; xi++)
					{
						left[xi*3+0] = v_0;
						left[xi*3+1] = v_1;
						left[xi*3+2] = v_2;
					}

					v_0 = ((unsigned char*)right)[-3+0];
					v_1 = ((unsigned char*)right)[-3+1];
					v_2 = ((unsigned char*)right)[-3+2];
					
					for (int xi=0; xi<nModel; xi++)
					{
						right[xi*3+0] = v_0;
						right[xi*3+1] = v_1;
						right[xi*3+2] = v_2;
					}
					
					break;
				}
				case 4:
				{
					v32 = ((uint32_t*)left)[nModel];
					
					for (int xi=0; xi<nModel; xi++)
					{
						((uint32_t*)left)[xi] = v32;
					}
					
					v32 = ((uint32_t*)right)[-1];
					
					for (int xi=0; xi<nModel; xi++)
					{
						((uint32_t*)right)[xi] = v32;
					}
					
					break;
				}
				case 12:
				{
					v_0 = ((uint32_t*)left)[nModel*3+0];
					v_1 = ((uint32_t*)left)[nModel*3+1];
					v_2 = ((uint32_t*)left)[nModel*3+2];
					
					for (int xi=0; xi<nModel; xi++)
					{
						((uint32_t*)left)[xi*3+0] = v_0;
						((uint32_t*)left)[xi*3+1] = v_1;
						((uint32_t*)left)[xi*3+2] = v_2;
					}

					v_0 = ((uint32_t*)right)[-3+0];
					v_1 = ((uint32_t*)right)[-3+1];
					v_2 = ((uint32_t*)right)[-3+2];
					
					for (int xi=0; xi<nModel; xi++)
					{
						((uint32_t*)right)[xi*3+0] = v_0;
						((uint32_t*)right)[xi*3+1] = v_1;
						((uint32_t*)right)[xi*3+2] = v_2;
					}
					
					break;
				}
			}
		}

		if (blockSize == 0)
		{
			blockSize = env->pref_block_size;
		}

		Buffer *input_buf, *output_buf;
		int ok_count = 0;

		while (true)
		{
			long long max_size = 0;

			int width = (std::min)(tempMat_2.view_width, blockSize);
			int height = (std::min)(tempMat_2.view_height, blockSize);

			for (int index = 0; index < (int)models.size(); index++)
			{
				long long bufsize =
					(long long)sizeof(float) *
					(long long)width *
					(long long)height *
					(long long)models[index]->getNOutputPlanes();

				max_size = (std::max)(max_size, (long long)bufsize);
			}

			if ((sizeof(void*)==4) && max_size >= INT_MAX)
			{
				/* pass */
			}
			else
			{
				input_buf = new Buffer(env, max_size);
				output_buf = new Buffer(env, max_size);

				if (input_buf->prealloc(conv, env) && output_buf->prealloc(conv, env))
				{
					break;
				}

				delete input_buf;
				delete output_buf;
			}

			blockSize /= 2;
			
			//DEBUG printf("blockSize = %d\n", blockSize);
			
			if (blockSize == 0)
			{
				abort();
			}
		}

		int blockWidth = (std::min)(blockSize, tempMat_2.view_width);
		int blockHeight = (std::min)(blockSize, tempMat_2.view_height);
		int clipWidth = blockWidth - 2*nModel;
		int clipHeight = blockHeight - 2*nModel;

		//DEBUG printf("blockSize = %d\n", blockSize);

		// calcurate split rows/cols
		unsigned int splitColumns = (inputWidth + (clipWidth-1)) / clipWidth;
		unsigned int splitRows = (inputHeight + (clipHeight-1)) / clipHeight;

		switch (fmt)
		{
			case IMAGE_BGR:
			case IMAGE_RGB:
			{
				outputPlane_2 = W2Mat(inputWidth, inputHeight, CV_8UC3);
				break;
			}
			case IMAGE_RGB_F32:
			{
				outputPlane_2 = W2Mat(inputWidth, inputHeight, CV_32FC3);
				break;
			}
			case IMAGE_Y:
			{
				outputPlane_2 = W2Mat(inputWidth, inputHeight, CV_32FC1);
				break;
			}
			default:
			{
				abort();
			}
		}

		for (unsigned int r = 0; r < splitRows; r++)
		{
			int clipStartY = r * clipHeight;
			int clipEndY = 0;

			if (r == splitRows - 1)
			{
				clipEndY = tempMat_2.view_height;
			}
			else
			{
				clipEndY = r * clipHeight + blockHeight;
			}

			for (unsigned int c = 0; c < splitColumns; c++)
			{
				// start to convert
				W2Mat processBlockOutput;

				int clipStartX = c * clipWidth;
				int clipEndX = 0;

				if (c == splitColumns - 1)
				{
					clipEndX = tempMat_2.view_width;
				}
				else 
				{
					clipEndX = c * (blockWidth - 2 * nModel) + blockWidth;
				}

				int curBlockWidth = clipEndX - clipStartX;
				int curBlockHeight = clipEndY - clipStartY;
				
				W2Mat processBlock(tempMat_2, clipStartX, clipStartY, curBlockWidth, curBlockHeight);

				if (log_level >= 3)
				{
					printf("Processing block, column (%02d/%02d), row (%02d/%02d) ...\n", (c+1), splitColumns, (r+1), splitRows);
				}

				int elemSize = 0;

				switch (fmt)
				{
					case IMAGE_BGR:
					case IMAGE_RGB:
					{
						elemSize = 3;
						break;
					}
					case IMAGE_RGB_F32:
					{
						elemSize = 12;
						break;
					}
					case IMAGE_Y:
					{
						elemSize = 4;
						break;
					}
					//FutureNote: no default(-break) ?
				}

				if (!convertWithModelsBasic
					(
						conv,
						env,
						processBlock,
						processBlockOutput,
						input_buf,
						output_buf,
						models,
						flops,
						fmt,
						log_level
					)
				)
				{
					std::cerr <<
						"w2xc::convertWithModelsBasic()\nin w2xc::convertWithModelsBlockSplit() : \n something error has occured. stop."
						<< std::endl;
						
					return false;
				}

				int srcStartY = nModel;
				int srcStartX = nModel;

				int dstStartY = r * (blockHeight - 2*nModel);
				int dstStartX = c * (blockWidth - 2*nModel);
				int copyWidth = curBlockWidth - (nModel * 2);
				int copyHeight = curBlockHeight - (nModel * 2);

				for (int yi=0; yi<copyHeight; yi++)
				{
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

