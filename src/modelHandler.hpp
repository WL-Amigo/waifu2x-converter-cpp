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

#ifndef MODEL_HANDLER_HPP_
#define MODEL_HANDLER_HPP_

#include <iostream>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include "picojson.h"
#include "Buffer.hpp"
#include "filters.hpp"
#include "w2xconv.h"
#include "cvwrap.hpp"
#include "tstring.hpp"

namespace w2xc
{

	class Model
	{
		private:
			int nInputPlanes;
			int nOutputPlanes;
			std::vector<W2Mat> weights;
			std::vector<double> biases;
			int kernelSize;

			Model() {}; // cannot use no-argument constructor

			// class inside operation function
			bool loadModelFromJSONObject(picojson::object& jsonObj);

			// thread worker function
			bool filterWorker
			(
				std::vector<W2Mat> &inputPlanes,
				std::vector<W2Mat> &weightMatrices,
				std::vector<W2Mat> &outputPlanes,
				unsigned int beginningIndex,
				unsigned int nWorks
			);

			bool filter_CV(
				ComputeEnv *env,
				Buffer *packed_input,
				Buffer *packed_output,
				const W2Size &size
			);

			bool filter_AVX_OpenCL(
				W2XConv *conv,
				ComputeEnv *env,
				Buffer *packed_input,
				Buffer *packed_output,
				const W2Size &size
			);

		public:
			// ctor and dtor
			Model(picojson::object &jsonObj)
			{
				// preload nInputPlanes,nOutputPlanes, and preserve required size vector
				nInputPlanes = static_cast<int>(jsonObj["nInputPlane"].get<double>());
				nOutputPlanes = static_cast<int>(jsonObj["nOutputPlane"].get<double>());

				if ((kernelSize = static_cast<int>(jsonObj["kW"].get<double>()))
					!= static_cast<int>(jsonObj["kH"].get<double>()))
				{
					std::cerr <<
						"Error : Model-Constructor : \nkernel in model is not square.\nstop." <<
						std::endl;
					std::exit(-1);
				} // kH == kW

				biases = std::vector<double>(nOutputPlanes, 0.0);

				if (!loadModelFromJSONObject(jsonObj))
				{
					std::cerr <<
						"Error : Model-Constructor : \nsomething error has been occured in loading model from JSON-Object.\nstop." <<
						std::endl;
					std::exit(-1);
				}
			}
			Model(FILE *binfp);
			Model(
				int nInputPlane,
				int nOutputPlane,
				const float *coef_list,
				const float *bias
			);

			~Model() {}

			// for debugging
			void printWeightMatrix();
			void printBiases();

			// getter function
			int getNInputPlanes();
			int getNOutputPlanes();

			std::vector<W2Mat> &getWeigts()
			{
				return weights;
			}
			std::vector<double> &getBiases()
			{
				return biases;
			}
			// setter function

			// public operation function
			bool filter
			(
				W2XConv *conv,
				ComputeEnv *env,
				Buffer *packed_input,
				Buffer *packed_output,
				const W2Size &size
			);
	};

	class modelUtility
	{
		private:
			static modelUtility* instance;
			int nJob;
			modelUtility() : nJob(4) {};
		public:
			static bool generateModelFromJSON
			(
				const _tstring &fileName,
				std::vector<std::unique_ptr<Model> > &models
			);
			static void generateModelFromMEM
			(
				int layer_depth,
				int num_input_plane,
				const int *num_map, // num_map[layer_depth]
				const float *coef_list, // coef_list[layer_depth][num_map][3x3]
				const float *bias, // bias[layer_depth][num_map]
				std::vector<std::unique_ptr<Model> > &models
			);

			static modelUtility& getInstance();
			bool setNumberOfJobs(int setNJob);
			int getNumberOfJobs();
	};
}

#endif /* MODEL_HANDLER_HPP_ */
