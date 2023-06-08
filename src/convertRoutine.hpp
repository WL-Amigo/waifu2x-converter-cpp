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

#ifndef CONVERTROUTINE_HPP_
#define CONVERTROUTINE_HPP_

#include "modelHandler.hpp"
#include "common.hpp"
#include "w2xconv.h"
#include "cvwrap.hpp"
#include <memory>
#include <vector>

namespace w2xc
{

	enum image_format
	{
		IMAGE_BGR,
		IMAGE_RGB,
		IMAGE_RGB_F32,
		IMAGE_Y
	};

#define IS_3CHANNEL(f) (((f)==w2xc::IMAGE_BGR) || ((f)==w2xc::IMAGE_RGB) || ((f)==w2xc::IMAGE_RGB_F32))

/**
 * convert inputPlane to outputPlane by convoluting with models.
 */
	bool convertWithModels
	(
		W2XConv *conv,
		ComputeEnv *env,
		W2Mat &inputPlanes,
		W2Mat &outputPlanes,
		std::vector<std::unique_ptr<Model> > &models,
		W2XConvFlopsCounter *flops,
		int blockSize,
		enum image_format fmt,
		int log_level
	);
}

#endif /* CONVERTROUTINE_HPP_ */
