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

#ifndef W2XC_ENV_HPP
#define W2XC_ENV_HPP

#include "w2xconv.h"

struct OpenCLDev;
struct CUDADev;

namespace w2xc {
	struct ThreadPool;
}

struct ComputeEnv
{
    int num_cl_dev;
    int num_cuda_dev;
    OpenCLDev *cl_dev_list;
    CUDADev *cuda_dev_list;
    double transfer_wait;

    unsigned int pref_block_size;

#if defined(_WIN32) || defined(__linux)
    w2xc::ThreadPool *tpool;
#endif
    ComputeEnv();
};

extern void clearError(W2XConv *conv);

#endif
