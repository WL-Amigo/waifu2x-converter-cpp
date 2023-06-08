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

#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include "compiler.h"
#include "cvwrap.hpp"
#include "tchar.h"

#define ALIGN_UP(v,a) (((v+(a-1))/(a))*(a))

void pack_mat(float *out, std::vector<W2Mat> &inputPlanes, int w, int h, int nplane);

void unpack_mat(std::vector<W2Mat> &outputPlanes, const float *in, int w, int h, int nplane);

void unpack_mat1(W2Mat &outputMat, const float *in, int w, int h);

void pack_mat_rgb(float *out, W2Mat &inputPlane, int w, int h);

void pack_mat_rgb_f32(float *out, W2Mat &inputPlane, int w, int h);

void pack_mat_bgr(float *out, W2Mat &inputPlane, int w, int h);

void unpack_mat_rgb(W2Mat &outputMat, const float *in, int w, int h);

void unpack_mat_rgb_f32(W2Mat &outputMat, const float *in, int w, int h);

void unpack_mat_bgr(W2Mat &outputMat, const float *in, int w, int h);

/*
 * src is exist && dst is not exist                       : true
 * src is exist && dst is exist && dst is older than src  : true
 * otherwise                                              : false
 */
bool update_test(const TCHAR *dst_path, const TCHAR *src_path);
#endif
