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

#ifndef W2MAT_HPP
#define W2MAT_HPP

#include <utility>
#include <vector>


#ifdef HAVE_OPENCV
	#include <opencv2/opencv.hpp>
#else
	#define CV_32FC3 12
	#define CV_32FC1 4
	#define CV_8UC3 3
	#define CV_8UC1 1
	#define CV_ELEM_SIZE(type) (type)
#endif

class W2Mat {
	public:
		bool data_owner;

		char *data;
		int data_byte_width;
		int data_height;

		int view_top;
		int view_left;
		int view_width;
		int view_height;

		int type;

		W2Mat(int data_width, int data_height, int type);
		W2Mat(int data_width, int data_height, int type, void *data, int data_step);
		W2Mat(const W2Mat &rhs, int view_left_offset, int view_top_offset, int view_width, int view_height);
		W2Mat();

		W2Mat(const W2Mat &) = delete;
		W2Mat& operator=(const W2Mat&) = delete;

		W2Mat& operator= (W2Mat &&);
		W2Mat(W2Mat &&rhs)
		{
			*this = std::move(rhs);
		}

		~W2Mat();

		void copyTo(W2Mat*);
			
#ifdef HAVE_OPENCV
		W2Mat(cv::Mat &);
		void to_cvmat(cv::Mat *);
#endif

		template<typename T> T *ptr(int yi);
		template<typename T> T &at(int y, int x)
		{
			return this->ptr<T>(y)[x];
		}
};

struct W2Size
{
    int width, height;

    W2Size(int w, int h) : width(w), height(h) {}
};

#ifdef HAVE_OPENCV

void extract_viewlist_from_cvmat(std::vector<W2Mat> &list, std::vector<cv::Mat> &cvmat);
void extract_viewlist_to_cvmat(std::vector<cv::Mat> &cvmat, std::vector<W2Mat> &list);
	
#endif

template<typename T> T * W2Mat::ptr(int yi)
{
    int off = 0;
    int elem_size = CV_ELEM_SIZE(this->type);

    off += (yi+view_top) * data_byte_width;
    off += view_left * elem_size;

    return (T*)(data + off);
}

#endif