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

#include <stdlib.h>
#include <string.h>
#include "cvwrap.hpp"

W2Mat::~W2Mat()
{
	if (data_owner)
	{
		data_owner=false;
		free(data);
		data = nullptr;
	}
}


W2Mat::W2Mat()
	: data_owner(false),
	data(nullptr),
	data_byte_width(0),
	data_height(0),
	view_top(0),
	view_left(0),
	view_width(0),
	view_height(0)
{
}

W2Mat::W2Mat(int width, int height, int type)
	: data_owner(true),
	data_byte_width(width*CV_ELEM_SIZE(type)),
	data_height(height),
	view_top(0),
	view_left(0),
	view_width(width),
	view_height(height),
	type(type)
{
	this->data = (char*)calloc(height, data_byte_width);
}

W2Mat::W2Mat(int width, int height, int type, void *data, int data_step)
	: data_owner(true),
	data_byte_width(data_step),
	data_height(height),
	view_top(0),
	view_left(0),
	view_width(width),
	view_height(height),
	type(type)
{
	this->data = (char*)calloc(height, data_byte_width);
	memcpy(this->data, data, height * data_byte_width);
}


W2Mat & W2Mat::operator=(W2Mat &&rhs)
{
    this->data_owner = rhs.data_owner;
    this->data = rhs.data;
    this->data_byte_width = rhs.data_byte_width;
    this->data_height = rhs.data_height;
    this->view_top = rhs.view_top;
    this->view_left = rhs.view_left;
    this->view_width = rhs.view_width;
    this->view_height= rhs.view_height;
    this->type = rhs.type;

    rhs.data_owner = false;
    rhs.data = NULL;

    return *this;
}

W2Mat::W2Mat(const W2Mat & rhs, int view_left_offset, int view_top_offset, int view_width, int view_height)
{	
    this->data_owner = false;
    this->data = rhs.data;
    this->data_byte_width = rhs.data_byte_width;
    this->data_height = rhs.data_height;

    this->view_left = rhs.view_left + view_left_offset;
    this->view_top = rhs.view_top + view_top_offset;
    this->view_width = view_width;
    this->view_height = view_height;

    this->type = rhs.type;
}

void W2Mat::copyTo(W2Mat* target)
{
	W2Mat ret(this->view_width, this->view_height, this->type);

	int elem_size = CV_ELEM_SIZE(this->type);
	int w = this->view_width;
	int h = this->view_height;

	for (int yi = 0; yi < h; yi++)
	{
		char *out = ret.ptr<char>(yi);
		char *in = this->ptr<char>(yi);

		memcpy(out, in, w * elem_size);
	}
	*target = std::move(ret);
}

#ifdef HAVE_OPENCV
W2Mat::W2Mat(cv::Mat &m) : data_owner(true), view_top(0), view_left(0)
{
	int w = m.size().width;
	int h = m.size().height;
	
	this->data_byte_width = w * CV_ELEM_SIZE(m.type());
	this->data_height = h;
	this->view_width = w;
	this->view_height = h;
	this->type = m.type();
	this->data = (char*)calloc(h, this->data_byte_width);


	for (int yi = 0; yi < h; yi++)
	{
		void *in = m.ptr(yi);
		char *out = this->ptr<char>(yi);

		memcpy(out, in, this->data_byte_width);
	}
}

void W2Mat::to_cvmat(cv::Mat *target)
{
	int w = this->view_width;
	int h = this->view_height;

	cv::Mat ret = cv::Mat::zeros(cv::Size(w, h), this->type);
	int elem_size = CV_ELEM_SIZE(this->type);

	for (int yi = 0; yi < h; yi++)
	{
		void *out = ret.ptr(yi);
		void *in = this->ptr<char>(yi);

		memcpy(out, in, w * elem_size);
	}
	*target = ret.clone();
}

// DO NOT USE IN NORMAL SITUAYION. Return wm is not own their data.
void extract_view_from_cvmat(W2Mat &wm, cv::Mat &m)
{		
	wm.data_owner = false;
	wm.data = (char*) m.data;
	wm.data_byte_width = (int) m.step;
	wm.data_height = m.size().height;
	
	wm.view_top = 0;
	wm.view_left = 0;
	wm.view_width = m.size().width;
	wm.view_height = m.size().height;
	wm.type = m.type();

}

void extract_view_to_cvmat(cv::Mat &m, W2Mat &wm)
{	
	int w = wm.view_width;
	int h = wm.view_height;
	
	char *data = (char*) wm.data;
	
	int byte_offset = 0;
	byte_offset += wm.view_top * wm.data_byte_width;
	byte_offset += wm.view_left * CV_ELEM_SIZE(wm.type);
	
	cv::Mat ret(cv::Size(w,h), wm.type, data + byte_offset, wm.data_byte_width);
	
	ret.copyTo(m);
}

// DO NOT USE IN NORMAL SITUAYION. Return wm is not own their data.
void extract_view_from_cvmat_offset
(
	W2Mat &wm,
	cv::Mat &m,
	int view_left_offset,
	int view_top_offset,
	int view_width,
	int view_height
)
{	
	extract_view_from_cvmat(wm, m);
	
	wm.view_top = view_top_offset;
	wm.view_left = view_left_offset;
	wm.view_width = view_width;
	wm.view_height = view_height;
}

void extract_viewlist_from_cvmat(std::vector<W2Mat> &list, std::vector<cv::Mat> &cvmat)
{
	std::for_each(cvmat.begin(), cvmat.end(),
		[&list](cv::Mat &cv) {
		W2Mat w2;
		extract_view_from_cvmat(w2, cv);
		list.push_back(std::move(w2));
	});
}

void extract_viewlist_to_cvmat(std::vector<cv::Mat> &cvmat, std::vector<W2Mat> &list)
{
	std::for_each(list.begin(), list.end(),
		[&cvmat](W2Mat &wm) {
		cv::Mat cv;
		extract_view_to_cvmat(cv, wm);
		cvmat.push_back(cv.clone());
	});
}

#endif
