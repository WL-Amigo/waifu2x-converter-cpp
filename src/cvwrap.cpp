#include <stdlib.h>
#include <string.h>
#include "cvwrap.hpp"

W2Mat::~W2Mat()
{
	if (data_owner) {
		data_owner=false;
		free(data);
		data = nullptr;
	}
}


W2Mat::W2Mat()
	:data_owner(false),
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
	:data_owner(true),
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
    :data_owner(true),
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


W2Mat &
W2Mat::operator=(W2Mat &&rhs)
{
	if(this->data_owner){
		free(data);
	}
	
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
    rhs.data = nullptr;

    return *this;
}

W2Mat::W2Mat(const W2Mat & rhs,
                 int view_left_offset, int view_top_offset,
                 int view_width, int view_height)
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

void
W2Mat::copy_full(W2Mat & target, W2Mat & rhs)
{
	target = W2Mat(rhs.view_width,
		rhs.view_height,
		rhs.type);

	int elem_size = CV_ELEM_SIZE(rhs.type);
	int w = rhs.view_width;
	int h = rhs.view_height;

	for (int yi = 0; yi < h; yi++) {
		char *out = target.ptr<char>(yi);
		char *in = rhs.ptr<char>(yi);

		memcpy(out, in, w * elem_size);
	}
}

#ifdef HAVE_OPENCV
W2Mat::W2Mat(cv::Mat &m)
    :data_owner(true),
     view_top(0),
     view_left(0)
{
	int w = m.size().width;
	int h = m.size().height;
	
	this->data_byte_width = w * CV_ELEM_SIZE(m.type());
	this->data_height = h;
	this->view_width = w;
	this->view_height = h;
	this->type = m.type();
	this->data = (char*)calloc(h, this->data_byte_width);


	for (int yi = 0; yi < h; yi++) {
		void *in = m.ptr(yi);
		char *out = this->ptr<char>(yi);

		memcpy(out, in, this->data_byte_width);
	}
}

void
W2Mat::to_cvmat(cv::Mat &target)
{
	int w = this->view_width;
	int h = this->view_height;

	target = cv::Mat::zeros(cv::Size(w, h), this->type);
	int elem_size = CV_ELEM_SIZE(this->type);

	for (int yi = 0; yi < h; yi++) {
		void *out = target.ptr(yi);
		void *in = this->ptr<char>(yi);

		memcpy(out, in, w * elem_size);
	}
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
	
	cv::Mat ret(cv::Size(w,h),
				wm.type,
				data + byte_offset,
				wm.data_byte_width);
	
	ret.copyTo(m);
}

// DO NOT USE IN NORMAL SITUAYION. Return wm is not own their data.
void extract_view_from_cvmat_offset(W2Mat &wm, cv::Mat &m,
                               int view_left_offset,
                               int view_top_offset,
                               int view_width,
                               int view_height)
{	
	extract_view_from_cvmat(wm, m);
	
	wm.view_top = view_top_offset;
	wm.view_left = view_left_offset;
	wm.view_width = view_width;
	wm.view_height = view_height;
}

void
extract_viewlist_from_cvmat(std::vector<W2Mat> &list, std::vector<cv::Mat> &cvmat)
{
	std::for_each(cvmat.begin(), cvmat.end(),
		[&list](cv::Mat &cv) {
		list.push_back(W2Mat(cv));
	});
}

void
extract_viewlist_to_cvmat(std::vector<cv::Mat> &cvmat, std::vector<W2Mat> &list)
{
	std::for_each(list.begin(), list.end(),
		[&cvmat](W2Mat &wm) {
		cv::Mat cv;
		extract_view_to_cvmat(cv, wm);
		cvmat.push_back(cv);
	});
}

#endif
