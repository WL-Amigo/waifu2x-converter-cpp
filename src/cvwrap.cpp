#include <stdlib.h>
#include <string.h>
#include "cvwrap.hpp"

W2Mat::~W2Mat()
{
    if (data_owner) {
        free(data);
    }
}


W2Mat::W2Mat()
    :data_owner(false),
     data(NULL),
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
    this->data = (char*)malloc(width * height * CV_ELEM_SIZE(type));
}

W2Mat::W2Mat(int width, int height, int type, void *data, int data_step)
    :data_owner(false),
     data((char*)data),
     data_byte_width(data_step),
     data_height(height),
     view_top(0),
     view_left(0),
     view_width(width),
     view_height(height),
     type(type)
{}


W2Mat &
W2Mat::operator=(W2Mat &&rhs)
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

W2Mat
W2Mat::clip_view(const W2Mat & rhs,
                 int view_left_offset, int view_top_offset,
                 int view_width, int view_height)
{
    W2Mat view;

    view.data_owner = false;
    view.data = rhs.data;
    view.data_byte_width = rhs.data_byte_width;
    view.data_height = rhs.data_height;

    view.view_left = rhs.view_left + view_left_offset;
    view.view_top = rhs.view_top + view_top_offset;
    view.view_width = view_width;
    view.view_height = view_height;

    view.type = rhs.type;

    return view;
}

W2Mat
W2Mat::copy_full(W2Mat & rhs)
{
    W2Mat ret(rhs.view_width,
              rhs.view_height,
              rhs.type);

    int elem_size = CV_ELEM_SIZE(rhs.type);
    int w = rhs.view_width;
    int h = rhs.view_height;

    for (int yi=0; yi<h; yi++) {
        char *out = ret.ptr<char>(yi);
        char *in = rhs.ptr<char>(yi);

        memcpy(out, in, w * elem_size);
    }
    return ret;
}

#ifdef HAVE_OPENCV
W2Mat
copy_from_cvmat(cv::Mat &m)
{
    int w = m.size().width;
    int h = m.size().height;

    W2Mat wm(w, h, m.type());
    int elem_size = CV_ELEM_SIZE(m.type());

    for (int yi=0; yi<h; yi++) {
        void *in = m.ptr(yi);
        char *out = wm.ptr<char>(yi);

        memcpy(out, in, w * elem_size);
    }

    return std::move(wm);
}

W2Mat
extract_view_from_cvmat(cv::Mat &m)
{
    W2Mat wm;

    int w = m.size().width;
    int h = m.size().height;

    wm.data_owner = false;
    wm.data = (char*)m.data;
    wm.data_byte_width = m.step;
    wm.data_height = m.size().height;

    wm.view_top = 0;
    wm.view_left = 0;
    wm.view_width = m.size().width;
    wm.view_height = m.size().height;
    wm.type = m.type();

    return std::move(wm);
}

W2Mat
extract_view_from_cvmat_offset(cv::Mat &m,
                               int view_left_offset,
                               int view_top_offset,
                               int view_width,
                               int view_height)
{
    W2Mat ret = extract_view_from_cvmat(m);

    ret.view_top = view_top_offset;
    ret.view_left = view_left_offset;
    ret.view_width = view_width;
    ret.view_height = view_height;

    return ret;
}

cv::Mat
copy_to_cvmat(W2Mat &m)
{
    int w = m.view_width;
    int h = m.view_height;

    cv::Mat ret = cv::Mat::zeros(cv::Size(w, h), m.type);
    int elem_size = CV_ELEM_SIZE(m.type);

    for (int yi=0; yi<h; yi++) {
        void *out = ret.ptr(yi);
        void *in = m.ptr<char>(yi);

        memcpy(out, in, w * elem_size);
    }

    return std::move(ret);

}

cv::Mat
extract_view_to_cvmat(W2Mat &m)
{
    int w = m.view_width;
    int h = m.view_height;

    char *data = (char*)m.data;

    int byte_offset = 0;
    byte_offset += m.view_top * m.data_byte_width;
    byte_offset += m.view_left * CV_ELEM_SIZE(m.type);

    cv::Mat ret(cv::Size(w,h),
                m.type,
                data + byte_offset,
                m.data_byte_width);

    return std::move(ret);

}

std::vector<W2Mat>
extract_viewlist_from_cvmat(std::vector<cv::Mat> &list)
{
    std::vector<W2Mat> ret;

    std::for_each(list.begin(), list.end(),
                  [&ret](cv::Mat &cv) {
                      W2Mat w2 = extract_view_from_cvmat(cv);
                      ret.push_back(std::move(w2));
                  });

    return std::move(ret);
}

std::vector<cv::Mat>
extract_viewlist_to_cvmat(std::vector<W2Mat> &list)
{
    std::vector<cv::Mat> ret;

    std::for_each(list.begin(), list.end(),
                  [&ret](W2Mat &w2) {
                      cv::Mat cv = extract_view_to_cvmat(w2);
                      ret.push_back(std::move(cv));
                  });

    return std::move(ret);
}
#endif