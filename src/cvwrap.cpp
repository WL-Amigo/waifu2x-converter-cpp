#include <stdlib.h>
#include "cvwrap.hpp"

W2Mat::~W2Mat()
{
    if (data_owner) {
        free(data);
    }
}

W2Mat::W2Mat(int width, int height)
    :data_owner(true),
     data_width(width),
     data_height(height),
     view_top(0),
     view_left(0),
     view_width(width),
     view_height(height)
{
    this->data = (float*)malloc(width * height * sizeof(float));
}


W2Mat
copy_from_cvmat(cv::Mat &m)
{
    int w = m.size().width;
    int h = m.size().height;

    W2Mat wm(w, h);

    for (int yi=0; yi<h; yi++) {
        float *in = (float*)m.ptr(yi);
        float *out = wm.data + yi * w;

        memcpy(out, in, w * sizeof(float));
    }

    return std::move(wm);
}

cv::Mat
copy_to_cvmat(W2Mat &m)
{
    int w = m.view_width;
    int h = m.view_height;

    cv::Mat ret = cv::Mat::zeros(cv::Size(w, h), CV_32FC1);

    for (int yi=0; yi<h; yi++) {
        float *out = (float*)ret.ptr(yi);
        float *in = m.data + yi * w;

        memcpy(out, in, w * sizeof(float));
    }

    return std::move(ret);

}
