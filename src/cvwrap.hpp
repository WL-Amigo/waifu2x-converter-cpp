#ifndef W2MAT_HPP
#define W2MAT_HPP

#include <utility>

struct W2Mat {
    bool data_owner;

    float *data;
    int data_width;
    int data_height;

    int view_top;
    int view_left;
    int view_width;
    int view_height;

    W2Mat(int data_witdth, int data_height);

    W2Mat(const W2Mat &) = delete;
    W2Mat& operator=(const W2Mat&) = delete;

    ~W2Mat();

    static W2Mat copy_view(const W2Mat &rhs);
    static W2Mat copy_full(const W2Mat &rhs);
    static W2Mat clip_view(const W2Mat &rhs,
                           int top, int left, int view_width, int view_height);

    W2Mat & operator = (W2Mat && rhs) {
        this->data_owner = rhs.data_owner;

        this->data_width = rhs.data_width;
        this->data_height = rhs.data_height;

        this->view_top = rhs.view_top;
        this->view_left = rhs.view_left;
        this->view_width = rhs.view_width;
        this->view_height = rhs.view_height;

        rhs.data_owner = false;
        rhs.data = nullptr;

        return *this;
    }

    W2Mat (W2Mat && rhs) {
        *this = std::move(rhs);
    }
};

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>

typedef cv::Mat Mat_t;
typedef cv::Point Point_t;

W2Mat copy_from_cvmat(cv::Mat &m);
cv::Mat copy_to_cvmat(W2Mat &m);

#endif

#endif