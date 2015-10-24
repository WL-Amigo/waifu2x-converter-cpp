#ifndef W2MAT_HPP
#define W2MAT_HPP

#include <utility>
#include <vector>

struct W2Mat {
    bool data_owner;

    float *data;
    int data_byte_width;
    int data_height;

    int view_top;
    int view_left;
    int view_width;
    int view_height;

    W2Mat(int data_witdth, int data_height);
    W2Mat();

    W2Mat(const W2Mat &) = delete;
    W2Mat& operator=(const W2Mat&) = delete;

    W2Mat & operator= (W2Mat &&);
    W2Mat(W2Mat &&rhs) {
        *this = std::move(rhs);
    }

    ~W2Mat();

    static W2Mat copy_view(const W2Mat &rhs);
    static W2Mat copy_full(const W2Mat &rhs);
    static W2Mat clip_view(const W2Mat &rhs,
                           int view_top_offset, int view_left_offset,
                           int view_width_offset, int view_height_offset);

};

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>

typedef cv::Mat Mat_t;
typedef cv::Point Point_t;

W2Mat copy_from_cvmat(cv::Mat &m);
cv::Mat copy_to_cvmat(W2Mat &m);
W2Mat extract_view_from_cvmat(cv::Mat &m);
cv::Mat extract_view_to_cvmat(W2Mat &m);

std::vector<W2Mat> extract_viewlist_from_cvmat(std::vector<cv::Mat> &list);
std::vector<cv::Mat> extract_viewlist_to_cvmat(std::vector<W2Mat> &list);

#endif

#endif