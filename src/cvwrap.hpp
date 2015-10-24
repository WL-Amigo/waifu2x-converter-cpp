#ifndef W2MAT_HPP
#define W2MAT_HPP

#include <utility>
#include <vector>

struct W2Mat {
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
    W2Mat();

    W2Mat(const W2Mat &) = delete;
    W2Mat& operator=(const W2Mat&) = delete;

    W2Mat & operator= (W2Mat &&);
    W2Mat(W2Mat &&rhs) {
        *this = std::move(rhs);
    }

    ~W2Mat();

    static W2Mat copy_full(W2Mat &rhs);
    static W2Mat clip_view(const W2Mat &rhs,
                           int view_left_offset, int view_top_offset, 
                           int view_width, int view_height);

    template<typename T> T *ptr(int yi);
    template<typename T> T &at(int y, int x) {
        return this->ptr<T>(y)[x];
    }
};

struct W2Size {
    int width, height;

    W2Size(int w, int h)
        :width(w), height(h)
    {}
};

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>

typedef cv::Mat Mat_t;
typedef cv::Point Point_t;

W2Mat copy_from_cvmat(cv::Mat &m);
cv::Mat copy_to_cvmat(W2Mat &m);
W2Mat extract_view_from_cvmat(cv::Mat &m);
cv::Mat extract_view_to_cvmat(W2Mat &m);

W2Mat extract_view_from_cvmat_offset(cv::Mat &m,
                                     int view_left_offset,
                                     int view_top_offset,
                                     int view_width,
                                     int view_height);

std::vector<W2Mat> extract_viewlist_from_cvmat(std::vector<cv::Mat> &list);
std::vector<cv::Mat> extract_viewlist_to_cvmat(std::vector<W2Mat> &list);

static inline cv::Size cvSize_from_w2(W2Size const &s) {
    return cv::Size(s.width, s.height);
}

static inline W2Size W2Size_from_cv(cv::Size const &sz) {
    return W2Size(sz.width, sz.height);
}

#else

#define CV_32FC3 12
#define CV_32FC1 4
#define CV_8UC3 3
#define CV_8UC1 1

#define CV_ELEM_SIZE(type) (type)

#endif

template<typename T> T *
W2Mat::ptr(int yi){
    int off = 0;
    int elem_size = CV_ELEM_SIZE(this->type);

    off += (yi+view_top) * data_byte_width;
    off += view_left * elem_size;

    return (T*)(data + off);
}

#endif