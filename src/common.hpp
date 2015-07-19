#ifndef COMMON_HPP
#define COMMON_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "compiler.h"

#define ALIGN_UP(v,a) (((v+(a-1))/(a))*(a))

void pack_mat(float *out,
              std::vector<cv::Mat> &inputPlanes,
              int w, int h, int nplane);

void unpack_mat(std::vector<cv::Mat> &outputPlanes,
                const float *in,
                int w, int h, int nplane);

void unpack_mat1(cv::Mat &outputMat,
                 const float *in,
                 int w, int h);

void pack_mat_rgb(float *out,
                  cv::Mat &inputPlane,
                  int w, int h);
void pack_mat_rgb_f32(float *out,
                      cv::Mat &inputPlane,
                      int w, int h);
void pack_mat_bgr(float *out,
                  cv::Mat &inputPlane,
                  int w, int h);
void unpack_mat_rgb(cv::Mat &outputMat,
                    const float *in,
                    int w, int h);
void unpack_mat_rgb_f32(cv::Mat &outputMat,
                        const float *in,
                        int w, int h);
void unpack_mat_bgr(cv::Mat &outputMat,
                    const float *in,
                    int w, int h);

#endif
