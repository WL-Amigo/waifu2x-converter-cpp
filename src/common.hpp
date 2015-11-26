#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include "compiler.h"
#include "cvwrap.hpp"

#define ALIGN_UP(v,a) (((v+(a-1))/(a))*(a))

void pack_mat(float *out,
              std::vector<W2Mat> &inputPlanes,
              int w, int h, int nplane);

void unpack_mat(std::vector<W2Mat> &outputPlanes,
                const float *in,
                int w, int h, int nplane);

void unpack_mat1(W2Mat &outputMat,
                 const float *in,
                 int w, int h);

void pack_mat_rgb(float *out,
                  W2Mat &inputPlane,
                  int w, int h);
void pack_mat_rgb_f32(float *out,
                      W2Mat &inputPlane,
                      int w, int h);
void pack_mat_bgr(float *out,
                  W2Mat &inputPlane,
                  int w, int h);
void unpack_mat_rgb(W2Mat &outputMat,
                    const float *in,
                    int w, int h);
void unpack_mat_rgb_f32(W2Mat &outputMat,
                        const float *in,
                        int w, int h);
void unpack_mat_bgr(W2Mat &outputMat,
                    const float *in,
                    int w, int h);

/*
 * src is exist && dst is not exist                       : true
 * src is exist && dst is exist && dst is older than src  : true
 * otherwise                                              : false
 */
bool update_test(const char *dst_path,
                 const char *src_path);

#endif
