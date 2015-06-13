#ifndef COMMON_HPP
#define COMMON_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "compiler.h"

#define ALIGN_UP(v,a) (((v+(a-1))/(a))*(a))

static void UNUSED
pack_mat(float *out,
	 std::vector<cv::Mat> &inputPlanes,
	 int w, int h, int nplane)
{
	for (int i=0; i<nplane; i++) {
#pragma omp parallel for
		for (int yi=0; yi<h; yi++) {
			const float *mat_line = (float*)inputPlanes[i].ptr(yi);
			float *packed_line = out + i + (yi * nplane * w);

			for (int xi=0; xi<w; xi++) {
				packed_line[xi*nplane] = mat_line[xi];
			}
		}
	}
}

static void UNUSED
unpack_mat(std::vector<cv::Mat> &outputPlanes,
	   const float *in,
	   int w, int h, int nplane)
{
	for (int i=0; i<nplane; i++) {
#pragma omp parallel for
		for (int yi=0; yi<h; yi++) {
			float *mat_line = (float*)outputPlanes[i].ptr(yi);
			const float *packed_line = in + i + (yi * nplane * w);

			for (int xi=0; xi<w; xi++) {
				mat_line[xi] = packed_line[xi*nplane];
			}
		}
	}
}

static void UNUSED
unpack_mat1(cv::Mat &outputMat,
            const float *in,
            int w, int h)
{
#pragma omp parallel for
    for (int yi=0; yi<h; yi++) {
        float *mat_line = (float*)outputMat.ptr(yi);
        const float *packed_line = in + (yi * w);

        for (int xi=0; xi<w; xi++) {
            mat_line[xi] = packed_line[xi];
        }
    }
}

struct FLOPSCounter {
        double flop;
        double sec;

        FLOPSCounter()
                :flop(0), sec(0)
        {}
};


#endif
