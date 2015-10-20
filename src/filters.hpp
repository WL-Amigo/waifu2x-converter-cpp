#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "w2xconv.h"
#include "Buffer.hpp"
#include <vector>

namespace w2xc {

void initOpenCLGlobal(std::vector<W2XConvProcessor> *proc_list);
void initCUDAGlobal(std::vector<W2XConvProcessor> *proc_list);

bool initOpenCL(W2XConv *c, ComputeEnv *env, W2XConvProcessor *proc);
void finiOpenCL(ComputeEnv *env);
bool initCUDA(ComputeEnv *env, int dev_id);
void finiCUDA(ComputeEnv *env);

extern void filter_SSE_impl(ComputeEnv *env,
                            const float *packed_input,
			    float *packed_output,
			    int nInputPlanes,
                            int nOutputPlanes,
                            const float *biases,
                            const float *weight,
                            int ip_width,
                            int ip_height,
			    int nJob);

extern void filter_AVX_impl(ComputeEnv *env,
                            const float *packed_input,
			    float *packed_output,
			    int nInputPlanes,
                            int nOutputPlanes,
                            const float *biases,
                            const float *weight,
                            int ip_width,
                            int ip_height,
			    int nJob);

extern void filter_FMA_impl(ComputeEnv *env,
                            const float *packed_input,
			    float *packed_output,
			    int nInputPlanes,
			    int nOutputPlanes,
                            const float *biases,
                            const float *weight,
                            int ip_width,
                            int ip_height,
			    int nJob);

extern void filter_NEON_impl(ComputeEnv *env,
                             const float *packed_input,
                             float *packed_output,
                             int nInputPlanes,
                             int nOutputPlanes,
                             const float *biases,
                             const float *weight,
                             int ip_width,
                             int ip_height,
                             int nJob);

extern void filter_OpenCL_impl(ComputeEnv *env,
			       Buffer *packed_input,
                               Buffer *packed_output,
                               int nInputPlanes,
                               int nOutputPlanes,
                               const float *biases,
                               const float *weight,
                               int ip_width,
                               int ip_height,
                               int nJob);

extern void filter_CUDA_impl(ComputeEnv *env,
                             Buffer *packed_input,
                             Buffer *packed_output,
                             int nInputPlanes,
                             int nOutputPlanes,
                             const float *biases,
                             const float *weight,
                             int ip_width,
                             int ip_height,
                             int nJob);

}

#endif
