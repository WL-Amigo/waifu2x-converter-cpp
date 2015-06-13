#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "Buffer.hpp"

namespace w2xc {

bool initOpenCL(ComputeEnv *env);
bool initCUDA(ComputeEnv *env);

extern void filter_AVX_impl(const float *packed_input,
			    float *packed_output,
			    int nInputPlanes,
                            int nOutputPlanes,
                            const float *biases,
                            const float *weight,
                            int ip_width,
                            int ip_height,
			    int nJob);

extern void filter_FMA_impl(const float *packed_input,
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
