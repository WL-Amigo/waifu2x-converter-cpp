#include <thread>
#include <immintrin.h>
#include <atomic>
#include "filters.hpp"
#include "sec.hpp"

#include "modelHandler_avx_func.hpp"

namespace w2xc {
void
filter_SSE_impl(ComputeEnv *env,
		const float *packed_input,
		float *packed_output,
		int nInputPlanes,
		int nOutputPlanes,
		const float *fbiases,
		const float *weight,
		int ip_width,
		int ip_height,
		int nJob)
{
	filter_AVX_impl0(env,
                         packed_input,
                         packed_output,
                         nInputPlanes,
                         nOutputPlanes,
                         fbiases,
                         weight,
                         ip_width,
                         ip_height,
                         nJob);
}


}
