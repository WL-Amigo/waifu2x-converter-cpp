#include "Env.hpp"
#include "Buffer.hpp"

#ifdef X86OPT
#ifdef __GNUC__
#include <cpuid.h>
#else
#include <intrin.h>
#endif
#endif // X86OPT

ComputeEnv::ComputeEnv()
	:num_cl_dev(0),
         num_cuda_dev(0),
         cl_dev_list(nullptr),
         cuda_dev_list(nullptr),
         transfer_wait(0)
{
	this->pref_block_size = 512;
}
