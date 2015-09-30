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
	this->flags = 0;

#ifdef X86OPT
	unsigned int eax=0, ebx=0, ecx=0, edx=0;

#ifdef __GNUC__
	__get_cpuid(1, &eax, &ebx, &ecx, &edx);
#else
	int cpuInfo[4];
	__cpuid(cpuInfo, 1);
	eax = cpuInfo[0];
	ebx = cpuInfo[1];
	ecx = cpuInfo[2];
	edx = cpuInfo[3];
#endif
	if (ecx & (1<<0)) {
		this->flags |= ComputeEnv::HAVE_CPU_SSE3;
	}

	if ((ecx & 0x18000000) == 0x18000000) {
		this->flags |= ComputeEnv::HAVE_CPU_AVX;
	}

	if (ecx & (1<<12)) {
		this->flags |= ComputeEnv::HAVE_CPU_FMA;
	}
#endif // X86OPT

	this->pref_block_size = 512;
}
