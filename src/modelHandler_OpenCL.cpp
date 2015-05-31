#include <CL/cl.hpp>
#include "modelHandler.hpp"

static cl::Platform platform;
static cl::Device dev;
static cl::Context context;

namespace w2xc {

bool have_OpenCL = false;

bool
initOpenCL()
{
	std::vector<cl::Platform> pls;
	cl::Platform::get(&pls);

	if (pls.empty()) {
		return false;
	}

	for (int i=0; i<pls.size(); i++) {
		cl_context_properties properties[] = 
			{ CL_CONTEXT_PLATFORM, (cl_context_properties)(pls[i])(), 0};

		cl::Context context(CL_DEVICE_TYPE_CPU, properties);
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	}
}

}