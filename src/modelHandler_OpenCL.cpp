#include <CL/cl.hpp>
#include "modelHandler.hpp"
#include "common.hpp"

static cl::Platform platform;
static cl::Device dev;
static cl::Context context;
static cl::CommandQueue queue;
static cl::Kernel ker;

static const char prog[] = 
#include "modelHandler_OpenCL.cl.h"
    ;

#define S_(a) #a
#define S(a) S_(a)

namespace w2xc {

bool have_OpenCL = false;

bool
initOpenCL()
{
		return false;
		std::vector<cl::Platform> pls;
		cl::Platform::get(&pls);

		if (pls.empty()) {
				return false;
		}

		for (int i=0; i<pls.size(); i++) {
				std::string name = pls[i].getInfo<CL_PLATFORM_NAME>();

				if (strncmp(name.c_str(), "Intel", 5) == 0) {
						continue;
				}

				cl_context_properties properties[] =
						{ CL_CONTEXT_PLATFORM, (cl_context_properties)(pls[i])(), 0};

				cl::Context cand_context(CL_DEVICE_TYPE_GPU, properties);
				std::vector<cl::Device> devices = cand_context.getInfo<CL_CONTEXT_DEVICES>();

				if (!devices.empty()) {
						platform = pls[i];
						dev = devices[0];
						context = cand_context;

						have_OpenCL = true;
						break;
				}
		}

		if (!have_OpenCL) {
				return false;
		}

		cl::Program::Sources src(1, std::make_pair(prog,strlen(prog)));
		cl::Program prog = cl::Program(context, src);

		std::vector<cl::Device> devs;
		devs.push_back(dev);

		cl_int err = prog.build(devs, "-DVEC_WIDTH=" S(GPU_VEC_WIDTH));
		if (err != CL_SUCCESS) {
				std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs[0]);
				puts(log.c_str());
				have_OpenCL = false;
				return false;
		}

		ker = cl::Kernel(prog, "filter", &err);
		if (err != CL_SUCCESS) {
				have_OpenCL = false;
				return false;
		}

		queue = cl::CommandQueue(context, dev, 0, &err);
		if (err != CL_SUCCESS) {
				have_OpenCL = false;
				return false;
		}

		printf("use GPU: %s\n",
			   dev.getInfo<CL_DEVICE_NAME>().c_str());

		return true;

}


void
filter_OpenCL_impl(const float *packed_input,
                   float *packed_output,
                   int nInputPlanes,
                   int nOutputPlanes,
                   const float *fbiases,
                   const float *weight,
                   cv::Size ipSize,
                   int nJob)
{
        int w = ipSize.width;
        int h = ipSize.height;
        cl::Buffer cl_packed_input(context,
                                   CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * w * h * nInputPlanes,
                                   (void*)packed_input);

        size_t out_size = sizeof(float) * w * h * nOutputPlanes;
        cl::Buffer cl_packed_output(context,
                                    CL_MEM_WRITE_ONLY,
                                    out_size);

        cl::Buffer cl_fbiases(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * nOutputPlanes,
                              (void*)fbiases
                );

        cl::Buffer cl_weight(context,
                             CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                             sizeof(float) * nOutputPlanes * nInputPlanes * 9,
                             (void*)weight
                );

        cl::Buffer cl_intermediate(context,
                                   CL_MEM_READ_WRITE,
                                   sizeof(float) * nOutputPlanes);

        cl_int err;

        int ai = 0;

        ker.setArg(ai++, cl_packed_input);
        ker.setArg(ai++, nInputPlanes);
        ker.setArg(ai++, cl_packed_output);
        ker.setArg(ai++, nOutputPlanes);
        ker.setArg(ai++, cl_fbiases);
        ker.setArg(ai++, h);
        ker.setArg(ai++, w);
        ker.setArg(ai++, cl_weight);
        ker.setArg(ai++, sizeof(float) * nOutputPlanes, NULL);

        cl::Event event;

        err = queue.enqueueNDRangeKernel(
                ker,
                cl::NullRange,
                cl::NDRange(w*GPU_VEC_WIDTH, h),
                cl::NDRange(GPU_VEC_WIDTH, 1),
                NULL,
                &event);

        if (err != CL_SUCCESS) {
                printf("enqueue ndrange error : %d\n", err);
                exit(1);
        }

        err = event.wait();

        if (err != CL_SUCCESS) {
                printf("wait ndrange error : %d\n", err);
                exit(1);
        }

        err = queue.enqueueReadBuffer(cl_packed_output,
                                      CL_TRUE,
                                      0, out_size, packed_output);

        if (err != CL_SUCCESS) {
                printf("read buffer error : %d\n", err);
                exit(1);
        }
}

}
