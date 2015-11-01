TOPDIR=..

first: $(ARCH)

$(ARCH):
	mkdir -p $(ARCH)
	echo $(BASE_OBJS) $(BASE_SRCS)

COMMON_CXXFLAGS=-std=gnu++11 -I$(TOPDIR)/include -I$(TOPDIR)/src -O2 -I$(ARCH) -MD -fPIC -fPIE -fpic
COMMON_CFLAGS=-std=gnu99 -I$(TOPDIR)/include -I$(TOPDIR)/src -I$(ARCH) -MD -fPIC -fPIE -fpic

BENCH_SRCS_BASENAME=runbench.c

BENCH_OBJS_BASENAME=$(BENCH_SRCS_BASENAME:.c=.o)
BENCH_SRCS:=$(foreach bn,$(BENCH_SRCS_BASENAME),$(TOPDIR)/w32-apps/$(bn))
BENCH_OBJS:=$(foreach bn,$(BENCH_OBJS_BASENAME),$(ARCH)/$(bn))

BASE_SRCS_BASENAME=modelHandler.cpp \
  modelHandler_OpenCL.cpp \
  convertRoutine.cpp \
  threadPool.cpp \
  w2xconv.cpp \
  common.cpp \
  cvwrap.cpp \
  Env.cpp \
  Buffer.cpp \
  modelHandler_CUDA.cpp 

conv:$(TOPDIR)/conv.c
	gcc -o $@ $<

$(ARCH)/modelHandler_OpenCL.o: $(ARCH)/modelHandler_OpenCL.cl.h
$(ARCH)/modelHandler_OpenCL.cl.h: $(TOPDIR)/src/modelHandler_OpenCL.cl conv
	./conv $< $@ str

BASE_OBJS_BASENAME=$(BASE_SRCS_BASENAME:.cpp=.o)
BASE_SRCS:=$(foreach bn,$(BASE_SRCS_BASENAME),$(TOPDIR)/src/$(bn))
BASE_OBJS:=$(foreach bn,$(BASE_OBJS_BASENAME),$(ARCH)/$(bn))
BASE_DEPS:=$(BASE_OBJS:.o=.d)

X86_SRCS_BASENAME=modelHandler_sse.cpp \
        modelHandler_avx.cpp \
        modelHandler_fma.cpp

X86_OBJS_BASENAME=$(X86_SRCS_BASENAME:.cpp=.o)
X86_SRCS:=$(foreach bn,$(X86_SRCS_BASENAME),$(TOPDIR)/src/$(bn))
X86_OBJS:=$(foreach bn,$(X86_OBJS_BASENAME),$(ARCH)/$(bn))
X86_DEPS:=$(X86_OBJS:.o=.d)

ARM_SRCS_BASENAME=modelHandler_neon.cpp 

ARM_OBJS_BASENAME=$(ARM_SRCS_BASENAME:.cpp=.o)
ARM_SRCS:=$(foreach bn,$(ARM_SRCS_BASENAME),$(TOPDIR)/src/$(bn))
ARM_OBJS:=$(foreach bn,$(ARM_OBJS_BASENAME),$(ARCH)/$(bn))
ARM_DEPS:=$(ARM_OBJS:.o=.d)

CROSS_CC?=$(TOOLCHAIN_PREFIX)$(ARCH)-gcc
CROSS_CXX?=$(TOOLCHAIN_PREFIX)$(ARCH)-g++

$(ARCH)/%.o: $(TOPDIR)/src/%.cpp
	$(CROSS_CXX) -o $@ $< $(CXXFLAGS) -c
$(ARCH)/%.o: $(TOPDIR)/src/%.c
	$(CROSS_CC) -o $@ $< $(CFLAGS) -c

$(ARCH)/%.o: $(TOPDIR)/w32-apps/%.cpp
	$(CROSS_CXX) -o $@ $< $(CXXFLAGS) -c
$(ARCH)/%.o: $(TOPDIR)/w32-apps/%.c
	$(CROSS_CC) -o $@ $< $(CFLAGS) -c


clean:
	rm -rf $(ARCH)
