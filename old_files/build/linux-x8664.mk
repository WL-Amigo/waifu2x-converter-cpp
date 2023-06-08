ARCH=x86_64-unknown-linux-gnu

CROSS_CC=gcc
CROSS_CXX=g++

all: first $(ARCH)/runbench

CXXFLAGS=-msse3 -mtune=native -DX86OPT $(COMMON_CXXFLAGS)
CFLAGS=-msse3 -mtune=native -DX86OPT $(COMMON_CFLAGS)

include common.mk

OBJS=$(BASE_OBJS) $(BENCH_OBJS) $(X86_OBJS)
-include $(X86_DEPS) $(BASE_DEPS)

$(ARCH)/runbench: $(OBJS)
	$(CROSS_CXX) -o $@ $^ -lpthread -ldl


$(ARCH)/modelHandler_avx.o: $(TOPDIR)/src/modelHandler_avx.cpp
	$(CROSS_CXX) -o $@ $< $(CXXFLAGS) -c -mavx -mtune=native

$(ARCH)/modelHandler_fma.o: $(TOPDIR)/src/modelHandler_fma.cpp
	$(CROSS_CXX) -o $@ $< $(CXXFLAGS) -c -mfma -mtune=native
