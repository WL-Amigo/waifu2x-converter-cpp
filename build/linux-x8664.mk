ARCH=x86_64-unknown-linux-gnu

all: first $(ARCH)/runbench

CXXFLAGS=-march=native -mtune=native -DX86OPT $(COMMON_CXXFLAGS)
CFLAGS=-march=native -mtune=native -DX86OPT $(COMMON_CFLAGS)

include common.mk

OBJS=$(BASE_OBJS) $(BENCH_OBJS) $(X86_OBJS)
-include $(X86_DEPS) $(BASE_DEPS)

$(ARCH)/runbench: $(OBJS)
	g++ -o $@ $^ -lpthread -ldl
