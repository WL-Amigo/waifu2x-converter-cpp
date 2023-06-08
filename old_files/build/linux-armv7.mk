ARCH=arm-linux-gnueabihf
all: first $(ARCH)/runbench

CXXFLAGS=-fPIC -g -mfpu=neon -mfloat-abi=hard -DARMOPT $(COMMON_CXXFLAGS)
CFLAGS=-fPIC -g -mfpu=neon -mfloat-abi=hard -DARMOPT $(COMMON_CFLAGS)

include common.mk

OBJS=$(BASE_OBJS) $(BENCH_OBJS) $(ARM_OBJS)

$(ARCH)/runbench: $(OBJS)
	g++ -o $@ $^ -lpthread -ldl

