#xx STL

ANDROID_TOOLCHAIN=/mnt/shared/android/android-ndk-r10e/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin
ANDROID_PLATFORM=/mnt/shared/android/android-ndk-r10e/platforms/android-21/arch-arm

ARCH=arm-linux-androideabi

all: first $(ARCH)/runbench

CXXFLAGS=-mfloat-abi=soft -DARMOPT $(COMMON_CXXFLAGS) --sysroot $(ANDROID_PLATFORM)
CFLAGS=-mfloat-abi=soft -DARMOPT $(COMMON_CFLAGS) --sysroot $(ANDROID_PLATFORM)

include common.mk

OBJS=$(BASE_OBJS) $(BENCH_OBJS) $(ARM_OBJS)

$(ARCH)/runbench: $(OBJS)
	$(CROSS_CXX) -o $@ $^ -lpthread -ldl
