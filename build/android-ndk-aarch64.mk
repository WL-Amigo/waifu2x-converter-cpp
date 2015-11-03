NDK_PREFIX=/mnt/shared/android/android-ndk-r10e
ANDROID_TOOLCHAIN=$(NDK_PREFIX)/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin
ANDROID_PLATFORM=$(NDK_PREFIX)/platforms/android-21/arch-arm64

TOOLCHAIN_PREFIX=$(ANDROID_TOOLCHAIN)/

STL=-I$(NDK_PREFIX)/sources/cxx-stl/gnu-libstdc++/4.9/include \
    -I$(NDK_PREFIX)/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/include \
    -I$(NDK_PERFIX)/sources/cxx-stl/gnu-libstdc++/4.9/include/backward

INCS=-I$(NDK_PREFIX)/sources/android/cpufeatures

ARCH=aarch64-linux-android

all: first $(ARCH)/runbench

CXXFLAGS=-DARMOPT $(COMMON_CXXFLAGS) --sysroot $(ANDROID_PLATFORM) $(STL) -I/usr/local/include -DPICOJSON_USE_LOCALE=0 $(INCS) -g
CFLAGS=-DARMOPT $(COMMON_CFLAGS) --sysroot $(ANDROID_PLATFORM) -g

include common.mk
-include $(ARM_DEPS) $(BASE_DEPS)

$(ARCH)/cpu-features.o:$(NDK_PREFIX)/sources/android/cpufeatures/cpu-features.c
	$(CROSS_CC) -o $@ $^ $(CFLAGS) -c

OBJS=$(BASE_OBJS) $(BENCH_OBJS) $(ARM_OBJS) $(ARCH)/cpu-features.o

$(ARCH)/runbench: $(OBJS)
	$(CROSS_CXX) -o $@ $^ $(CXXFLAGS) $(NDK_PREFIX)/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/libgnustl_static.a -g -fPIC -fPIE -fpic -pie
