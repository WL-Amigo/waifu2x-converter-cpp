NDK_PREFIX=/mnt/shared/android/android-ndk-r10e
ANDROID_TOOLCHAIN=$(NDK_PREFIX)/toolchains/x86-4.8/prebuilt/linux-x86_64/bin
ANDROID_PLATFORM=$(NDK_PREFIX)/platforms/android-21/arch-x86

TOOLCHAIN_PREFIX=$(ANDROID_TOOLCHAIN)/

STL=-I$(NDK_PREFIX)/sources/cxx-stl/gnu-libstdc++/4.8/include \
    -I$(NDK_PREFIX)/sources/cxx-stl/gnu-libstdc++/4.8/libs/x86/include \
    -I$(NDK_PERFIX)/sources/cxx-stl/gnu-libstdc++/4.8/include/backward

INCS=-I$(NDK_PREFIX)/sources/android/cpufeatures

ARCH=i686-linux-android

all: first $(ARCH)/runbench

CXXFLAGS=-msse3 -DX86OPT $(COMMON_CXXFLAGS) --sysroot $(ANDROID_PLATFORM) $(STL) -I/usr/local/include -DPICOJSON_USE_LOCALE=0 $(INCS)
CFLAGS=-msse3 -DX86OPT $(COMMON_CFLAGS) --sysroot $(ANDROID_PLATFORM)

include common.mk
-include $(X86_DEPS) $(BASE_DEPS)

$(ARCH)/cpu-features.o:$(NDK_PREFIX)/sources/android/cpufeatures/cpu-features.c
	$(CROSS_CC) -o $@ $^ $(CFLAGS) -c

OBJS=$(BASE_OBJS) $(BENCH_OBJS) $(X86_OBJS) $(ARCH)/cpu-features.o

$(ARCH)/runbench: $(OBJS)
	$(CROSS_CXX) -o $@ $^ $(CXXFLAGS) $(NDK_PREFIX)/sources/cxx-stl/gnu-libstdc++/4.8/libs/x86/libgnustl_static.a -fPIC -fPIE -fpic -pie


$(ARCH)/modelHandler_avx.o: $(TOPDIR)/src/modelHandler_avx.cpp
	$(CROSS_CXX) -o $@ $< $(CXXFLAGS) -c -mavx

$(ARCH)/modelHandler_fma.o: $(TOPDIR)/src/modelHandler_fma.cpp
	$(CROSS_CXX) -o $@ $< $(CXXFLAGS) -c -mfma
