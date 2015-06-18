all: waifu2x-converter-cpp

OPENCV=/usr

CODEXL_ANALYZER=/opt/AMD/CodeXL_1.7-7300/CodeXLAnalyzer
DEBUG= -g
CCFE=-o
CXXFLAGS=-I$(OPENCV)/include -I$(CURDIR)/include -std=c++11 -pthread -Wall -Wmissing-declarations -MMD -save-temps -O2 $(DEBUG) -fopenmp
LDFLAGS=-L$(OPENCV)/lib -pthread -Wl,-rpath,$(OPENCV)/lib $(DEBUG)
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -fopenmp -ldl 
OSFX=.o
EXE=
RM=

include Makefile.common

waifu2x-converter-cpp: $(OBJS) libw2xc.so
	g++ $(LDFLAGS) -o $@ $^ -lw2xc

libw2xc.so: $(DLL_OBJS)
	g++ $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

#INPUT=fhd.png
INPUT=./a.png
#INPUT=./b.png
#INPUT=./c.png
#INPUT=./d.png
#INPUT=./e.png

run: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -i $(INPUT) --model_dir models

run8: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -j 8 -i $(INPUT) --model_dir models

run8d: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -j 8 -i $(INPUT) --model_dir models --disable-gpu

run8r: waifu2x-converter-cpp
	perf record ./waifu2x-converter-cpp -j 8 -i $(INPUT) --model_dir models

run4: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -j 4 -i $(INPUT) --model_dir models

run4d: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -j 4 -i $(INPUT) --model_dir models --disable-gpu

run2d: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -j 2 -i $(INPUT) --model_dir models --disable-gpu

run1: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -m scale -j 1 -i $(INPUT) --model_dir models

src/modelHandler_avx.o: src/modelHandler_avx.cpp
	$(CXX) -c $(CXXFLAGS) -mavx -o $@ $<

src/modelHandler_fma.o: src/modelHandler_fma.cpp
	$(CXX) -c $(CXXFLAGS) -mfma -o $@ $<

-include $(OBJS:.o=.d) $(DLL_OBJS:.o=.d)

conv$(EXE): conv.c
	$(CC) $(CCFE)o $@ $<

src/modelHandler_CUDA.ptx20: src/modelHandler_CUDA.cu
	nvcc -use_fast_math -arch=sm_20 -ccbin /usr/bin/gcc-4.7 -m64 -ptx -o $@ $< -O2

src/modelHandler_CUDA.ptx30: src/modelHandler_CUDA.cu
	nvcc -use_fast_math -arch=sm_30 -ccbin /usr/bin/gcc-4.7 -m64 -ptx -o $@ $< -O2


