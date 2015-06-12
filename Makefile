all: waifu2x-converter-cpp filter-Spectre.isa filter_in1_out32-Spectre.isa

OPENCV=$(HOME)/usr

DEBUG= -g
CXXFLAGS=-I$(OPENCV)/include -I$(CURDIR)/include -std=c++11 -pthread -Wall -Wmissing-declarations -MMD -save-temps -O2 $(DEBUG) -fopenmp
LDFLAGS=-L$(OPENCV)/lib -pthread -Wl,-rpath,$(OPENCV)/lib $(DEBUG)
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -fopenmp -ldl

OBJS=src/main.o src/modelHandler.o src/modelHandler_avx.o src/modelHandler_fma.o \
	src/modelHandler_OpenCL.o src/convertRoutine.o src/threadPool.o

src/modelHandler_OpenCL.cpp: src/modelHandler_OpenCL.cl.h

waifu2x-converter-cpp: $(OBJS)
	g++ $(LDFLAGS) -o $@ $^ $(LDLIBS)

INPUT=./a.png
#INPUT=./b.png
#INPUT=./c.png
#INPUT=./d.png
#INPUT=./e.png

%-Spectre.isa: src/modelHandler_OpenCL.cl
	/opt/AMD/CodeXL_1.7-7300/CodeXLAnalyzer -s CL $< -k $* --isa $*.isa -c Spectre

run: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -i $(INPUT) --model_dir models

run8: waifu2x-converter-cpp
	perf stat ./waifu2x-converter-cpp -j 8 -i $(INPUT) --model_dir models

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
	g++ -c $(CXXFLAGS) -mavx -o $@ $<

src/modelHandler_fma.o: src/modelHandler_fma.cpp
	g++ -c $(CXXFLAGS) -mfma -o $@ $<

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) waifu2x-converter-cpp

conv: conv.c
	gcc -o $@ $<

src/modelHandler_OpenCL.cl.h:src/modelHandler_OpenCL.cl conv
	./conv $< $@


