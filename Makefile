all: waifu2x-converter-cpp

OPENCV=$(HOME)/usr

CXXFLAGS=-I$(OPENCV)/include -I$(CURDIR)/include -std=c++11 -pthread -Wall -MMD -save-temps -O2 -g
LDFLAGS=-L$(OPENCV)/lib -pthread -Wl,-rpath,$(OPENCV)/lib -g
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d -lOpenCL

OBJS=src/main.o src/modelHandler.o src/modelHandler_avx.o src/modelHandler_OpenCL.o

src/modelHandler_OpenCL.cpp: src/modelHandler_OpenCL.cl.h

waifu2x-converter-cpp: $(OBJS)
	g++ $(LDFLAGS) -o $@ $^ $(LDLIBS)

INPUT=./b.png
#INPUT=~/test/a.png

run: waifu2x-converter-cpp
	./waifu2x-converter-cpp -i $(INPUT) --model_dir models

run8: waifu2x-converter-cpp
	./waifu2x-converter-cpp -m scale -j 8 -i $(INPUT) --model_dir models

run4: waifu2x-converter-cpp
	./waifu2x-converter-cpp -j 4 -i $(INPUT) --model_dir models

run1: waifu2x-converter-cpp
	./waifu2x-converter-cpp -j 1 -i $(INPUT) --model_dir models

src/modelHandler_avx.o: src/modelHandler_avx.cpp
	g++ -c $(CXXFLAGS) -mfma -o $@ $<

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) waifu2x-converter-cpp

conv: conv.c

src/modelHandler_OpenCL.cl.h:src/modelHandler_OpenCL.cl conv
	./conv $< $@


