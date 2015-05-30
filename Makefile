OPENCV=$(HOME)/usr

CXXFLAGS=-I$(OPENCV)/include -I$(CURDIR)/include -std=c++11 -pthread -Wall -fopenmp
LDFLAGS=-L$(OPENCV)/lib -pthread -Wl,-rpath,$(OPENCV)/lib
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d -fopenmp

waifu2x-converter-cpp: src/main.o src/modelHandler.o
	g++ $(LDFLAGS) -o $@ $^ $(LDLIBS)

run: waifu2x-converter-cpp
	./waifu2x-converter-cpp -j 8 -i ~/test/a.png --model_dir models

run1: waifu2x-converter-cpp
	./waifu2x-converter-cpp -j 1 -i ~/test/a.png --model_dir models