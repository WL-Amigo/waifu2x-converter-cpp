OPENCV=$(HOME)/usr

CXXFLAGS=-I$(OPENCV)/include -I$(CURDIR)/include -std=c++11 -pthread -Wall -fopenmp -MMD -save-temps -O2 -march=native -g
LDFLAGS=-L$(OPENCV)/lib -pthread -Wl,-rpath,$(OPENCV)/lib -g
LDLIBS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d -fopenmp

OBJS=src/main.o src/modelHandler.o
waifu2x-converter-cpp: $(OBJS)
	g++ $(LDFLAGS) -o $@ $^ $(LDLIBS)

run: waifu2x-converter-cpp
	./waifu2x-converter-cpp -j 8 -i ~/test/a.png --model_dir models

run1: waifu2x-converter-cpp
	./waifu2x-converter-cpp -j 1 -i ~/test/a.png --model_dir models

-include $(OBJS:.o=.d)

clean:
	rm -f $(OBJS) waifu2x-converter-cpp