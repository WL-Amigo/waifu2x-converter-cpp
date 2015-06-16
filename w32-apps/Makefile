all: w2xc.exe w2xc.dll

CXXFLAGS=-I../src
LDLIBS=..\w2xc.lib


w2xc.exe: w2xc.c
	cl $(CXXFLAGS) $< /Fe$@ ..\w2xc.lib /MT /O2


w2xc.dll: ..\w2xc.dll
	copy ..\w2xc.dll .

clean:
	del w2xc.exe 