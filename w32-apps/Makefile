all: w2xc.exe w2xcr.exe

CXXFLAGS=-I../src
LDLIBS=..\w2xc.lib


w2xc.exe: w2xc.c
	cl $(CXXFLAGS) $< /Fe$@ ..\w2xc.lib /MT /O2

w2xcr.exe: w2xcr.c
	cl $(CXXFLAGS) $< /Fe$@ ..\w2xc.lib /MT /O2 user32.lib Pathcch.lib shlwapi.lib gdi32.lib

clean:
	del w2xc.exe w2xcr