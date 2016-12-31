# Download pre-built binaries from:

https://github.com/DeadSix27/waifu2x-converter-cpp/releases

# How to build yourself:

## Windows (64 Bit)

### VS2015 (OpenCV 3.2,VC14 tested on AMD GPU only)

(If you have an nVidia GPU and want to test, open an Issue and tell me)

Download and install/extract: 
* [VS2015 Community](https://www.visualstudio.com/downloads/)
* [OpenCV 3.2 for Windows](http://opencv.org/downloads.html)
* [cmake-3.7.1-win64-x64](https://cmake.org/download/)
* [AMD APP SDK 3.0](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)

#### Command line (console):

Open a Command promt and run:

`"%VSINSTALL%\VC\vcvarsall.bat" amd64` (where %VSINSTAL%L is `C:\Program Files (x86)\Microsoft Visual Studio 14.0` for example)ö
or open the VS2015 x64 Native Tools Command Prompt

Then run the following:

```
>git clone https://github.com/DeadSix27/waifu2x-converter-cpp
>cd waifu2x-converter-cpp
>mkdir output
>cd output
>cmake -DCMAKE_GENERATOR="Visual Studio 14 2015 Win64" -DOPENCV_PREFIX=%OpenCV%\build\ ..
>msbuild Project.sln /p:Configuration=Release /p:Platform=x64
>copy %OpenCV%\opencv32\build\x64\vc14\bin\opencv_world320.dll Release\
```
Note: %OpenCV% is your path to OpenCV, e.g `G:\w2x\opencv32`

After that you will find your Binaries in `waifu2x-converter-cpp\output\Release`

#### CMake GUI:

1. Clone DeadSix27/waifu2x-converter-cpp from master
2. Download & install VS2015 Community, OpenCV 3.2 for Windows, cmake-3.7.1-win64-x64.msi and AMD APP SDK 3.0
3. Run Cmake GUI, press browse source, select waifu2x-converter-cpp folder
4. Add OPENCV_PREFIX entry, set type to path and point it to %OpenCV%\build\ (%OpenCV% is the installed OpenCV location) -- ***NOTE:** Make very sure to set OPENCV_PREFIX before clicking configure and clear cmake cache everytime you configure again.*
5. Press Configure, choose Visual Studio 14 2015 Win64
6. Press Generate
7. Press Open Project
8. Right click Solution 'Project' and hit Build Solution.
9. Don't forget to copy models_rgb from waifu2x-converter-cpp and %OpenCV%\build\x64\vc14\bin\opencv_world320.dll into the output folder.

## Ubuntu

See README.md
## MacOS

See README.md

