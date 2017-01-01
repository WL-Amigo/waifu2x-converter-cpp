### Building guides for Ubuntu-16.10-64bit and Windows VS2015-64bit by DeadSix27

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

#### Command line (console) (OR use the Cmake GUI guide below, I prefer console):

Open a Command promt and run:

`"%VSINSTALL%\VC\vcvarsall.bat" amd64` (where %VSINSTAL%L is `C:\Program Files (x86)\Microsoft Visual Studio 14.0` for example) or open the VS2015 x64 Native Tools Command Prompt

Then run the following:

```
>git clone https://github.com/DeadSix27/waifu2x-converter-cpp
>cd waifu2x-converter-cpp
>mkdir output && cd output
>cmake -DCMAKE_GENERATOR="Visual Studio 14 2015 Win64" -DOPENCV_PREFIX=%OpenCV%\build\ ..
>msbuild Project.sln /p:Configuration=Release /p:Platform=x64
>copy %OpenCV%\build\x64\vc14\bin\opencv_world320.dll Release\
>mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
```
**Note:** %OpenCV% is your path to OpenCV, e.g `G:\w2x\opencv32`

**Note:** If you are sure you have OpenCV3.2 installed yet it does not detect it, add `-DOVERRIDE_OPENCV=1` to cmake.

**Note:** You can also just download from github and not use git clone.

After that you will find your Binaries in `waifu2x-converter-cpp\output\Release`

#### CMake GUI:

1. Clone DeadSix27/waifu2x-converter-cpp from master
2. Download and install the same files as noted above in: "Download and install/extract"
3. Run Cmake GUI, press browse source, select waifu2x-converter-cpp folder
4. Add OPENCV_PREFIX entry, set type to path and point it to %OpenCV%\build\ (%OpenCV% is the installed OpenCV location) -- ***NOTE:** Make very sure to set OPENCV_PREFIX before clicking configure and clear cmake cache everytime you configure again.*
5. Press Configure, choose Visual Studio 14 2015 Win64
6. Press Generate
7. Press Open Project
8. Right click Solution 'Project' and hit Build Solution.
9. Don't forget to copy models_rgb from waifu2x-converter-cpp and %OpenCV%\build\x64\vc14\bin\opencv_world320.dll into the output folder.
10. And also copy OpenCV%\build\x64\vc14\bin\opencv_world320.dll to the output folder

## Ubuntu 16.10 amd64 for AMD GPUs

### We have to build and install some requirements

#### AMD SDK

* Download [AMD APP SDK v3.0 (AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2)](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/)

Create a new folder:
```
$ mkdir w2x && cd w2x
```

Place AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2 in this folder, run:

```
$ bzip2 -dc AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2 | tar xvf -
$ sudo ./AMD-APP-SDK-v3.0.130.136-GA-linux64.sh
```
It will verify the package, show license, read or press "Q" then hit "Y"

It will ask for a location, the default /opt is fine, hit "enter"

It will install into /opt/AMDAPPSDK-3.0

It will tell you: You will need to log back in/open another terminal for the environment variable updates to take effect.

Do that, close ssh or terminal then log back in and cd to the folder we created above then continue:


We have to install opencl packages, this is only tested on amd64 & ubuntu 16.10 and might change in the future:

```
$ sudo apt install clinfo mesa-opencl-icd opencl-headers
$ clinfo
Number of platforms                               1
  Platform Name                                   Clover
  Platform Vendor                                 Mesa
  Platform Version                                OpenCL 1.1 Mesa 12.0.3
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_icd
  Platform Extensions function suffix             MESA

  Platform Name                                   Clover
Number of devices                                 1
  Device Name                                     AMD KAVERI (DRM 2.46.0 / 4.8.0-32-generic, LLVM 3.8.1)
  .....
  .....
```

#### OpenCV 3.2


You can also download the prebuilt package I built for Ubuntu 16.10-amd64 here: [libopencv_3.2-1_amd64.deb](https://github.com/DeadSix27/waifu2x-converter-cpp/releases)

It will install into: `/usr/local`

But I did not fully test this, I recommend building it on your own using the guide below this.

* Download [OpenCV 3.2 Linux Source (3.2.0.zip)](http://opencv.org/downloads.html)

Create a subfolder in the folder we created above and name it ocv_source for example.

Install the following packages:

```
sudo apt-get install build-essential libwebp libjpeg libtiff zlib1
```

Then:

```
$ mkdir ocv_source && cd ocv_source
$ wget https://github.com/opencv/opencv/archive/3.2.0.zip
$ unzip 3.2.0.zip && cd opencv-3.2.0/
$ mkdir release && cd release
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D ENABLE_PRECOMPILED_HEADERS=OFF ..
$ make -j4
$ sudo make install
$ cd ../..
```

Now lets build waifu2x-converter-cpp:

```
$ git clone https://github.com/DeadSix27/waifu2x-converter-cpp && cd waifu2x-converter-cpp
$ mkdir release && cd release
$ cmake -DOPENCV_PREFIX=/usr/local .. -DOVERRIDE_OPENCV=1
$ make -j4
$ cp ../models_rgb/ . -r
```
Done!
You should now have a fully working linux built of waifu2x-converter-cpp.
Try it out like below, should return your processors and GPUs:

``` 
$ ./waifu2x-converter-cpp --list-processor
   0: AMD KAVERI (DRM 2.46.0 / 4.8.0-32-generic, LLVM 3.8.1)(OpenCL    ): num_core=6
   1: AMD A8-7600 Radeon R7, 10 Compute Cores 4C+6G  (FMA       ): num_core=4
```

## MacOS

(I have no MAC PC myself so I will never be able to do a guide for this, sorry)

See [README.md](README.md)


____

# Archived buildguides:

### for Ubuntu

Sorry, under construction...

These are hints for building :

 * I recommend to install OpenCV from sources. (build instruction is found [here](http://opencv.org/quickstart.html))
 * include path : `include/` `(/path/to/opencv/installed/directory)/include`
 * library path : `(/path/to/opencv/installed/directory)/lib` 
     - if you have built and installed OpenCV from source, and have changed install directory(by using `CMAKE_INSTALL_PREFIX`), you may need to set environment variable `LD_LIBRARY_PATH` for your OpenCV installed directory.
 * libraries to link : `opencv_core` `opencv_imgproc` `opencv_imgcodecs` `opencv_features2d`
 * standard of C++ : `c++11`

### for Windows (Windows x64 (nVidia is untested))

#### VS2013:

1. Download and install VS2013, OpenCV 3.0, CMake x64 and AMD APP SDK v2.9
2. Run CMake GUI, Press Browse source and choose waifu2x-converter-cpp folder
3. Add OPENCV_PREFIX entry, folder location point to %OpenCV%\build\ (%OpenCV% is the installed OpenCV location)
4. Press Configure, choose Visual Studio 12 2013 x64
5. Compile it with VS2013 and done!


