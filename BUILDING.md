### Building guides for Ubuntu-16.10-64bit and Windows VS2017-64bit by DeadSix27, MacOS by toyg

# Download pre-built binaries from:

https://github.com/DeadSix27/waifu2x-converter-cpp/releases

# How to build yourself:

## Windows (64 Bit)

### VS2017 (OpenCV 3.3,VC15 tested on AMD GPU only)

(If you have an nVidia GPU and want to test, open an Issue and tell me)

Download and install/extract: 
* [Visual Studio Community 2017](https://www.visualstudio.com/downloads/)
* [OpenCV 3.3.0 for Windows](http://opencv.org/releases.html)
* [cmake-3.9.4-win64-x64](https://cmake.org/download/)
* [AMD APP SDK 3.0 via AMD-SDK-InstallManager](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)

#### Command line (console) (OR use the Cmake GUI guide below, I prefer console):

Open a Command promt and run:

`"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64` or open the `x64 Native Tools Command Prompt for VS 2017`

Then run the following (assuming OpenCV3.3 is installed to `G:\w2x\opencv33\build` for example):

```
>git clone https://github.com/DeadSix27/waifu2x-converter-cpp
>cd waifu2x-converter-cpp
>mkdir output && cd output
>cmake .. -DCMAKE_GENERATOR="Visual Studio 15 2017 Win64" -DOPENCV_PREFIX="G:\w2x\opencv33\build\" 
>msbuild waifu2xcpp.sln /p:Configuration=Release /p:Platform=x64
>copy G:\w2x\opencv33\build\x64\vc14\bin\opencv_world330.dll Release\
>mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
```

**Note:** If you have CUDASDK and AMDSDK installed, you can force it to use either, by using -DFORCE_AMD=ON or -DFORCE_CUDA=ON

After that you will find your Binaries in `waifu2x-converter-cpp\output\Release`

#### CMake GUI:

(Assumes OpenCV3.3 is installed to `G:\w2x\opencv33\build` for example)

1. Clone https://github.com/DeadSix27/waifu2x-converter-cpp from master
2. Download and install the same files as noted above in: "Download and install/extract"
3. Run Cmake GUI, press browse source, select waifu2x-converter-cpp folder
4. Add OPENCV_PREFIX entry, set type to path and point it to G:\w2x\opencv33\build\ -- ***NOTE:** Make very sure to set OPENCV_PREFIX before clicking configure and clear cmake cache everytime you configure again.*
5. Press Configure, choose Visual Studio 15 2017 Win64
6. Press Generate
7. Press Open Project
8. Right click Solution 'waifu2xcpp' and hit Build Solution.
9. Don't forget to copy models_rgb from waifu2x-converter-cpp and G:\w2x\opencv33\build\x64\vc14\bin\opencv_world330.dll into the output folder.
10. And also copy G:\w2x\opencv33\build\x64\vc14\bin\opencv_world330.dll to the output folder

## Ubuntu 17.04 amd64 for AMD GPUs

### We have to build and install some requirements

#### AMD SDK

* Download [AMD APP SDK v3.0 (AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/) (Sadly there is no wget-able link due to the EULA)

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


We have to install opencl packages, this is only tested on amd64 & ubuntu 17.04 and might change in the future:

## For AMD GPU:
`$ sudo apt install clinfo mesa-opencl-icd opencl-headers`
## For Intel iGPU (open source driver):
`$ sudo apt install clinfo beignet-opencl-icd opencl-headers`

## For Intel iGPU (intel official driver, only use if above does not work):
Download the latest Intel OpenCL runtime here: https://software.intel.com/en-us/articles/opencl-drivers#latest_CPU_runtime (wget link is below)
`wget http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz`
`tar xvf opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz`
Note, this runtime supports only ubuntu 14.04, but we like to live dangerous and ignore that (seems to work fine on 17.04 anyway)
`sudo apt install lsb-core`
`cd opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25`
`sudo ./install.sh` (click continue when it warns about unsupported OS)

Then verify by running `clinfo`, if it shows `Number of devices` is greater than 0, like below, it's working.
```
$ clinfo
Number of platforms                               2
  Platform Name                                   Clover
  Platform Vendor                                 Mesa
  Platform Version                                OpenCL 1.1 Mesa 17.0.7
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_icd
  Platform Extensions function suffix             MESA

  Platform Name                                   Intel(R) OpenCL
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 1.2 LINUX
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_icd cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_                                                                                       base_atomics cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_depth_images cl_khr_3d_image_writes cl_intel_exec_by_local_threa                                                                                       d cl_khr_spir cl_khr_fp64
  Platform Extensions function suffix             INTEL

  Platform Name                                   Clover
Number of devices                                 0

  Platform Name                                   Intel(R) OpenCL
Number of devices                                 1
  Device Name                                     Intel(R) Pentium(R) CPU G4620 @ 3.70GHz
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 1.2 (Build 25)
  Driver Version                                  1.2.0.25
  Device OpenCL C Version                         OpenCL C 1.2
```

#### OpenCV 3.3


~~You can also download the prebuilt package I built for Ubuntu 17.04-amd64 here: [libopencv_3.3-1_amd64.deb](https://github.com/DeadSix27/waifu2x-converter-cpp/releases/tag/5.0)~~

~~It will install into: `/usr/local`~~

~~But I did not fully test this, I recommend building it on your own using the guide below this.~~
(for now build it yourself)

* Download [OpenCV 3.3 Linux Source (3.3.0.tar.gz)](http://opencv.org/releases.html)

Create a subfolder in the folder we created above and name it ocv_source for example.

Install the following packages:

```
sudo apt-get install build-essential libwebp6 libjpeg9 zlib1g libtiff5 libtiff5-dev
```

Then:

```
$ mkdir ocv_source && cd ocv_source
$ wget https://github.com/opencv/opencv/archive/3.3.0.tar.gz
$ tar -xvzf 3.3.0.tar.gz && cd opencv-3.3.0/
$ mkdir release && cd release
$ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_PRECOMPILED_HEADERS=OFF
$ make -j4
$ sudo make install
$ cd ../..
```

Now lets build waifu2x-converter-cpp:

```
$ git clone https://github.com/DeadSix27/waifu2x-converter-cpp && cd waifu2x-converter-cpp
$ mkdir release && cd release 
$ cmake .. -DOPENCV_PREFIX=/usr/local
$ make -j4
$ cp ../models_rgb/ . -r
```
Done!
You should now have a fully working linux built of waifu2x-converter-cpp.
Try it out like below, should return your processors and GPUs like in these examples:

``` 
$ ./waifu2x-converter-cpp --list-processor
   0: Intel(R) HD Graphics Kabylake Desktop GT2    (OpenCL    ): num_core=23
   1: Intel(R) Pentium(R) CPU G4620 @ 3.70GHz      (SSE3      ): num_core=4

$ ./waifu2x-converter-cpp --list-processor
   0: AMD KAVERI (DRM 2.46.0 / 4.8.0-32-generic, LLVM 3.8.1)(OpenCL    ): num_core=6
   1: AMD A8-7600 Radeon R7, 10 Compute Cores 4C+6G  (FMA       ): num_core=4
```

## MacOS / OSX

You need [Homebrew](https://brew.sh) installed, as well as XCode for the compiler.
The following has been tested on OSX Sierra 10.12.6:

```
$ brew tap science && brew install opencv3
$ git clone https://github.com/DeadSix27/waifu2x-converter-cpp && cd waifu2x-converter-cpp
$ cmake -DOPENCV_PREFIX=/usr/local/Cellar/opencv3/<your version here> .
$ make -j4
$ cp -r models_rgb models
```
Done! Try it out like below, should return your processors and GPUs:

``` 
$ ./waifu2x-converter-cpp --list-processor
   0: Intel(R) HD Graphics 530                     (OpenCL    ): num_core=24
   1: AMD Radeon Pro 455 Compute Engine            (OpenCL    ): num_core=12
...
```
____

# Archived buildguides (ignore these):

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


