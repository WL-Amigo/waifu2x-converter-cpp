# Waifu2x source building guides

#### You can download pre-built Windows binaries from:

https://github.com/DeadSix27/waifu2x-converter-cpp/releases

#### Guide Index:
- [Windows x64](#windows)
- [Linux](#linux)
- [MacOS](#macos--osx)

---

# Windows

### Requirements:
- [Visual Studio 2017](https://www.visualstudio.com/downloads/) (Community Edition is enough)
  - Make sure to Select these components:
    - `Desktop development with C++`
	- `VC++ 2017 version 15.4 v14.11 toolset`
- CMake 3.11 [ [cmake-3.11.0-win64-x64.msi](https://cmake.org/download/) ]
  - Select: Add CMake to the system PATH for the current user
- Windows 7 or newer
- [Git for Windows](https://git-scm.com/download/win) (Optional, for the git clone later on)

#### AMD GPUs only:
1. Download the AMD APP SDK [ [AMD-SDK-InstallManager-v1.4.87-1.exe](https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/) ]

2. Install the AMD SDK _(you only need `OpenCL` and `OpenCL runtime`)_

#### nVidia GPUs only:

1. Download the CUDA SDK 9.1 [ [cuda_9.1.85_win10.exe](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) (and all the patches) ]

2. Install the CUDA SDK _(you only need to select `CUDA->Development` and `CUDA->Runtime`)_ and all it's patches.
	* As of April 2018 we still need to manually change one line in the CUDA Headers:
	* Open `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include\crt\host_config.h` in Notepad++ or similar
	* Change line 133 from: `#if _MSC_VER < 1600 || _MSC_VER > 1911` to  `#if _MSC_VER < 1600 || _MSC_VER > 1912`
	* And instead of just running `...\vcvarsall.bat x64` in build step 4, you need run `...\vcvarsall.bat" x64 -vcvars_ver=14.11`

### Building:
##### We will be using `K:/w2x` as our base folder for this guide.
##### If you want to build for both GPU brands, just install both SDKs (see above).

1. Download OpenCV 3.4 [ [opencv-3.4.1-vc14_vc15.exe](https://opencv.org/releases.html) ]

2. Extract OpenCV to your base folder e.g `K:/w2x/opencv`

3. Open a Command prompt and run the following command: `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.11`
4. Now run these commands in order, to build w2x:
	```cmd
		cd "K:/w2x/"
		git clone "https://github.com/DeadSix27/waifu2x-converter-cpp"
		cd "waifu2x-converter-cpp"
		mkdir out && cd out
		cmake .. -DCMAKE_GENERATOR="Visual Studio 15 2017 Win64" -DOPENCV_PREFIX="K:/w2x/opencv/build/"
		msbuild waifu2xcpp.sln /p:Configuration=Release /p:Platform=x64
		copy K:\w2x\opencv\build\x64\vc15\bin\opencv_world341.dll Release\
		mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
	```
---

# Linux

### Requirements:
- GCC
- CMake
- OpenCV 3.4

#### Intel GPUs:
- Arch:
  -  sudo pacman -S beignet clinfo
- Ubuntu:
   - sudo apt install beignet-opencl-icd opencl-headers

#### AMD GPUs:
- Arch:
  -  sudo pacman -S opencl-mesa clinfo
- Ubuntu:
   - sudo apt install mesa-opencl-icd opencl-headers

#### nVidia GPUs:
- _Feel free to contribute to this guide_

### Building:
##### If you want to build for all GPU brands, just install all packages (untested, see above).
Run these commands in order:
```cmd
git clone "https://github.com/DeadSix27/waifu2x-converter-cpp"
cd waifu2x-converter-cpp
mkdir out && cd out
cmake ..
make -j4
sudo make install
```

# MacOS / OSX

You need [Homebrew](https://brew.sh) installed, as well as XCode for the compiler.
The following has been tested on OSX Sierra 10.12.6:

```
$ brew tap science && brew install opencv3
$ git clone https://github.com/DeadSix27/waifu2x-converter-cpp && cd waifu2x-converter-cpp
$ cmake -DOPENCV_PREFIX=/usr/local/Cellar/opencv3/<your version here> .
$ make -j4
$ cp -r models_rgb models
```
