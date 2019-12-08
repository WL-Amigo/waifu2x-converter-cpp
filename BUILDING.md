# Waifu2x source building guides

#### You can download pre-built Windows binaries from:

https://github.com/DeadSix27/waifu2x-converter-cpp/releases

#### Index:
- [Windows x64](#windows)
- [Linux](#linux) (ARM only tested via Raspberry Pi 3)
- [MacOS](#macos--osx)

---

##### CMake options:
- `-DENABLE_OPENCV`
	- Build with OpenCV support? (Default: ON)
- `-DENABLE_UNICODE`
	- Build with Unicode support? (Default: ON)
- `-DENABLE_CUDA`
	- Build with CUDA support? (Default: ON)
- `-DINSTALL_MODELS`
	- Install models? (Default: ON on Linux, OFF on Windows)
- `-DENABLE_TESTS`
	- Build test binaries? (Default: OFF)
- `-DENABLE_GUI` _(Windows only)_
	- Build basic Windows GUI? (Default: ON)	

---

# Windows

### Requirements:
- Windows 7 or newer (only tested on 10)
- AMD, Intel or nVidia GPU
- [Visual Studio 2019](https://www.visualstudio.com/downloads/) (Community Edition is enough)
  - Make sure to Select these components:
    - `Desktop development with C++`
- CMake (latest version) [ [cmake-*.*.*-win64-x64.msi](https://cmake.org/download/) ]
  - Select: Add CMake to the system PATH for the current user
- [Git for Windows](https://git-scm.com/download/win)
	- (Optional, for the git clones, you'll have to download it as zip otherwise)
- Khronos OpenCL Headers: https://github.com/KhronosGroup/OpenCL-Headers
	- Git clone it anywhere, example: `K:\w2x\OpenCL-Headers`
- OpenCV 4.1.0 [ [opencv-4.1.0-vc14_vc15.exe](https://opencv.org/releases.html) ] (required for the main binary/program, not the library)
	- Extract to, for example: `K:\w2x\opencv`

#### CUDA (Optional, requires driver version v419.35 or newer):
1. Download the CUDA SDK 10.1 [ [cuda_10.1.105_418.96_win10.exe](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) ]
2. Install the CUDA SDK _(you only need to select `CUDA->Development` and `CUDA->Runtime`)_.

### Building:
##### We will be using `K:/w2x` as our base folder for this guide.

1. Open a Command prompt and run the following command: `"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64`

2. Now run these commands in order, to build w2x:
	- Make sure have `OpenCL_INCLUDE_DIR` and `OPENCV_PREFIX` set to the correct path
		```bat
		cd "K:\w2x\"
		git clone "https://github.com/DeadSix27/waifu2x-converter-cpp"
		cd "waifu2x-converter-cpp"
		mkdir out && cd out
		cmake .. -DCMAKE_GENERATOR="Visual Studio 16 2019" -A x64 -DOPENCV_PREFIX="K:/w2x/opencv/build/" -DOpenCL_INCLUDE_DIR="K:/w2x/OpenCL-Headers"
		msbuild waifu2xcpp.sln /p:Configuration=Release /p:Platform=x64 -m
		copy K:\w2x\opencv\build\x64\vc15\bin\opencv_world410.dll Release\
		mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
		cd ..
		```
4. (Optional) Add to SendTo menu (right click on file in Windows Explorer).

	- If you desire a GUI, try: https://github.com/YukihoAA/waifu2x_snowshell/releases
		```bat
		copy /y w32-apps\icon.ico out\Release\
		copy /y w32-apps\install.js out\Release\
		copy /y w32-apps\uninstall.js out\Release\
		cd out\Release
		wscript install.js
		cd .. && cd .. && cd ..
		```

# Linux

### Requirements:
- GCC 5+
- CMake
- OpenCL
- OpenCV 3+ (required for the main binary/program, not the library)
	- On Arch you probably also need: `gtk3 hdf5 vtk glew` because something seems broken in their CV package.

#### Packages:

##### Intel GPUs:
- Arch: `beignet`
- Ubuntu: `beignet-opencl-icd opencl-headers`
   
#### AMD GPUs:
- Arch: `opencl-mesa`
- Ubuntu: `mesa-opencl-icd opencl-headers ocl-icd-opencl-dev`

#### nVidia GPUs:
- Arch: `opencl-nvidia opencl-headers ocl-icd`
- Ubuntu: `ocl-icd-opencl-dev`

### nVidia GPUs (CUDA):
- Arch: `cuda`
- Ubuntu: `nvidia-cuda-toolkit`

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
If needed run `sudo ldconfig` after the install.

# MacOS / OSX

You need [Homebrew](https://brew.sh) installed, as well as a newer llvm installed (since Xcode's llvm does not have
 filesystem library)

The following has been tested on macOS Mojave 10.14.3:
```
$ brew install llvm opencv
$ git clone https://github.com/DeadSix27/waifu2x-converter-cpp && cd waifu2x-converter-cpp
$ cmake -DOPENCV_PREFIX=/usr/local/Cellar/opencv/<your version here> .
$ make -j4
$ sudo make install
```
