# Waifu2x source building guides

#### You can download pre-built Windows binaries from:

https://github.com/DeadSix27/waifu2x-converter-cpp/releases

#### Index:
- [Windows x64](#windows)
- [Linux](#linux) (ARM is not supported, but could work maybe, open an issue if you get it working.)
- [MacOS](#macos--osx)

---

##### CMake options:
- `-DENABLE_OPENCV`
	- Build with OpenCV support? (Default: ON)
- `-ENABLE_UNICODE`
	- Build with Unicode support? (Default: ON)
- `-ENABLE_CUDA`
	- Build with CUDA support? (Default: ON)
- `-INSTALL_MODELS`
	- Install models? (Default: ON on Linux, OFF on Windows)

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
	- (Optional, for the git clones (you'll have to download the zip otherwise)
- Khronos OpenCL Headers: https://github.com/KhronosGroup/OpenCL-Headers
	- Git clone it anywhere, example: `K:\w2x\OpenCL-Headers`
- OpenCV 4.1.0 [ [opencv-4.1.0-vc14_vc15.exe](https://opencv.org/releases.html) ] (optional, but recommended)
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
		msbuild waifu2xcpp.sln /p:Configuration=Release /p:Platform=x64
		copy K:\w2x\opencv\build\x64\vc15\bin\opencv_world410.dll Release\
		mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
		cd ..
		```
4. (Optional) Add to SendTo menu (right click on file in Windows Explorer).

	- If you desire a GUI, try: https://github.com/YukihoAA/waifu2x_snowshell/releases
		```bat
		copy /y w32-apps\install.bat out\Release\
		copy /y w32-apps\install.js out\Release\
		copy /y w32-apps\uninstall.bat out\Release\
		copy /y w32-apps\uninstall.js out\Release\
		mkdir out\Release\ExtendedSendTo\ && copy /y w32-apps\ExtendedSendTo\ out\Release\ExtendedSendTo\
		cd out\Release
		install.bat
		cd ExtendedSendTo
		install.wsf
		cd .. && cd .. && cd ..
		```

# Linux

### Requirements:
- GCC 5+
- CMake
- OpenCL
- OpenCV 3+ (optional, but recommended)
	- On Arch you probably also need: `gtk3 hdf5 vtk glew` because something seems broken in their CV package.

#### Packages:

##### Intel GPUs:
- Arch: `beignet`
- Ubuntu: `beignet-opencl-icd opencl-headers`
   
#### AMD GPUs:
- Arch: `opencl-mesa`
- Ubuntu: `mesa-opencl-icd opencl-headers`

#### nVidia GPUs:
- Arch: `opencl-nvidia opencl-headers ocl-icd`
- Ubuntu: `?`

### nVidia GPUs (CUDA):
- Arch: `cuda`
- Ubuntu: `?` (Submit PR if you know)

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
$ cp -r models_rgb models
```
