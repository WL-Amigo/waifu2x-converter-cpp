# waifu2x (converter only version)

This is a reimplementation of waifu2x ([original](https://github.com/nagadomi/waifu2x)) converter function, in C++, using OpenCV.
This is also a reimplementation of [waifu2x python version](https://marcan.st/transf/waifu2x.py) by [Hector Martin](https://marcan.st/blog/).
You can use this as command-line tool of image noise reduction or/and scaling.

## Prebuilt binary-form release

https://github.com/DeadSix27/waifu2x-converter-cpp/releases

## waifu2x converter GUI

 * [waifu2x_win_koroshell](http://inatsuka.com/extra/koroshell/)
   - waifu2x-converter GUI frontend that is easy to use, and so cute. You need only drag&drop to convert your image. (and also you can set converting mode, noise reduction level, scale ratio, etc..., on GUI)
   - Both waifu2x-converter x86 and x64 are included this package, and GUI see your windows architecture(x86|x64) and selects automatically which to use. 
   - For windows only.


## Dependencies

### Platform

 * Ubuntu  
 * Windows  
 * Mac OS X?  
 
(This program probably can be built under MacOSX, because OpenCV and other libraries support OS X)

### Libraries

 * [OpenCV](http://opencv.org/)(version 3.2)
 * Now requires openCV 3.2

This programs also depends on libraries shown below, but these are already included in this repository.
*CUDA Support in OpenCV is optional, since not required. (in version 1.0.0, CUDA Support is not used.)*

 * [picojson](https://github.com/kazuho/picojson)
 * [TCLAP(Templatized C++ Command Line Parser Library)](http://tclap.sourceforge.net/)

## How to build

See [BUILDING.md](BUILDING.md) for more information.

## Usage

Usage of this program can be seen by executing this with `-h` option.

(My native language is not English, then I'm sorry for my broken English.)

## modifided by tanakamura
 * Added CUDA, OpenCL(AMD GPU), x86 FMA, x86 AVX Support (That is selected automatically at runtime)
  * OpenCL(AMD GPU) version achieves 40% of peak performance (291GFLOPS @ A10-7850K)
 * Added CMakeLists.txt
  * You can build it by cmake ($ cmake -D OPENCV_PREFIX=&lt;OpenCV include/lib dir&gt;)
 * [DLL interface](src/w2xconv.h)
  * You can use waifu2x as library. include w2xconv.h & link w2xc.lib.

## modifided by max20091
 * Updated tclap, PicoJSON, Waifu2x model
 * Added Noise Reduction Level 3
 * Using nagadomi original model
 * Added Windows build guide

## modifided by DeadSix27
* Improved build guide [BUILDING.md](BUILDING.md)
* Added support for Ubuntu 16.10 amd64
* Added support for Visual Studio 2015 (VC14) amd64
* Added option to override OpenCV Detection for when you're sure its there (OVERRIDE_OPENCV)
* VC14: Now requires OpenCV3.2 (VC12 still requires OpenCV3.0, Linux now requires OpenCV3.2 as well (only tested Ubuntu 16.10))
* VC14: Will not statically link OpenCV, you will need opencv_world320.dll in the w2x folder.
* Added Cuda checks to prevent the cuda code mess, and get rid of the extra NoCuda branch of max20091 may or may not work properly, I have no nVidia GPU to test
* TODO: Fix CL-binary file handling (**Now fixed** see: [f963753](https://github.com/DeadSix27/waifu2x-converter-cpp/commit/f963753227a09749291e93bd6769446ba1bb3945))
* TODO: Add proper UTF8/Unicode support
* ^Keep an eye out on my fork for those: https://github.com/DeadSix27/waifu2x-converter-cpp
