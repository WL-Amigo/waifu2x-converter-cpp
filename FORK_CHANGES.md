## 1. Fork by [tanakamura](https://github.com/tanakamura/waifu2x-converter-cpp)
  - Added CUDA, OpenCL(AMD GPU), x86 FMA, x86 AVX Support (That is selected automatically at runtime)
    - OpenCL(AMD GPU) version achieves 40% of peak performance (291GFLOPS @ A10-7850K)
  - Added CMakeLists.txt
    - You can build it by cmake ($ cmake -D OPENCV_PREFIX=&lt;OpenCV include/lib dir&gt;)
  - [DLL interface](src/w2xconv.h)
    - You can use waifu2x as library. include w2xconv.h & link w2xc.lib.

## 2. Fork by [max20091](https://github.com/max20091/waifu2x-converter-cpp)
  - Updated tclap, PicoJSON, Waifu2x model
  - Added Noise Reduction Level 3
  - Using nagadomi original model
  - Added Windows build guide

## 3. Fork by [DeadSix27](https://github.com/DeadSix27/waifu2x-converter-cpp) (this fork)
  - Improved build guide [BUILDING.md](BUILDING.md)
  - Added support for Ubuntu 17.04 amd64
  - Added support for Visual Studio 2017 amd64
  - Added option to override OpenCV Detection for when you're sure its there (OVERRIDE_OPENCV)
  - VC14: Now requires OpenCV3.2 (VC12 still requires OpenCV3.0, Linux now requires OpenCV3.2 as well (only tested Ubuntu 16.10))
  - VC14: Will not statically link OpenCV, you will need opencv_world320.dll in the w2x folder.
  - Added Cuda checks to prevent the cuda code mess, and get rid of the extra NoCuda branch of max20091 may or may not work properly, I have no nVidia GPU to test
  - Fixed CL-binary file handling (See: [f963753](https://github.com/DeadSix27/waifu2x-converter-cpp/commit/f963753227a09749291e93bd6769446ba1bb3945))
  - Added recursive folder conversion
  - TODO: Add proper UTF8/Unicode support
  
