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
  - See here for every change (too many to list):
  https://github.com/tanakamura/waifu2x-converter-cpp/compare/c427738c169545f7ac86d5d0083bfae4a41b696f...DeadSix27:master
  
