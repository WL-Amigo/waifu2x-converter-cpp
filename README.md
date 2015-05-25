# waifu2x (converter only version)

This is a reimplementation of waifu2x ([original](https://github.com/nagadomi/waifu2x)) converter function, in C++, using OpenCV.
This is also a reimplementation of [waifu2x python version](https://marcan.st/transf/waifu2x.py) by [Hector Martin](https://marcan.st/blog/).

## Dependencies

### Platform

 * Ubuntu
 * Mac OS X?
 * Windows?
 
(This program probably can be built under MacOSX and Windows, because OpenCV and other libraries support these OS)

### Libraries

 * [OpenCV](http://opencv.org/)(C++, version 3.0.0 rc1)

This programs also depends on libraries shown below, but these are already included in this repository.

 * [picojson](https://github.com/kazuho/picojson)
 * [TCLAP(Templatized C++ Command Line Parser Library)](http://tclap.sourceforge.net/)

## How to build

### for Ubuntu

Sorry, under construction...

These are hints for building :

 * I recommend to install OpenCV from sources. (build instruction is found [here](http://opencv.org/quickstart.html))
 * include path : `include/` `(/path/to/opencv/installed/directory)/include`
 * library path : `(/path/to/opencv/installed/directory)/lib` 
     - if you have built and installed OpenCV from source, and have changed install directory(by using `CMAKE_INSTALL_PREFIX`), you may need to set environment variable `LD_LIBRARY_PATH` for your OpenCV installed directory.
 * libraries to link : `opencv_core` `opencv_imgproc` `opencv_imgcodecs` `opencv_features2d`
 * standard of C++ : `c++11`



## Usage

Usage of this program can be seen by executing this with `--help` option.



(My native language is not English, then I'm sorry for my broken English.)
