@echo off
set CMAKE=C:\cmake\bin\cmake.exe
set OpenCV=C:/opencv
set OpenCV_VER=4.0.0
cd /d %~dp0
rd /S /q output
mkdir output
cd output
%CMAKE% -DCMAKE_GENERATOR="Visual Studio 15 2017 Win64" -DOPENCV_PREFIX=%OpenCV%/build/ -DFORCE_DUAL=ON -DUNICODE=ON ..
msbuild waifu2xcpp.sln /p:Configuration=Release /p:Platform=x64
set OpenCV=%OpenCV:/=\%
set OpenCV_VER=%OpenCV_VER:.=%
copy %OpenCV%\build\x64\vc15\bin\opencv_world%OpenCV_VER%.dll Release\
mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
cd ..
