set CMAKE=D:\cmake-3.11.0-rc1-win64-x64\bin\cmake.exe
set OpenCV=C:\opencv
cd /d %~dp0
rd /S /q output
mkdir output
cd output
%CMAKE% -DCMAKE_GENERATOR="MinGW Makefiles" -DFORCE_AMD=ON -DOPENCV_PREFIX=%OpenCV%\build\ ..
mingw32-make
copy %OpenCV%\build\x64\vc14\bin\opencv_world330.dll Release\
mkdir Release\models_rgb\ && copy ..\models_rgb Release\models_rgb\
cd ..
