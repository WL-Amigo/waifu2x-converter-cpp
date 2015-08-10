cmake_minimum_required(VERSION 2.8)

add_executable(conv conv.c)

set(OPENCV_PREFIX "/usr"
  CACHE FILEPATH "OpenCV 2.x path")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  # include_directories("/usr")
else()
  find_package(OpenCL REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
endif()

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OPENCV_PREFIX}/include)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(OCV_COMPILER_DIRNAME "mingw")
  elseif(MSVC11)
    set(OCV_COMPILER_DIRNAME "vc11")
  else()
    set(OCV_COMPILER_DIRNAME "vc12")
  endif()

  if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /MT")
  endif()

  if(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -static")
    SET(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -static")
  endif()

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OCV_SYSTEM_DIRNAME "x64")
  else()
    set(OCV_SYSTEM_DIRNAME "x86")
  endif()

  link_directories(${OPENCV_PREFIX}/${OCV_SYSTEM_DIRNAME}/${OCV_COMPILER_DIRNAME}/staticlib)
else()
  link_directories(${OPENCV_PREFIX}/lib)
endif()

find_library(IMGCODECS_LIBRARY NAMES opencv_imgcodecs opencv_highgui
  HINTS ${OPENCV_PREFIX}/lib)

add_executable(waifu2x-converter-cpp
  src/main.cpp)
add_library(w2xc SHARED
  src/modelHandler.cpp src/modelHandler_avx.cpp src/modelHandler_fma.cpp src/modelHandler_sse.cpp
  src/modelHandler_OpenCL.cpp src/convertRoutine.cpp src/threadPool.cpp
  src/modelHandler_CUDA.cpp src/w2xconv.cpp src/common.cpp
  src/Env.cpp
)

add_dependencies(w2xc gensrcs)


if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s")

  set_source_files_properties(src/modelHandler_avx.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx -s")
  set_source_files_properties(src/modelHandler_fma.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mfma -s")
  set_source_files_properties(src/modelHandler_sse.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse3 -s")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd 4819")
  set_source_files_properties(src/modelHandler_avx.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX")
  set_source_files_properties(src/modelHandler_fma.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(NVCC_FLAGS "-m32")
endif()


if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" "-ccbin" "/usr/bin/g++-4.7" "${NVCC_FLAGS}")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" "-Xcompiler" "/wd 4819" "${NVCC_FLAGS}")
  set_target_properties(w2xc PROPERTIES PREFIX "")
  set_target_properties(w2xc PROPERTIES IMPORT_PREFIX "")
  set_target_properties(w2xc PROPERTIES IMPORT_SUFFIX ".lib")
else()
  set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()

target_link_libraries(waifu2x-converter-cpp PUBLIC w2xc)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(CMAKE_COMPILER_IS_GNUCXX)
    target_link_libraries(w2xc
      opencv_core300 opencv_imgproc300 opencv_imgcodecs300 opencv_features2d300 libjpeg libpng libtiff libjasper opencv_hal300
      IlmImf zlib libwebp user32 opencv_core300 opencv_hal300) # xx 300
  else()
    target_link_libraries(w2xc opencv_core300 opencv_imgproc300 opencv_imgcodecs300
      opencv_features2d300 ippicvmt libjpeg libpng libtiff libjasper
      opencv_hal300 IlmImf zlib libwebp user32 opencv_core300 opencv_imgproc300)
  endif()

  add_executable(w2xcr WIN32 w32-apps/w2xcr.c)
  target_link_libraries(w2xcr w2xc user32 shlwapi gdi32)
else()
  target_link_libraries(w2xc opencv_core opencv_imgproc ${IMGCODECS_LIBRARY} opencv_features2d)
endif()

set(CONV_EXE "$<TARGET_FILE_DIR:conv>/conv")

cuda_compile_ptx(src/modelHandler_CUDA.ptx20
  src/modelHandler_CUDA.cu
  OPTIONS -arch=sm_20)

cuda_compile_ptx(src/modelHandler_CUDA.ptx30
  src/modelHandler_CUDA.cu
  OPTIONS -arch=sm_30)


add_custom_command(
  OUTPUT modelHandler_CUDA.ptx20.h
  COMMAND ${CONV_EXE} modelHandler_CUDA.ptx20 modelHandler_CUDA.ptx20.h str
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx20 conv
  )

add_custom_command(
  OUTPUT modelHandler_CUDA.ptx30.h
  COMMAND ${CONV_EXE} modelHandler_CUDA.ptx30 modelHandler_CUDA.ptx30.h str
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx30 conv
  )


add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx20
  COMMAND ${NVCC} -arch=sm_20 -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx20 ${CMAKE_CURRENT_SOURCE_DIR}/src/modelHandler_CUDA.cu
  DEPENDS src/modelHandler_CUDA.cu
  )
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx30
  COMMAND ${NVCC} -arch=sm_30 -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx30 ${CMAKE_CURRENT_SOURCE_DIR}/src/modelHandler_CUDA.cu
  DEPENDS src/modelHandler_CUDA.cu
  )


add_custom_command(
  OUTPUT modelHandler_OpenCL.cl.h
  COMMAND ${CONV_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/src/modelHandler_OpenCL.cl ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_OpenCL.cl.h  str
  DEPENDS src/modelHandler_OpenCL.cl conv
  )

add_custom_target(gensrcs ALL
  DEPENDS modelHandler_OpenCL.cl.h modelHandler_CUDA.ptx30.h modelHandler_CUDA.ptx20.h)


add_executable(runtest w32-apps/runtest.c)
target_link_libraries(runtest w2xc)