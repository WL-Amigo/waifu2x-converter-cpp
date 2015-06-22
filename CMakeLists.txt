cmake_minimum_required(VERSION 2.8)

add_executable(conv conv.c)

set(OPENCV_PREFIX "/usr"
  CACHE FILEPATH "OpenCV 2.x path")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OPENCV_PREFIX}/include)
link_directories(${OPENCV_PREFIX}/lib)

add_executable(waifu2x-converter-cpp
  src/main.cpp)
add_library(w2xc SHARED
  src/modelHandler.cpp src/modelHandler_avx.cpp src/modelHandler_fma.cpp
  src/modelHandler_OpenCL.cpp src/convertRoutine.cpp src/threadPool.cpp
  src/modelHandler_CUDA.cpp src/w2xconv.cpp)

add_dependencies(w2xc gensrcs)


find_package(CUDA REQUIRED)


if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

  set_source_files_properties(src/modelHandler_avx.cpp PROPERTIES COMPILE_FLAGS -mavx)
  set_source_files_properties(src/modelHandler_fma.cpp PROPERTIES COMPILE_FLAGS -mfma)

  set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" "-ccbin" "/usr/bin/g++-4.7")
endif()

target_link_libraries(waifu2x-converter-cpp PUBLIC w2xc)
target_link_libraries(w2xc opencv_core opencv_imgproc opencv_highgui opencv_features2d)

cuda_compile_ptx(src/modelHandler_CUDA.ptx20
  src/modelHandler_CUDA.cu
  OPTIONS -arch=sm_20)

cuda_compile_ptx(src/modelHandler_CUDA.ptx30
  src/modelHandler_CUDA.cu
  OPTIONS -arch=sm_30)


add_custom_command(
  OUTPUT modelHandler_CUDA.ptx20.h
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/conv modelHandler_CUDA.ptx20 modelHandler_CUDA.ptx20.h str
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx20
  )

add_custom_command(
  OUTPUT modelHandler_CUDA.ptx30.h
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/conv modelHandler_CUDA.ptx30 modelHandler_CUDA.ptx30.h str
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_CUDA.ptx30
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
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/conv ${CMAKE_CURRENT_SOURCE_DIR}/src/modelHandler_OpenCL.cl ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_OpenCL.cl.h  str
  DEPENDS src/modelHandler_OpenCL.cl
  )

add_custom_target(gensrcs ALL
  DEPENDS modelHandler_OpenCL.cl.h modelHandler_CUDA.ptx30.h modelHandler_CUDA.ptx20.h)
