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

find_package(CUDA)

if(CUDA_FOUND)
  add_definitions(-DHAVE_CUDA)
else()
  message(WARNING "cuda not found. disabled.")
endif()

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

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64" OR
    CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR
    CMAKE_SYSTEM_PROCESSOR MATCHES "^i.86$")
  set(X86_TRUE true)
else()
  set(X86_TRUE false)
endif()

option(X86OPT "Enable X86 SIMD optimizations" ${X86_TRUE})
if(X86OPT)
  add_definitions(-DX86OPT)
  set(X86OPT_SOURCES src/modelHandler_avx.cpp src/modelHandler_fma.cpp src/modelHandler_sse.cpp)
endif()

add_executable(waifu2x-converter-cpp
  src/main.cpp)
add_library(w2xc SHARED
  src/modelHandler.cpp ${X86OPT_SOURCES}
  src/modelHandler_OpenCL.cpp src/convertRoutine.cpp src/threadPool.cpp
  src/modelHandler_CUDA.cpp src/w2xconv.cpp src/common.cpp
  src/Env.cpp src/Buffer.cpp
)

add_dependencies(w2xc gensrcs)


if(CMAKE_COMPILER_IS_GNUCXX OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

  if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(REMOVE_SYMBOL "")
  else()
    set(REMOVE_SYMBOL "-s")
  endif()

  set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${REMOVE_SYMBOL}")

  set_source_files_properties(src/modelHandler_avx.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx ${REMOVE_SYMBOL}")
  set_source_files_properties(src/modelHandler_fma.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mfma ${REMOVE_SYMBOL}")
  set_source_files_properties(src/modelHandler_sse.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse3 ${REMOVE_SYMBOL}")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd 4819 /wd 4996 /wd 4800")
  set_source_files_properties(src/modelHandler_avx.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX")
  set_source_files_properties(src/modelHandler_fma.cpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(NVCC_FLAGS "-m32")
endif()


if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  if(CUDA_FOUND)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" "-ccbin" "/usr/bin/g++-4.7" "${NVCC_FLAGS}")
  endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(CUDA_FOUND)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" "-Xcompiler" "/wd 4819" "${NVCC_FLAGS}")
  endif()
  set_target_properties(w2xc PROPERTIES PREFIX "")
  set_target_properties(w2xc PROPERTIES IMPORT_PREFIX "")
  set_target_properties(w2xc PROPERTIES IMPORT_SUFFIX ".lib")
else()
  if(CUDA_FOUND)
    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
  endif()
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

  add_executable(w2xcr w32-apps/w2xcr.c)
  target_link_libraries(w2xcr w2xc user32 shlwapi gdi32)
else()
  target_link_libraries(w2xc opencv_core opencv_imgproc ${IMGCODECS_LIBRARY} opencv_features2d)
endif()

set(CONV_EXE "$<TARGET_FILE_DIR:conv>/conv")

add_custom_command(
  OUTPUT modelHandler_OpenCL.cl.h
  COMMAND ${CONV_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/src/modelHandler_OpenCL.cl ${CMAKE_CURRENT_BINARY_DIR}/modelHandler_OpenCL.cl.h  str
  DEPENDS src/modelHandler_OpenCL.cl conv
  )

set(GPU_CODE modelHandler_OpenCL.cl.h)

if(CUDA_FOUND)
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

  set(GPU_CODE ${GPU_CODE} modelHandler_CUDA.ptx30.h modelHandler_CUDA.ptx20.h)
endif()

add_custom_target(gensrcs ALL
  DEPENDS ${GPU_CODE})

enable_testing()
add_executable(runtest w32-apps/runtest.c)
target_link_libraries(runtest w2xc)
add_test(runtest runtest)

install(TARGETS w2xc waifu2x-converter-cpp
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION "lib${LIB_SUFFIX}"
  ARCHIVE DESTINATION "lib${LIB_SUFFIX}"
)
install(FILES src/w2xconv.h DESTINATION include)

# Better add support for multiple directories where to find models
# with paths separated by : (a la PATH) then install unconditionally
option(INSTALL_MODELS "Install waifu2x RGB models" false)
if(INSTALL_MODELS)
  add_definitions(-DDEFAULT_MODELS_DIRECTORY="${CMAKE_INSTALL_PREFIX}/share/waifu2x-converter-cpp")
  install(DIRECTORY models_rgb/ DESTINATION share/waifu2x-converter-cpp)
endif()

option(INSTALL_DOCS "Install developer documentation" false)
if(INSTALL_DOCS)
  install(DIRECTORY appendix docs/internals
    DESTINATION share/doc/waifu2x-converter-cpp)
endif()
