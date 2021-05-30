#
# Help from :
# https://rix0r.nl/blog/2015/08/13/cmake-guide/
#

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(gpu_cqf VERSION 0.2 LANGUAGES CUDA)
if (NOT CMAKE_BUILD_TYPE)
   set (CMAKE_BUILD_TYPE "Release")
endif()

# We require C++17
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


#find_package(CUDA REQUIRED)

if (NH)
   set(ARCH_FLAGS "")
   set(ARCH_DEFS "")
   message(STATUS "Compiling without Haswell instructions")
else()
   set(ARCH_FLAGS "")
   set(ARCH_DEFS "-D__SSE4_2_")
endif()

#nvcc doesn't like warning flags
#set(CQF_CUDA_WARN "-Wno-implicit-function-declaration")
set(CQF_CUDA_FLAGS "${ARCH_DEFS};${CQF_CUDA_WARN}")

if (SDSL_INSTALL_PATH)
   message("Adding ${SDSL_INSTALL_PATH}/include to the include path")
   message("Adding ${SDSL_INSTALL_PATH}/build/lib to the build path")
   list(APPEND CQF_CUDA_FLAGS "-I${SDSL_INSTALL_PATH}/include")
   list(APPEND CQF_CUDA_FLAGS "-L${SDSL_INSTALL_PATH}/lib")
endif()

include(GNUInstallDirs)

add_subdirectory(src)
