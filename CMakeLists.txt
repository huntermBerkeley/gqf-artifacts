cmake_minimum_required(VERSION 3.10)
project(cqf LANGUAGES C CUDA)


find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

include(GNUInstallDirs)


add_subdirectory(/src})

include_directories(
	/include
	)


link_directories(
	/include
	)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

