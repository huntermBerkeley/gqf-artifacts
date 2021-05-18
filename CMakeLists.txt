cmake_minimum_required(VERSION 3.10)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

project(cqf LANGUAGES C CUDA)


find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)

include(GNUInstallDirs)


add_subdirectory(src)

include_directories(
include
	)


link_directories(
	include
	)
add_subdirectory(src)

