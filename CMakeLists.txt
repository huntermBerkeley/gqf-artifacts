cmake_minimum_required(VERSION 3.8)
project(cqf LANGUAGES C CUDA)

include_directories(
	$ENV{HOME}/include 
	$ENV{HOME}/src
	)

add_executable(cqf 
  src/test.cu
    )


set_property(TARGET cqf PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)


