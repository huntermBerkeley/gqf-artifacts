cmake_minimum_required(VERSION 3.8)
project(cqf LANGUAGES C CUDA)

include_directories(
	"include"
	"src""
	)

add_executable(cqf 
  src/test.cu
    )




