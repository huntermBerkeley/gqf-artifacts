cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
 
include(CTest)
 
add_library(cqf STATIC
	src/bm.cu
	src/gqf.cu
	src/gqf_file.cu
	src/hashutil.cu
	src/partitioned_counter.cu
	src/zipf.cu
	include/gqf.cuh
	include/gqf_int.cuh
	include/gqf_wrapper.cuh
	include/hashutil.cuh
	include/partitioned_counter.cuh
	include/zipf.cuh
	)  
	       
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(cqf PUBLIC cxx_std_11)
	       
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( cqf
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
				      
add_executable(cqf_test src/test.cu)
				      
set_property(TARGET cqf_test 
            PROPERTY CUDA_SEPARABLE_COMPILATION)
target_link_libraries(cqf_test PRIVATE cqf)



