cmake_minimum_required(VERSION 3.8)
project(cqf LANGUAGES C CUDA)

find_package(openssl REQUIRED)

add_library(cqf STATIC
  
  include/gqf_file.h
  include/gqf_int.h
  include/gqf_wrapper.h
  include/hashutil.h
  include/partitioned_counter.h
  include/zipf.h
  src/bm.c
  src/gqf_file.c
  src/hashutil.c
  src/partitioned_counter.c
  src/gqf.cu
  src/test_partitioned_counter.c
  src/test_threadsafe.c
  src/zipf.c
  )

add_executable(runtest src/test.cu)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( runtest
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
set_property(TARGET runtest PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

