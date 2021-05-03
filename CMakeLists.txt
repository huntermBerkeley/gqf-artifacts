cmake_minimum_required(VERSION 3.8)
project(cqf LANGUAGES C CUDA)

find_package(openssl REQUIRED)

add_executable(cqf 
  src/test.cu
  src/gqf.cu
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
  src/test_partitioned_counter.c
  src/test_threadsafe.c
  src/zipf.c
    )


set_property(TARGET cqf PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

