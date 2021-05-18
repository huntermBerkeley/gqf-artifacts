
add_library(cqf_core STATIC
	zipf.c
	partitioned_counter.c
	bm.c
	hashutil.c
	gqf.c
	gqf_call.cu
	)
target_include_directories(cqf_core PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

add_executable(cqf test.cu)
target_include_directories(cqf PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(cqf cqf_core)
