# make a library lib cqf_core that will contain
# most of the relevant API
add_library(cqf_core STATIC
	bm.cu
	gqf.cu
	gqf_file.cu
	hashutil.cu
	partitioned_counter.cu
	zipf.cu
	)

set(CQF_DEBUG_CUFLAGS "${CQF_CUDA_FLAGS};-g")
set(CQF_RELEASE_CUFLAGS "${CQF_CUDA_FLAGS};-O3")

target_include_directories(cqf_core PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_compile_options(cqf_core PUBLIC "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:${CQF_DEBUG_CUFLAGS}>")
target_compile_options(cqf_core PUBLIC "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:${CQF_RELEASE_CUFLAGS}>")
target_compile_definitions(cqf_core PUBLIC "${ARCH_DEFS}")

set_target_properties( cqf_core
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link libmantis_core with the required libraries
target_link_libraries(cqf_core
	CUDA)
# The driver program is `mantis`, which is a front-end
# with the relevant sub-commands
add_executable(cqf test.cu)
target_include_directories(cqf PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_link_libraries(cqf cqf_core)
target_compile_options(cqf PUBLIC "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:${CQF_DEBUG_CUFLAGS}>")
target_compile_options(cqf PUBLIC "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CUDA>>:${CQF_RELEASE_CUFLAGS}>")
target_compile_definitions(cqf PUBLIC "${ARCH_DEFS}")

#add_executable(estimateNumOfKners estimateNumOfKmers.cc)
#target_include_directories(estimateNumOfKners PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
#target_link_libraries(estimateNumOfKners mantis_core)

# TODO: look more into why this is necessary
if (SDSL_INSTALL_PATH)
   set_property(TARGET cqf APPEND_STRING PROPERTY LINK_FLAGS "-L${SDSL_INSTALL_PATH}/lib")
   set_property(TARGET cqf_core APPEND_STRING PROPERTY LINK_FLAGS "-L${SDSL_INSTALL_PATH}/lib")
endif()

#install(TARGETS mantis
#        RUNTIME DESTINATION bin)
