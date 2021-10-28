/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef BLOOM_WRAPPER_CUH
#define BLOOM_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
//#include "include/bloom.cuh"
#include "include/cpu_bloom_filter.hpp"

#ifndef BLOOM_CHECK
#define BLOOM_CHECK(ans)                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
}
}

#endif

device_bloom_filter * bloom_map;


extern inline void bloom_test(){

	uint64_t x =0;
	x+=1;
	return;
}


extern inline uint64_t bloom_xnslots();

extern inline int bloom_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{


	bloom_parameters parameters;

	uint64_t nslots = 1 << nbits;

	parameters.projected_element_count = nslots;
	parameters.false_positive_probability = 0.0009;
	parameters.random_seed = 0xA5A5A5A5;

	if (!parameters)
	{
		printf("Error - Invalid set of bloom filter parameters!");
		return 1;
	}

	parameters.compute_optimal_parameters();
		
	bloom_filter * host_filter;
	host_filter = new bloom_filter(parameters);

	device_bloom_filter * host_dev_filter;

	BLOOM_CHECK(cudaMallocManaged((void **) & host_dev_filter, sizeof(device_bloom_filter)));


	cudaMalloc((void **)& bloom_map, sizeof(device_bloom_filter));
	//BLOOM_CHECK(cudaMemcpy(host_dev_filter, host_filter, sizeof(bloom_filter), cudaMemcpyHostToDevice));

	unsigned int * dev_salt;

	unsigned char * dev_bit_table;

	cudaMalloc((void**)& dev_salt, host_filter->salt_.size()*sizeof(unsigned int));

	cudaMalloc((void **)& dev_bit_table, sizeof(char)*host_filter->raw_table_size_*8);

	BLOOM_CHECK(cudaMemcpy(dev_salt, host_filter->salt_.data(), host_filter->salt_.size()*sizeof(unsigned int), cudaMemcpyHostToDevice));

	BLOOM_CHECK(cudaMemcpy(dev_bit_table, host_filter->bit_table_, sizeof(char)*host_filter->raw_table_size_, cudaMemcpyHostToDevice));

	//nasty manual copy
	//host_dev_filter->salt_ = host_filter->salt[0].data();

	//need to copy over salt and bits



	//use raw_table_size_for bytes
	host_dev_filter->salt_ = dev_salt;
	host_dev_filter->bit_table_ = dev_bit_table;
	host_dev_filter->salt_count_ = host_filter->salt_count_;
	host_dev_filter->table_size_ = host_filter->table_size_;
	host_dev_filter->raw_table_size_ = host_filter->raw_table_size_;
	host_dev_filter->projected_element_count_ = host_filter->projected_element_count_;
	host_dev_filter->inserted_element_count_ = host_filter->inserted_element_count_ ;
	host_dev_filter->random_seed_ = host_filter->random_seed_ ;
	host_dev_filter->desired_false_positive_probability_ = host_filter->desired_false_positive_probability_ ;
	host_dev_filter->salt_size = host_filter->salt_count_;

	//set up done, copy to final filter
	cudaMemcpy(bloom_map, host_dev_filter, sizeof(device_bloom_filter), cudaMemcpyDeviceToDevice);

	free(host_filter);
	cudaFree(host_dev_filter);


	//cudaMemcpy(bloom_map, host)

	return 0;

	
}

//defunct don't use
extern inline int bloom_insert(uint64_t val, uint64_t count)
{
	//qf_insert(g_quotient_filter, val, 0, count, QF_NO_LOCK);
	return 0;
}


//defunct dont use
extern inline int bloom_lookup(uint64_t val)
{
	return 0;
	//qf_count_key_value(g_quotient_filter, val, 0, 0);
}


//defunct don't use
//these funcs need to be defined for other tables, so they stay in the filter definition
extern inline uint64_t bloom_range()
{
	return 0;
}

//shocker - don't use
extern inline uint64_t bloom_xnslots()
{
	return 0;
}

extern inline int bloom_destroy()
{
	//bloom_map.clear();
	

	return 0;
}


//defunct
extern inline int bloom_iterator(uint64_t pos)
{
	//qf_iterator_from_position(g_quotient_filter, &g_quotient_filter_itr, pos);
	return 0;
}


extern inline int bloom_get(uint64_t *key, uint64_t *value, uint64_t *count)
{
	return 0; //qfi_get_hash(&g_quotient_filter_itr, key, value, count);
}


extern inline int bloom_next()
{
	//return qfi_next(&g_quotient_filter_itr);
	return 0;
}

/* Check to see if the if the end of the QF */
extern inline int bloom_end()
{
	return 0;
	//return qfi_end(&g_quotient_filter_itr);
}


//this one does work!
__global__ void bulk_insert_kernel(device_bloom_filter * bloom_map, uint64_t * vals, uint64_t count){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= count) return;

	uint64_t val = vals[tid];

	bloom_map->insert(val);
}

 
extern inline int bloom_bulk_insert(uint64_t * vals, uint64_t count)
{

  //cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bloom_map.bulk_insert(vals, count);
	bulk_insert_kernel<<<(count-1)/512+1,512>>>(bloom_map, vals, count);
	cudaDeviceSynchronize();
	return 0;
}


__global__ void bulk_get_kernel(device_bloom_filter * bloom_map, uint64_t * vals, uint64_t count, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= count) return;

	uint64_t val = vals[tid];

	if (!bloom_map->contains(val)){

			atomicAdd((unsigned long long int *) misses, 1);

	}
		


}

extern inline uint64_t bloom_bulk_get(uint64_t * vals, uint64_t count){

  
	//the hmh2 table uses quadratic probing
	uint64_t * misses;
	//this is fine, should never be triggered
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  misses[0] = 0;


	bulk_get_kernel<<<(count-1)/512+1, 512>>>(bloom_map, vals, count, misses);

	cudaDeviceSynchronize();

	uint64_t toReturn = *misses;
	cudaFree(misses);

	return toReturn;
}
  //return bloom_map.bulk_find(vals, count);

//replace vals with a cudaMalloced Array for gpu inserts
//I solemnly swear I will clean this up later
extern inline uint64_t * bloom_prep_vals(__uint128_t * vals, uint64_t count){


	uint64_t *hostvals;


	return hostvals;
}

#endif
