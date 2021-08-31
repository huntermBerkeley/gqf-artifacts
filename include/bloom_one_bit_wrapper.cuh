/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef BLOOM_ONE_WRAPPER_CUH
#define BLOOM_ONE_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "include/bloom_one_bit.cuh"

bloom_one_bit_cuda one_bit_bloom_map;


extern inline void one_bit_bloom_test(){

	uint64_t x =0;
	x+=1;
	return;
}


extern inline uint64_t one_bit_bloom_xnslots();

extern inline int one_bit_bloom_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//just pass in the estimated number of elements, internal sizing is handled by the filter
	one_bit_bloom_map.init((1ULL << nbits));

	//cudaMalloc((void **)& map, sizeof(mhm2CountsMap));

	//cudaMemcpy(map, hostMap, sizeof(mhm2CountsMap), cudaMemcpyHostToDevice);
	return 0;
}

//defunct don't use
extern inline int one_bit_bloom_insert(uint64_t val, uint64_t count)
{
	//qf_insert(g_quotient_filter, val, 0, count, QF_NO_LOCK);
	return 0;
}


//defunct dont use
extern inline int one_bit_bloom_lookup(uint64_t val)
{
	return 0;
	//qf_count_key_value(g_quotient_filter, val, 0, 0);
}


//defunct don't use
//these funcs need to be defined for other tables, so they stay in the filter definition
extern inline uint64_t one_bit_bloom_range()
{
	return 0;
}

//shocker - don't use
extern inline uint64_t one_bit_bloom_xnslots()
{
	return 0;
}

extern inline int one_bit_bloom_destroy()
{
	one_bit_bloom_map.clear();
	

	return 0;
}


//defunct
extern inline int one_bit_bloom_iterator(uint64_t pos)
{
	//qf_iterator_from_position(g_quotient_filter, &g_quotient_filter_itr, pos);
	return 0;
}


extern inline int one_bit_bloom_get(uint64_t *key, uint64_t *value, uint64_t *count)
{
	return 0; //qfi_get_hash(&g_quotient_filter_itr, key, value, count);
}


extern inline int one_bit_bloom_next()
{
	//return qfi_next(&g_quotient_filter_itr);
	return 0;
}

/* Check to see if the if the end of the QF */
extern inline int one_bit_bloom_end()
{
	return 0;
	//return qfi_end(&g_quotient_filter_itr);
}


//this one does work!
extern inline int one_bit_bloom_bulk_insert(uint64_t * vals, uint64_t count)
{

  //cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	one_bit_bloom_map.bulk_insert(vals, count);
	cudaDeviceSynchronize();
	return 0;
}

extern inline uint64_t one_bit_bloom_bulk_get(uint64_t * vals, uint64_t count){

  

  return one_bit_bloom_map.bulk_find(vals, count);

}

//replace vals with a cudaMalloced Array for gpu inserts
//I solemnly swear I will clean this up later
extern inline uint64_t * one_bit_bloom_prep_vals(__uint128_t * vals, uint64_t count){


	uint64_t *hostvals;
	//= (uint64_t * ) calloc(count, sizeof(uint64_t));
	cudaMallocManaged((void **)&hostvals, count*sizeof(uint64_t));

	for (uint64_t i=0; i < count; i++){
		hostvals[i] = vals[i];
	}

	return hostvals;
}

#endif
