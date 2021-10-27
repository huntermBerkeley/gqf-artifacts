/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef APP_WRAPPER_CUH
#define APP_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "gqf_int.cuh"
#include "gqf_file.cuh"


QF* approx_quotient_filter;


#ifndef NUM_SLOTS_TO_LOCK
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#endif

extern inline uint64_t app_xnslots();

extern inline int app_init(uint64_t nbits, uint64_t hash, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	//consolidate all of the device construction into one convenient func!
	qf_malloc_device(&approx_quotient_filter, nbits);

	

	return 0;
}

extern inline int app_insert(uint64_t val, uint64_t count)
{
	
	return 0;
}

extern inline int app_lookup(uint64_t val)
{
	return 0;
	//return qf_count_key_value(g_quotient_filter, val, 0, 0);
}

extern inline uint64_t app_range()
{
	//fix me im on device
	//have to deep copy

	QF* host_qf;

	cudaMallocHost((void **)&host_qf, sizeof(QF));

	cudaMemcpy(host_qf, approx_quotient_filter, sizeof(QF), cudaMemcpyDeviceToHost);

	uint64_t range;
	qfmetadata* _metadata;
	cudaMallocHost((void **)&_metadata, sizeof(qfmetadata));
	cudaMemcpy(_metadata, host_qf->metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);
	range = _metadata->range;

	cudaFreeHost(_metadata);
	cudaFreeHost(host_qf);
	return range;
}

extern inline uint64_t app_xnslots()
{
	//fix me im on device
	//have to deep copy

	QF* host_qf;

	cudaMallocHost((void **)&host_qf, sizeof(QF));

	cudaMemcpy(host_qf, approx_quotient_filter, sizeof(QF), cudaMemcpyDeviceToHost);

	uint64_t range;
	qfmetadata* _metadata;
	cudaMallocHost((void **)&_metadata, sizeof(qfmetadata));
	cudaMemcpy(_metadata, host_qf->metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);
	range = _metadata->xnslots;

	cudaFreeHost(_metadata);
	cudaFreeHost(host_qf);
	return range;
}

extern inline int app_destroy()
{
	//fix me this isn't going to work
	
	qf_destroy_device(approx_quotient_filter);
	return 0;
}


extern inline int app_iterator(uint64_t pos)
{
	//qf_iterator_from_position(g_quotient_filter, &g_quotient_filter_itr, pos);
	return 0;
}

/* Returns 0 if the iterator is still valid (i.e. has not reached the
 * end of the QF. */
extern inline int app_get(uint64_t *key, uint64_t *value, uint64_t *count)
{

	return 0;
	//return qfi_get_hash(&g_quotient_filter_itr, key, value, count);
}

/* Advance to next entry.  Returns whether or not another entry is
 * found.  */
extern inline int app_next()
{

	return 0;
	//return qfi_next(&g_quotient_filter_itr);
}

/* Check to see if the if the end of the QF */
extern inline int app_end()
{
	return 0;
	//return qfi_end(&g_quotient_filter_itr);
}

extern inline int app_bulk_insert(uint64_t * vals, uint64_t count)
{

	//calculate ratios
	//total_items += count;

	//int ratio = num_slots/total_items;

	//no sense in inflating the locks 200x for one insert
	//if (ratio > 15) ratio = 15;
	//printf("Dividing ratio %d\n", ratio);
	
  //cudaMemset((uint64_t *) buffer_sizes, 0, ratio*num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_one_hash(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK/ratio, num_locks*ratio, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
    approx_bulk_insert<<<(count-1)/32+1,32>>>(approx_quotient_filter, vals, count);
	
	cudaDeviceSynchronize();
	return 0;
}

//dummy func
extern inline uint64_t app_bulk_get(uint64_t * vals, uint64_t count){

	return approx_get_wrapper(approx_quotient_filter, vals, count);


}

//replace vals with a cudaMalloced Array for gpu inserts
extern inline uint64_t * app_prep_vals(__uint128_t * vals, uint64_t count){


	uint64_t *hostvals;
	//= (uint64_t * ) calloc(count, sizeof(uint64_t));
	cudaMallocManaged((void **)&hostvals, count*sizeof(uint64_t));

	for (uint64_t i=0; i < count; i++){
		hostvals[i] = vals[i];
	}

	return hostvals;
}

#endif
