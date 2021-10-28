/*
 * =====================================================================================
 *
 *       Filename:  qf_wrapper.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  10/28/2015 04:48:55 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef BF_WRAPPER_H
#define BF_WRAPPER_H

#include "bloom_filter.hpp"
#include <inttypes.h>

bloom_filter *b_filter;

inline int bf_init(uint64_t nbits)
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
		
	b_filter = new bloom_filter(parameters);
	return 0;
}

inline int bf_insert(__uint64_t val)
{
	b_filter->insert(val);
	return 0;
}

inline int bf_lookup(__uint128_t val)
{
	return b_filter->contains(val);
}

inline __uint128_t bf_range()
{
	__uint128_t one = 1;
	return (one << (sizeof(size_t)*8));
//	return UINT64_MAX;
}

inline int bf_destroy()
{
	b_filter->~bloom_filter();
	return 0;
}

#endif
