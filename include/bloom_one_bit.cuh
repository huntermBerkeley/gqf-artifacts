
#ifndef _BLOOM_ONE_CUH_
#define _BLOOM_ONE_CUH_

#define KEY_EMPTY 0


#include <cuda.h>
#include "hashutil.cuh"
#include <cmath>
#include <stdio.h>

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


//a modified version of the hash table found in mhm2, this only stores keys so that performance is comparable with the other data structures
struct bloom_one_bit_cuda {
  // Arrays for keys and values. They are separate because the keys get initialized with max number and the vals with zero
public:

  void init(uint64_t ht_capacity);
  void clear();
  __host__ void bulk_insert(uint64_t * inp_keys, uint64_t num_items);
  __host__ uint64_t bulk_find(uint64_t * inp_keys, uint64_t num_items);


  //__global__ void bulk_find_kernel(uint64_t * inp_keys, uint64_t num_items, volatile uint64_t * misses);

  //__global__ void bulk_insert_kernel(uint64_t * inp_keys, uint64_t num_items);

  //these are whatever cuda wants them to be
  unsigned int *keys = nullptr;
  //uint64_t *vals = nullptr;

  //bug - capacity lives on the cpu
  uint64_t capacity = 0;
  uint64_t k;
  int num = 0;
  
};


void bloom_one_bit_cuda::init(uint64_t ht_capacity) {


	//ht_capacity is the number of bits, use some formulae to figure out where to send them

	double p = .00390625; 

  capacity = - std::ceil((ht_capacity * std::log(p)) / std::pow(std::log(2), 2));

  k = 1.0 * capacity/ ht_capacity * std::log(2);

  capacity = 4*capacity;

  k = 5;
 

  printf("%llu slots requested for %llu, %d hashes\n", capacity, ht_capacity, k);

  printf("These give fp k %f\n", std::pow(.5, k));
  printf("These give fp m/n %f\n", std::pow(.5, 1.0*capacity/ht_capacity));
  printf("Size: %llu\n", ((capacity-1)/32+1) * sizeof(unsigned int));


  BLOOM_CHECK(cudaMalloc(&keys, ((capacity-1)/32 +1) * sizeof(unsigned int)));
  BLOOM_CHECK(cudaMemset((void *)keys, KEY_EMPTY, ((capacity-1)/32+1) * sizeof(unsigned int)));
  // cudaMalloc(&vals, capacity * sizeof(uint64_t));
  // cudaMemset(vals, 0, capacity * sizeof(uint64_t));


}


void bloom_one_bit_cuda::clear() {
  cudaFree((void *)keys);
  //cudaFree(vals);
}

__global__ void one_bit_bulk_insert_kernel(unsigned int * keys, uint64_t capacity, uint64_t k, uint64_t * inp_keys, uint64_t num_items){

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf
	for (uint64_t i=0; i < k; i++){
		 uint64_t slot = MurmurHash64A(((void *)&key), 8, 1+i) % capacity;
		 uint64_t trueSlot = slot/32;
		 int bit = slot % 32; 
		 
		 atomicOr(keys + trueSlot, 1 << bit);
	}
	
	__threadfence();


}


//the hmh2 table uses quadratic probing
__host__ void bloom_one_bit_cuda::bulk_insert(uint64_t * inp_keys, uint64_t num_items){

	one_bit_bulk_insert_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, k, inp_keys, num_items);

}

__global__ void one_bit_bulk_find_kernel(unsigned int * keys, uint64_t capacity, uint64_t k, uint64_t * inp_keys, uint64_t num_items, uint64_t * misses){

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf
	for (uint64_t i=0; i < k; i++){
		 uint64_t slot = MurmurHash64A(((void *)&key), 8, 1+i) % capacity;

		 uint64_t trueSlot = slot/32;
		 int bit = slot % 32;
		 if (!((keys[trueSlot] >> bit) & 1)){
		 	//not found, this a miss
		 	atomicAdd((unsigned long long int *) misses, 1);
		 	return;
		 }
		 return;
	}
	
	
}


//the hmh2 table uses quadratic probing
__host__ uint64_t bloom_one_bit_cuda::bulk_find(uint64_t * inp_keys, uint64_t num_items){

	uint64_t * misses;
	//this is fine, should never be triggered
  	cudaMallocManaged((void **)&misses, sizeof(uint64_t));
	one_bit_bulk_find_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, k, inp_keys, num_items, misses);

	cudaDeviceSynchronize();

	uint64_t toReturn = *misses;
	cudaFree(misses);

	return toReturn;
}

#endif