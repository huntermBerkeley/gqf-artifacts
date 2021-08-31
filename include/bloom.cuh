
#ifndef _BLOOM_CUH_
#define _BLOOM_CUH_

#define KEY_EMPTY 0


#include <cuda.h>
#include "hashutil.cuh"
#include <cmath>


//a modified version of the hash table found in mhm2, this only stores keys so that performance is comparable with the other data structures
struct bloomCuda {
  // Arrays for keys and values. They are separate because the keys get initialized with max number and the vals with zero
public:

  void init(uint64_t ht_capacity);
  void clear();
  __host__ void bulk_insert(uint64_t * inp_keys, uint64_t num_items);
  __host__ uint64_t bulk_find(uint64_t * inp_keys, uint64_t num_items);


  //__global__ void bulk_find_kernel(uint64_t * inp_keys, uint64_t num_items, volatile uint64_t * misses);

  //__global__ void bulk_insert_kernel(uint64_t * inp_keys, uint64_t num_items);

  uint8_t *keys = nullptr;
  //uint64_t *vals = nullptr;

  //bug - capacity lives on the cpu
  uint64_t capacity = 0;
  uint64_t k;
  int num = 0;
  
};


void bloomCuda::init(uint64_t ht_capacity) {


	//ht_capacity is the number of bits, use some formulae to figure out where to send them

  capacity = ht_capacity * (std::log(.01) / std::log(.6185));
  k = std::log(2) * capacity/ht_capacity;
  cudaMalloc(&keys, capacity * sizeof(uint8_t));
  cudaMemset((void *)keys, KEY_EMPTY, capacity * sizeof(uint8_t));
  // cudaMalloc(&vals, capacity * sizeof(uint64_t));
  // cudaMemset(vals, 0, capacity * sizeof(uint64_t));
}



void bloomCuda::clear() {
  cudaFree((void *)keys);
  //cudaFree(vals);
}

__global__ void bulk_insert_kernel(uint8_t * keys, uint64_t capacity, uint64_t k, uint64_t * inp_keys, uint64_t num_items){

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf
	for (uint64_t i=0; i < k; i++){
		 uint64_t slot = MurmurHash64A(((void *)&key), sizeof(key), 1+i) % capacity;
		 keys[slot] = 1;
	}
	
	//insert with quadprobe
	__threadfence();


}


//the hmh2 table uses quadratic probing
__host__ void bloomCuda::bulk_insert(uint64_t * inp_keys, uint64_t num_items){

	bulk_insert_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, k, inp_keys, num_items);

}

__global__ void bulk_find_kernel(uint8_t * keys, uint64_t capacity, uint64_t k, uint64_t * inp_keys, uint64_t num_items, uint64_t * misses){

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf
	for (uint64_t i=0; i < k; i++){
		 uint64_t slot = MurmurHash64A(((void *)&key), sizeof(key), 1+i) % capacity;
		 if (!keys[slot]){
		 	//not found, this a miss
		 	atomicAdd((unsigned long long int *) misses, 1);
		 	return;
		 }
		 return;
	}
	
	
}


//the hmh2 table uses quadratic probing
__host__ uint64_t bloomCuda::bulk_find(uint64_t * inp_keys, uint64_t num_items){

	uint64_t * misses;
	//this is fine, should never be triggered
  	cudaMallocManaged((void **)&misses, sizeof(uint64_t));
	bulk_find_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, k, inp_keys, num_items, misses);

	cudaDeviceSynchronize();

	uint64_t toReturn = *misses;
	cudaFree(misses);

	return toReturn;
}

#endif