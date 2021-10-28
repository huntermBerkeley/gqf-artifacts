
#ifndef _BLOOM_CUH_
#define _BLOOM_CUH_

#define KEY_EMPTY 0


#include <cuda.h>
#include "hashutil.cuh"
#include <cmath>
#include "MurmurHash3.cuh"

#define BIG_PRIME 2702173

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
	double p = .0009; 
  	

	double num = log(p);
  double denom = 0.480453013918201; // ln(2)^2
  double bpe = -(num / denom);

  double dentries = (double)ht_capacity;
   uint64_t num_bits = (uint64_t)(dentries * bpe);

  //capacity = - std::ceil((ht_capacity * std::log(p)) / std::pow(std::log(2), 2));



  //k = 1.0 * capacity/ ht_capacity * std::log(2);

  capacity = (uint64_t) (dentries * bpe);
  k = (int)ceil(0.693147180559945 * bpe);

  //k=4;

  capacity = capacity*4;

  //capacity = 4 * capacity;

  cudaMalloc(&keys, capacity * sizeof(uint8_t));
  cudaMemset((void *)keys, 0, capacity * sizeof(uint8_t));
  // cudaMalloc(&vals, capacity * sizeof(uint64_t));
  // cudaMemset(vals, 0, capacity * sizeof(uint64_t));

  printf("%llu bytes requested for %llu, %d hashes\n", capacity, ht_capacity, k);
  printf("Bits per item: %d\n", capacity/ht_capacity);

  printf("These give fp k %f\n", std::pow(.5, k));
  //printf("These give fp m/n %f\n", std::pow(.5, 1.0*capacity/ht_capacity));

}



void bloomCuda::clear() {
  cudaFree((void *)keys);
  //cudaFree(vals);
}



//use murmurhash to fill two uint64_t with random bits
__device__ void get_murmurbits(uint64_t * key, uint64_t &first, uint64_t &second){


	uint64_t out[2];

	MurmurHash3_x86_128 (key, 8, 5, out);

	first = out[0];
	second = out[1];


}

__device__ uint64_t get_nth_hash(uint64_t i, uint64_t first, uint64_t second){


		return first + i*second;
}


__global__ void bulk_insert_kernel(uint8_t * keys, uint64_t capacity, uint64_t k, uint64_t * inp_keys, uint64_t num_items){

	uint64_t threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf

	uint64_t first;
	uint64_t second;

	get_murmurbits(&key, first, second);


	for (uint64_t i=0; i < k; i++){
		uint64_t slot = get_nth_hash(i, first, second) % capacity;

		//uint64_t slot = hash_64(key + i*BIG_PRIME, 0xFFFFFFFF) % capacity;
		 keys[slot] = 1;
	}
	
	//insert with quadprobe


}


//the hmh2 table uses quadratic probing
__host__ void bloomCuda::bulk_insert(uint64_t * inp_keys, uint64_t num_items){


	printf("Capacity: %llu k: %d\n", capacity, k);
	bulk_insert_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, k, inp_keys, num_items);

}

__global__ void bulk_find_kernel(uint8_t * keys, uint64_t capacity, uint64_t k, uint64_t * inp_keys, uint64_t num_items, uint64_t * misses){

	uint64_t threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf
	uint64_t first;
	uint64_t second;

	get_murmurbits(&key, first, second);
	for (uint64_t i=0; i < k; i++){
		 uint64_t slot = get_nth_hash(i, first, second) % capacity;
		 //uint64_t slot = hash_64(key + i*BIG_PRIME, 0xFFFFFFFF) % capacity;
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
  misses[0] = 0;


  printf("Capacity: %llu k: %d\n", capacity, k);
	bulk_find_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, k, inp_keys, num_items, misses);

	cudaDeviceSynchronize();

	uint64_t toReturn = *misses;
	cudaFree(misses);

	return toReturn;
}

#endif