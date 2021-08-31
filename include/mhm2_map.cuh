
#ifndef _HASH_CUH_
#define _HASH_CUH_

#define KEY_EMPTY 0

#include <cuda.h>
#include "hashutil.cuh"


//a modified version of the hash table found in mhm2, this only stores keys so that performance is comparable with the other data structures
struct mhm2CountsMap {
  // Arrays for keys and values. They are separate because the keys get initialized with max number and the vals with zero
public:

  void init(uint64_t ht_capacity);
  void clear();
  __host__ void bulk_insert(uint64_t * inp_keys, uint64_t num_items);
  __host__ uint64_t bulk_find(uint64_t * inp_keys, uint64_t num_items);


  //__global__ void bulk_find_kernel(uint64_t * inp_keys, uint64_t num_items, volatile uint64_t * misses);

  //__global__ void bulk_insert_kernel(uint64_t * inp_keys, uint64_t num_items);

  uint64_t *keys = nullptr;
  //uint64_t *vals = nullptr;

  //bug - capacity lives on the cpu
  uint64_t capacity = 0;
  int num = 0;
  
};


//capacity is n 
void mhm2CountsMap::init(uint64_t ht_capacity) {
  capacity = ht_capacity;
  cudaMalloc(&keys, capacity * sizeof(uint64_t));
  cudaMemset((void *)keys, KEY_EMPTY, capacity * sizeof(uint64_t));
  // cudaMalloc(&vals, capacity * sizeof(uint64_t));
  // cudaMemset(vals, 0, capacity * sizeof(uint64_t));
}



void mhm2CountsMap::clear() {
  cudaFree((void *)keys);
  //cudaFree(vals);
}

__global__ void bulk_insert_kernel(uint64_t * keys, uint64_t capacity, uint64_t * inp_keys, uint64_t num_items){

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf

	//one is seed, may want to pass in with map constructor
	uint64_t start_slot = MurmurHash64A(((void *)&key), sizeof(key), 1) % capacity;
	uint64_t slot = start_slot;
	//insert with quadprobe
	const int MAX_PROBE = (capacity < 200 ? capacity : 200);
	uint64_t old_key = key;
	for (int j =0; j < MAX_PROBE; j++){


		old_key = atomicCAS((unsigned long long int *)keys+slot, KEY_EMPTY, key); 

		//success
		if (old_key == KEY_EMPTY){

			__threadfence();
			return;
		}

		//fail, move to next quadratic slot
		//could do power of 2 table
		slot = (start_slot+j*j) % capacity;


	}

	//abort if a kernel doesn't insert
	__trap();
}


//the hmh2 table uses quadratic probing
__host__ void mhm2CountsMap::bulk_insert(uint64_t * inp_keys, uint64_t num_items){

	bulk_insert_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity, inp_keys, num_items);

}

__global__ void bulk_find_kernel(uint64_t * keys, uint64_t capacity, uint64_t * inp_keys, uint64_t num_items, uint64_t * misses){

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid >= num_items) return;


	uint64_t key = inp_keys[threadid];

	//samehash as cqf

	//one is seed, may want to pass in with map constructor
	uint64_t start_slot = MurmurHash64A(((void *)&key), sizeof(key), 1) % capacity;
	uint64_t slot = start_slot;
	//insert with quadprobe
	const int MAX_PROBE = (capacity < 200 ? capacity : 200);
	for (int j =0; j < MAX_PROBE; j++){



		//success
		if (key == keys[slot]){

			
			return;
		} 

		//if the slot is empty, we should have seen it before - performance issues on the false lookup without this
		if (keys[slot] == KEY_EMPTY) break;

		//fail, move to next quadratic slot
		//could do power of 2 table
		slot = (start_slot+j*j) % capacity;


	}

	//not found
	atomicAdd((unsigned long long int *) misses, 1);
}


//the hmh2 table uses quadratic probing
__host__ uint64_t mhm2CountsMap::bulk_find(uint64_t * inp_keys, uint64_t num_items){

	uint64_t * misses;
	//this is fine, should never be triggered
  	cudaMallocManaged((void **)&misses, sizeof(uint64_t));
	bulk_find_kernel<<<(num_items-1)/512+1, 512>>>(keys, capacity,inp_keys, num_items, misses);

	cudaDeviceSynchronize();

	uint64_t toReturn = *misses;
	cudaFree(misses);

	return toReturn;
}

#endif