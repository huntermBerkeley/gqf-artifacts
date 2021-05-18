#include "include/gqf_call.cuh"

extern C{
	#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/gqf_file.h"
#include "hashutil.h"

}
#define BITMASK(nbits)((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))


__global__ void hash_all(uint64_t* vals, uint nvals) {
	int idx = threadIdx + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < nvals; i += stride) {
		vals[i] = hash_64(vals[i], BITMASK(nhashbits));
	}
}
__global__ void qf_insert_gpu() {QF *qf, uint64_t* keys, uint64_t value, uint64_t count uint64_t nvals, uint8_t flags) {



 

}