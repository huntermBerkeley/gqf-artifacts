#include cuda.h


__global__ void qf_insert_gpu(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint8_t
	flags);