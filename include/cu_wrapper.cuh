/*
 * ============================================================================
 *
 *        Authors:  Hunter McCoy <hjmccoy@lbl.gov>
 *                  
 *
 * ============================================================================
 */

#ifndef CURAND_WRAPPER_CUH
#define CURAND_WRAPPER_CUH
#endif



#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>

struct curand_generator {


	void init_curand(uint64_t seed, int rand_type, uint64_t backing_size);
	void gen_next_batch(uint64_t noutputs);
	void reset_to_defualt();
	__device__ uint64_t get_next(uint64_t tid);
	uint64_t * yield_backing();

	void destroy();

private:


	uint64_t * backing;
	uint64_t state;
	uint64_t seed;
	uint64_t backing_size;
	curandGenerator_t * gen;
	int type;


};

//STATE TYPES
//state 0: uniform pregen
//state 1: streaming uniform pregen


//backing size must be large enough to satisfy one full request set


//for now, keep as global-ish entity wrapped in this file
void curand_generator::init_curand(uint64_t inp_seed, int rand_type, uint64_t _backing_size){

	//malloc backing
	seed = inp_seed;
	state = rand_type;

	backing_size = _backing_size;

	curandGenerator_t temp_generator;
	gen = &temp_generator;

	uint64_t * temp_backing;
	cudaMalloc((void **) &temp_backing,backing_size*sizeof(uint64_t));
	curandCreateGenerator(gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
	curandSetPseudoRandomGeneratorSeed(*gen, seed);
	backing = temp_backing;

}

void curand_generator::gen_next_batch(uint64_t noutputs){


	if (state==0 || state == 1){

		//this may not work
		curandStatus_t status = curandGenerateLongLong(*gen, (unsigned long long *) backing, backing_size);

		if (status == CURAND_STATUS_NOT_INITIALIZED){
			printf("Not init\n");
		}
		if (status == CURAND_STATUS_PREEXISTING_FAILURE){
			printf("Prev failure\n");
		}
		if (status == CURAND_STATUS_LENGTH_NOT_MULTIPLE){
			printf("Not multiple\n");
		}
		if (status == CURAND_STATUS_LAUNCH_FAILURE){
			printf("generic failure\n");
		}
		if (status == CURAND_STATUS_TYPE_ERROR){
			printf("Not 64bit\n");
		}
		if (status == CURAND_STATUS_SUCCESS){
			printf("No failure\n");
		}


	} else {

		printf("generator not configured for this type yet.\n");
		abort();
	}



}

void curand_generator::reset_to_defualt(){

	curandSetPseudoRandomGeneratorSeed(*gen, seed);

}

__device__ uint64_t curand_generator::get_next(uint64_t tid){

	return backing[tid];

}

uint64_t * curand_generator::yield_backing(){

	if (state ==0){
		return backing;
	} else if (state == 1){

		uint64_t * temp_backing = backing;
		uint64_t * new_backing;
		cudaMalloc((void **) &new_backing, backing_size*sizeof(uint64_t));
		backing = new_backing;

		return temp_backing;



	} else {

		printf("generator not configured for this type yet.\n");
	}
	
}

void curand_generator::destroy(){

	curandDestroyGenerator(*gen);
	cudaFree(backing);


}

