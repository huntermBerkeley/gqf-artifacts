/*
 * ============================================================================
 *
 *        Authors:  
 *					Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <openssl/rand.h>
#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

#include "include/gqf_int.cuh"
#include "hashutil.cuh"
#include "include/gqf.cuh"
//#include "src/gqf.cu"
#include <fstream>
#include <string>
#include <algorithm>

#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Please specify the log of the number of slots in the CQF.\n");

		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies] filename\n");

		exit(1);

	}
	if (argc < 3){

		fprintf(stderr, "Please specify 'bulk' or 'reduce'\n");
		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies] filename\n");

		exit(1);
	}

	if (argc < 4) {
		fprintf(stderr, "Please specify random or preset data.\n");
		printf("Usage: ./gqf_only 28 [nbits] 0 [0 bulk, 1 reduce] 0 [0 random, 1 file, 2 random copies] filename\n");

		exit(1);

	}

	printf("This is a test to show how the GQF performs on skewed data.\n");
	printf("Most testing is handled by test.cu which has optimized .\n");
	printf("Generation is done using RAND_bytes and a single CPU thread, expect a long  wait while data is generated.\n");
	printf("For the Zipfian data used in our experiments, email hjmccoy@lbl.gov");


	QF qf;
	uint64_t qbits = atoi(argv[1]);
	uint64_t rbits = 8;
	uint64_t nhashbits = qbits + rbits;
	uint64_t nslots = (1ULL << qbits);
	//this can be changed to change the % it fills up
	uint64_t nvals = 95 * nslots / 100;
	//uint64_t nvals =  nslots/2;
	//uint64_t nvals = 4;
	//uint64_t nvals = 1;
	uint64_t* vals;

	uint64_t * nums;
	uint64_t * counts;

	/* Initialise the CQF */
	if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, false, 0)) {
		fprintf(stderr, "Can't allocate CQF.\n");
		abort();
	}


	bool bulk = true;

	if (atoi(argv[2]) != 0){

		printf("Using reduce.\n");
		bulk = false;

	}


	//check if pregen
	int preset = atoi(argv[3]);


	nums = (uint64_t*)malloc(nvals * sizeof(vals[0]));
	counts = (uint64_t*)malloc(nvals * sizeof(vals[0]));

	vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
	uint64_t i = 0;

	qf_set_auto_resize(&qf, false);

	if (preset == 1){

		printf("Using preset data\n");

		std::fstream preset_data;

		preset_data.open(argv[4]);

		if (preset_data.is_open()){
			std::string tp;
			while(std::getline(preset_data, tp) && i < nvals){

				char * end;
				vals[i] = std::strtoull(tp.c_str(), &end, 10);
				vals[i] = (1 * vals[i]) % qf.metadata->range;
				i++;
			}





		} else {
			printf("Error opening file %s\n", argv[4]);
		}

		preset_data.close();

		if (i < nvals);

		nvals = i;


	


	} else if (preset == 2){




		RAND_bytes((unsigned char*)nums, sizeof(*vals) * nvals);
		RAND_bytes((unsigned char*)counts, sizeof(*vals) * nvals);


		printf("Generated backing data\n");
		uint64_t cap = 10;


		uint64_t i = 0;

		while (i < nvals){


			uint64_t num;
			uint64_t count;
			

			num = (1 * nums[i]) % qf.metadata->range;

			count = (1 * counts[i]) % cap + 1;

			assert(count > 0);

			for (uint64_t j =i; j < i+count; j++){

				if (j < nvals) vals[j] = num;

			}

			i+=count;



		}

		//shuffle vals
		std::random_device rd;
	    std::mt19937 g(rd());
	 
		std::shuffle(vals, vals+nvals, g);

	} else {

		printf("Using regular data\n");

		/* Generate random values */
		
		RAND_bytes((unsigned char*)vals, sizeof(*vals) * nvals);
		//uint64_t* _vals;
		for (uint64_t i = 0; i < nvals; i++) {
		vals[i] = (1 * vals[i]) % qf.metadata->range;
		//vals[i] = hash_64(vals[i], BITMASK(nhashbits));
		}

	}

	

	//copy vals to device

	uint64_t * dev_vals;

	cudaMalloc((void **)&dev_vals, nvals*sizeof(uint64_t));

	cudaMemcpy(dev_vals, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);

	// vals = (uint64_t *) malloc(nvals * sizeof(uint64_t));
	// for (uint64_t i =0l; i< nvals; i++){
	// 	vals[i] = i;
	// }

	srand(0);
	/* Insert keys in the CQF */
	printf("starting kernel\n");
	//qf_gpu_launch(&qf, vals, nvals, key_count, nhashbits, nslots);

	QF* dev_qf;
	qf_malloc_device(&dev_qf, qbits, true);
	cudaDeviceSynchronize();
	auto start = std::chrono::high_resolution_clock::now();

	if (bulk){	
		bulk_insert(dev_qf, nvals, dev_vals, 0);
	} else {
		bulk_insert_reduce(dev_qf, nvals, dev_vals, 0);
	}
	
	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Inserts per second: %f\n", nvals/diff.count());

	cudaMemcpy(dev_vals, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	start = std::chrono::high_resolution_clock::now();

	uint64_t misses = bulk_get_misses_wrapper(dev_qf, dev_vals, nvals);

	cudaDeviceSynchronize();

	end = std::chrono::high_resolution_clock::now();


  	diff = end-start;

	assert(misses == 0);

	std::cout << "Queried " << nvals << " in " << diff.count() << " seconds\n";

  	printf("Queries per second: %f\n", nvals/diff.count());



	qf_destroy_device(dev_qf);

	printf("GPU launch succeeded\n");
	fflush(stdout);


	return 0;

}