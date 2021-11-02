/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <openssl/rand.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <assert.h> 
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <chrono>

//include for cuRand generators
#include "include/cu_wrapper.cuh"



#include "include/gqf_wrapper.cuh"
#include "include/approx_wrapper.cuh"
#include "include/rsqf_wrapper.cuh"
#include "include/sqf_wrapper.cuh"
#include "include/bloom_wrapper.cuh"



#ifndef  USE_MYRANDOM
#define RFUN random
#define RSEED srandom
#else
#define RFUN myrandom
#define RSEED mysrandom



static unsigned int m_z = 1;
static unsigned int m_w = 1;
static void mysrandom (unsigned int seed) {
	m_z = seed;
	m_w = (seed<<16) + (seed >> 16);
}

static long myrandom()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return ((m_z << 16) + m_w) % 0x7FFFFFFF;
}
#endif

static float tdiff (struct timeval *start, struct timeval *end) {
	return (end->tv_sec-start->tv_sec) +1e-6*(end->tv_usec - start->tv_usec);
}




typedef void * (*rand_init)(uint64_t maxoutputs, __uint128_t maxvalue, void *params);
typedef int (*gen_rand)(void *state, uint64_t noutputs, __uint128_t *outputs);
typedef void * (*duplicate_rand)(void *state);


//filter ops
typedef int (*init_op)(uint64_t nvals, uint64_t hash, uint64_t buf_size);
typedef int (*destroy_op)();
typedef int (*bulk_insert_op)(uint64_t * vals, uint64_t nvals);
typedef uint64_t (*bulk_find_op)(uint64_t * vals, uint64_t nvals);


typedef struct rand_generator {
	rand_init init;
	gen_rand gen;
	duplicate_rand dup;
} rand_generator;

typedef struct filter {
	init_op init;
	destroy_op destroy;
	bulk_insert_op bulk_insert;
	bulk_find_op bulk_lookup;

} filter;

typedef struct uniform_pregen_state {
	uint64_t maxoutputs;
	uint64_t nextoutput;
	__uint128_t *outputs;
} uniform_pregen_state;

typedef struct uniform_online_state {
	uint64_t maxoutputs;
	uint64_t maxvalue;
	unsigned int seed;
	char *buf;
	int STATELEN;
	struct random_data *rand_state;
} uniform_online_state;

typedef struct zipf_params {
	double exp;
	long universe;
	long sample;
} zipf_params;

typedef struct zipfian_pregen_state {
	zipf_params *params;
	uint64_t maxoutputs;
	uint64_t nextoutput;
	__uint128_t *outputs;
} zipfian_pregen_state;

typedef struct app_params {
	char *ip_file;
	int num;
} app_params;

typedef struct app_pregen_state {
	app_params *params;
	uint64_t maxoutputs;
	uint64_t nextoutput;
	__uint128_t *outputs;
} app_pregen_state;





filter gqf = {
	gqf_init,
	gqf_destroy,
	gqf_bulk_insert,
	gqf_bulk_get
};

// filter mhm2_map = {
// 	map_init,
// 	map_destroy,
// 	map_bulk_insert,
// 	map_bulk_get
// };

filter bloom = {
	bloom_init,
	bloom_destroy,
	bloom_bulk_insert,
	bloom_bulk_get
};

// filter one_bit_bloom = {
// 	one_bit_bloom_init,
// 	one_bit_bloom_insert,
// 	one_bit_bloom_lookup,
// 	one_bit_bloom_range,
// 	one_bit_bloom_destroy,
// 	one_bit_bloom_iterator,
// 	one_bit_bloom_get,
// 	one_bit_bloom_next,
// 	one_bit_bloom_end,
// 	one_bit_bloom_bulk_insert,
// 	one_bit_bloom_prep_vals,
// 	one_bit_bloom_bulk_get,
// 	one_bit_bloom_xnslots
// };

filter point = {
	point_init,
	point_destroy,
	point_bulk_insert,
	point_bulk_get
};


filter rsqf = {
	rsqf_init,
	rsqf_destroy,
	rsqf_bulk_insert,
	rsqf_bulk_get
};

filter sqf = {
	sqf_init,
	sqf_destroy,
	sqf_bulk_insert,
	sqf_bulk_get
};

uint64_t * zipfian_backing;


uint64_t tv2msec(struct timeval tv)
{
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int cmp_uint64_t(const void *a, const void *b)
{
	const uint64_t *ua = (const uint64_t*)a, *ub = (const uint64_t *)b;
	return *ua < *ub ? -1 : *ua == *ub ? 0 : 1;
}

void usage(char *name)
{
	printf("%s [OPTIONS]\n"
				 "Options are:\n"
				 "  -n nslots     [ log_2 of filter capacity.  Default 22 ]\n"
				 "  -r nruns      [ number of runs.  Default 1 ]\n"
				 "  -p npoints    [ number of points on the graph.  Default 20 ]\n"
				 "  -b buf_size   [ log_2 of buffer capacity, default is nslots/npoints ]\n"
				 "  -m randmode   [ Data distribution, one of \n"
				 "                    uniform_pregen\n"
				 "                    uniform_online\n"
				 "                    zipfian_pregen\n"
				 "                    custom_pregen\n"
				 "                  Default uniform_pregen ]\n"
				 "  -d datastruct  [ Default gqf] [ gqf | map | bloom | bloom_one_bit ]\n"
				 "  -a number of filters for merging  [ Default 0 ] [Optional]\n"
				 "  -f outputfile  [ Default gqf. ]\n"
				 "  -i input file for app specific benchmark [Optional]\n"
				 "  -v num of values in the input file [Optional]\n"
				 "  -u universe for zipfian distribution  [ Default nvals ] [Optional]\n"
				 "  -s constant for zipfian distribution  [ Default 1.5 ] [Optional]\n",
				 name);
}

int main(int argc, char **argv)
{


	uint32_t nbits = 22, nruns = 1;
	unsigned int npoints = 20;
	uint64_t nslots = (1ULL << nbits), nvals = 950*nslots/1000;
	double s = 1.5; long universe = nvals;
	int numvals = 0;
	int numfilters = 0;
	char *randmode = "uniform_pregen";
	char *datastruct = "gqf";
	char *outputfile = "gqf";
	char *inputfile = "gqf";
	void *param = NULL;

	filter filter_ds;

	unsigned int i, j, exp, run;
	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_insert[100][1];
	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_exit_lookup[100][1];
	struct std::chrono::time_point<std::chrono::high_resolution_clock> tv_false_lookup[100][1];

	struct std::chrono::duration<int64_t, std::nano> insert_times[100];
	struct std::chrono::duration<int64_t, std::nano> exit_times[100];
	struct std::chrono::duration<int64_t, std::nano> false_times[100];


	uint64_t fps = 0;
	//default buffer of 20;
	uint64_t buf_bits = 20;
	uint64_t buf_size = (1ULL << 20);


	#ifndef __x86_64

	printf("Detected IBM version\n");


	const char *dir = "/gpfs/alpine/bif115/scratch/hjmccoy/";

	printf("Writing files to %s\n", dir);

	#else 

	const char *dir = "./";

	#endif
	
	const char *insert_op = "-insert.txt\0";
	const char *exit_lookup_op = "-exists-lookup.txt\0";
	const char *false_lookup_op = "-false-lookup.txt\0";
	char filename_insert[256];
	char filename_exit_lookup[256];
	char filename_false_lookup[256];

	/* Argument parsing */
	int opt;
	char *term;

	while((opt = getopt(argc, argv, "n:r:p:b:m:d:a:f:i:v:s")) != -1) {
		switch(opt) {
			case 'n':
				nbits = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -n must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				nslots = (1ULL << nbits);
				nvals = 950*nslots/1000;
				universe = nvals;
				//buf_size = nbits - log2(npoints);
				break;
			case 'r':
				nruns = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -r must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				break;
			case 'p':
				npoints = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -p must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				break;
			case 'b':
				buf_bits = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -n must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				buf_size = (1ULL << buf_bits);
				break;
			case 'm':
				randmode = optarg;
				break;
			case 'd':
				datastruct = optarg;
				break;
			case 'f':
				outputfile = optarg;
				break;
			case 'i':
				inputfile = optarg;
				break;
			case 'v':
				numvals = (int)strtol(optarg, &term, 10);
				break;
			case 's':
				s = strtod(optarg, NULL);
				break;
			case 'u':
				universe = strtol(optarg, &term, 10);
				break;
			case 'a':
				numfilters = strtol(optarg, &term, 10);
				if (*term) {
					fprintf(stderr, "Argument to -p must be an integer\n");
					usage(argv[0]);
					exit(1);
				}
				break;
			default:
				fprintf(stderr, "Unknown option\n");
				usage(argv[0]);
				exit(1);
				break;
		}
	}




	if (strcmp(datastruct, "gqf") == 0) {
		filter_ds = gqf;
	} else if (strcmp(datastruct, "bloom") == 0) {
		filter_ds = bloom;
	} else if (strcmp(datastruct, "point") == 0) {
		filter_ds = point;
	} else if (strcmp(datastruct, "rsqf") == 0) {
		filter_ds = rsqf;
	} else if (strcmp(datastruct, "sqf") == 0) {
		filter_ds = sqf;
	} else {
		fprintf(stderr, "Unknown filter.\n");
		usage(argv[0]);
		exit(1);
	}
	

	snprintf(filename_insert, strlen(dir) + strlen(outputfile) + strlen(insert_op) + strlen(datastruct) + 1, "%s%s%s", dir, outputfile, insert_op);
	snprintf(filename_exit_lookup, strlen(dir) + strlen(outputfile) + strlen(exit_lookup_op) + 1, "%s%s%s", dir, outputfile, exit_lookup_op);

	snprintf(filename_false_lookup, strlen(dir) + strlen(outputfile) + strlen(false_lookup_op) + 1, "%s%s%s", dir, outputfile, false_lookup_op);


	FILE *fp_insert = fopen(filename_insert, "w");
	FILE *fp_exit_lookup = fopen(filename_exit_lookup, "w");
	FILE *fp_false_lookup = fopen(filename_false_lookup, "w");

	if (fp_insert == NULL) {
		printf("Can't open the data file %s\n", filename_insert);
		exit(1);
	}

	if (fp_exit_lookup == NULL ) {
	    printf("Can't open the data file %s\n", filename_exit_lookup);
		exit(1);
	}

	if (fp_false_lookup == NULL) {
		printf("Can't open the data file %s\n", filename_false_lookup);
		exit(1);
	}


	for (run = 0; run < nruns; run++) {
		fps = 0;
		filter_ds.init(nbits, nbits+8, buf_size);
		

		//run setup here
		// vals_gen_state = vals_gen->init(nvals, filter_ds.range(), param);

		// if (strcmp(randmode, "zipfian_pregen") == 0) {
		// 	for (exp =0; exp < 2*npoints; exp+=2){


		// 	i = (exp/2)*(nvals/npoints);
		// 	j = ((exp/2) + 1)*(nvals/npoints);
		// 	printf("Round: %d\n", exp/2);

		// 	for (;i < j; i += 1<<16) {
		// 		int nitems = j - i < 1<<16 ? j - i : 1<<16;
		// 		__uint128_t vals[1<<16];
		// 		int m;
		// 		assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);


		// 		}

		// 	}	
		// }





		//init curand here
		//setup for curand here
		//three generators - two clones for get/find and one random for fp testing
		curand_generator curand_put{};
		curand_put.init(run, 0, buf_size);
		curand_generator curand_get{};
		curand_get.init(run, 0, buf_size);
		curand_generator curand_false{};
		curand_false.init((run+1)*2702173, 0, buf_size);
		
		cudaDeviceSynchronize();

		sleep(1);




		for (exp = 0; exp < 2*npoints; exp += 2) {
			i = (exp/2)*(nvals/npoints);
			j = ((exp/2) + 1)*(nvals/npoints);
			//printf("Round: %d\n", exp/2);


			insert_times[exp+1] = std::chrono::duration<int64_t>::zero();;
			std::chrono::time_point<std::chrono::high_resolution_clock> insert_start;
			std::chrono::time_point<std::chrono::high_resolution_clock> insert_end;

			tv_insert[exp][run] = std::chrono::high_resolution_clock::now();

			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * vals;
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

				//prep vals for filter
				//cudaProfilerStart();

				//get timing
				insert_start = std::chrono::high_resolution_clock::now();

				curand_put.gen_next_batch(nitems);
				vals = curand_put.yield_backing();
				//cudaDeviceSynchronize();
				insert_end = std::chrono::high_resolution_clock::now();
				
				insert_times[exp+1] += insert_end-insert_start;

			
					
					
				filter_ds.bulk_insert(vals, nitems);
					//cudaProfilerStop();

				
			}

			cudaDeviceSynchronize();

			tv_insert[exp+1][run] = std::chrono::high_resolution_clock::now();

			//don't need this
			//curand_test.reset_to_defualt();

			exit_times[exp+1] =  std::chrono::duration<int64_t>::zero();
			std::chrono::time_point<std::chrono::high_resolution_clock> exit_start;
			std::chrono::time_point<std::chrono::high_resolution_clock> exit_end;

			

			i = (exp/2)*(nvals/npoints);
			tv_exit_lookup[exp][run]= std::chrono::high_resolution_clock::now();
			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				
				int m;
				//assert(vals_gen->gen(old_vals_gen_state, nitems, vals) == nitems);
			
		
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);
				uint64_t * insert_vals;
				//prep vals for filter

				exit_start = std::chrono::high_resolution_clock::now();
				curand_get.gen_next_batch(nitems);
				insert_vals = curand_get.yield_backing();
				exit_end = std::chrono::high_resolution_clock::now();
				exit_times[exp+1] += exit_end-exit_start;

				uint64_t result = filter_ds.bulk_lookup(insert_vals, nitems);
				//uint64_t result = 0;
				if (result != 0){

				printf("Failed to find %llu items\n", result);
				abort();

				}

			}

			cudaDeviceSynchronize();
			tv_exit_lookup[exp+1][run] = std::chrono::high_resolution_clock::now();

			//this looks right
			false_times[exp+1] = std::chrono::duration<int64_t>::zero();;
			std::chrono::time_point<std::chrono::high_resolution_clock> false_start;
			std::chrono::time_point<std::chrono::high_resolution_clock> false_end;

			//curand_test.destroy();
			//curand_generator othervals_curand{};
			//othervals_curand.init_curand(5, 0, buf_size);

			i = (exp/2)*(nvals/npoints);
			tv_false_lookup[exp][run] = std::chrono::high_resolution_clock::now();
			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * othervals;
				int m;

				false_start = std::chrono::high_resolution_clock::now();
				curand_false.gen_next_batch(nitems);
				othervals = curand_false.yield_backing();
				false_end = std::chrono::high_resolution_clock::now();
				false_times[exp+1] += false_end-false_start;

		

					
				fps += nitems-filter_ds.bulk_lookup(othervals, nitems);
					
				
			}

			cudaDeviceSynchronize();
			tv_false_lookup[exp+1][run] = std::chrono::high_resolution_clock::now();
		}


		//and destroy

		//all inserts done, reset main counter




		curand_put.destroy();
		curand_get.destroy();
		curand_false.destroy();

		
		
		curand_generator get_end{};
		get_end.init(run, 0, buf_size);
		
		for (exp = 0; exp < 2*npoints; exp += 2) {
			i = (exp/2)*(nvals/npoints);
			j = ((exp/2) + 1)*(nvals/npoints);
			//printf("Round: %d\n", exp/2);


			for (;i < j; i += buf_size) {
				int nitems = j - i < buf_size ? j - i : buf_size;
				uint64_t * vals;
				//int m;
				//assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

				//prep vals for filter
				//cudaProfilerStart();
				get_end.gen_next_batch(nitems);
				vals = get_end.yield_backing();

				uint64_t result = filter_ds.bulk_lookup(vals, nitems);
				if (result != 0){

					printf("Failed to find %llu items\n", result);
					abort();

				}


			}	

		}
		
		get_end.destroy();
		
		filter_ds.destroy();
		cudaDeviceSynchronize();


	}



	printf("Wiring results to file: %s\n",  filename_insert);
	fprintf(fp_insert, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_insert, "    y_%d", run);
	}
	fprintf(fp_insert, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_insert, "%d", ((exp/2)*(100/npoints)));
		for (run = 0; run < nruns; run++) {
			fprintf(fp_insert, " %f",
							0.001 * (nvals/npoints)/ ((tv_insert[exp+1][run] - tv_insert[exp][run])-insert_times[exp+1]).count()*1000000);
		}
		fprintf(fp_insert, "\n");
	}
	printf("Insert Performance written\n");

	printf("Wiring results to file: %s\n", filename_exit_lookup);
	fprintf(fp_exit_lookup, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_exit_lookup, "    y_%d", run);
	}
	fprintf(fp_exit_lookup, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_exit_lookup, "%d", ((exp/2)*(100/npoints)));
		for (run = 0; run < nruns; run++) {
			fprintf(fp_exit_lookup, " %f",
							0.001 * (nvals/npoints)/((tv_exit_lookup[exp+1][run]- tv_exit_lookup[exp][run])-exit_times[exp+1]).count()*1000000);
		}
		fprintf(fp_exit_lookup, "\n");
	}
	printf("Existing Lookup Performance written\n");

	printf("Wiring results to file: %s\n", filename_false_lookup);
	fprintf(fp_false_lookup, "x_0");
	for (run = 0; run < nruns; run++) {
		fprintf(fp_false_lookup, "    y_%d", run);
	}
	fprintf(fp_false_lookup, "\n");
	for (exp = 0; exp < 2*npoints; exp += 2) {
		fprintf(fp_false_lookup, "%d", ((exp/2)*(100/npoints)));
		for (run = 0; run < nruns; run++) {
			fprintf(fp_false_lookup, " %f",
							0.001 * (nvals/npoints)/((tv_false_lookup[exp+1][run]- tv_false_lookup[exp][run])-false_times[exp+1]).count()*1000000);
		}
		fprintf(fp_false_lookup, "\n");
	}
	printf("False Lookup Performance written\n");

	printf("FP rate: %f (%lu/%lu)\n", 1.0 * fps / nvals, fps, nvals);

	fclose(fp_insert);
	fclose(fp_exit_lookup);
	fclose(fp_false_lookup);

	return 0;
}
