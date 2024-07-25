#include <inttypes.h>
#include <stdio.h>
#include "testbench.h"
#include "tmd.h"

#define CONCURRENT_TB 76
#define LOOPS 32
#define TB_SIZE 512

__global__ void vecInc(float *a, int *flag) {
	size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (size_t j=0; j<LOOPS; j++) {
		a[global_idx * LOOPS + j]++;
	}
	__threadfence_system();
	if (global_idx == 0)
		*flag = 0;
	while (0) { }
}

int
main(int argc, char *argv[]) {
    uint64_t startingBit;
    uint64_t fieldSize;
    uint64_t payload;
    int modify = 0;
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Invalid usage\n");
        fprintf(stderr, "Usage: %s --start=<value> --size=<value> [--new=<value>]\n", argv[0]);
        return EXIT_FAILURE;
    }
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--start=", 8) == 0) {
            startingBit = strtoull(argv[i] + 8, NULL, 10);
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            fieldSize = strtoull(argv[i] + 7, NULL, 10);
        } else if (strncmp(argv[i], "--new=", 6) == 0) {
            payload = strtoull(argv[i] + 6, NULL, 10);
            modify = 1; // Mark that --new was provided
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s --start=<value> --size=<value> [--new=<value>]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

	// vector length
	size_t n = CONCURRENT_TB * TB_SIZE * LOOPS;
	// host memory
	float *h_a;
	h_a = (float *) malloc(n * sizeof(float));
	for (int i=0; i<n; i++) {
		h_a[i] = 1;
	}

	// device memory
	float *d_a;
	size_t bytes = n * sizeof(float);
	printf("Allocating %" PRIu64 " bytes\n", bytes);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	
	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// launch kernel
    if (modify) {
        uint64_t *p_payload = &payload;
        printf("Value will be modified to: %" PRIu64 "\n", *p_payload);
	    extract_tmd_field(544, 18, p_payload);
    } else {
        printf("Value will not be modified\n");
	    extract_tmd_field(startingBit, fieldSize, NULL);
    }
	vecInc<<<CONCURRENT_TB, TB_SIZE>>>(d_a, flag);
	//while (*flag) { /* wait for caches to saturate */ }
	printf("GPU is spinning!\n");
	SAFE(cudaDeviceSynchronize());

	// print result
	SAFE(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
	for (int i=0; i<10; i++) {
		printf("%f\t", h_a[i]);
	}
	printf("\n");

	// free memory
	SAFE(cudaFree(d_a));
	free(h_a);
	return 0;
}
