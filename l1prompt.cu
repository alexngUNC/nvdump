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
main() {
    // User will input their bit index
    uint64_t startingBit;
    uint64_t fieldSize;

    // Prompt the user for input
    printf("Enter TMD field starting bit number: ");

    // Read user input
    if (scanf("%" SCNu64, &startingBit) != 1) {
        printf("Invalid input. Please enter a valid 64-bit unsigned integer.\n");
        return 1;
    }

    printf("Enter TMD field size in bits: ");
    
    if (scanf("%" SCNu64, &fieldSize) != 1) {
        printf("Invalid input. Please enter a valid 64-bit unsigned integer.\n");
        return 1;
    } 

    getchar();

    char input[20];

    printf("Enter a uint64_t value: ");
    if (fgets(input, sizeof(input), stdin) == NULL) {
        // Handle error or end-of-file condition
        fprintf(stderr, "Error reading input\n");
        return 1;
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
    if (input[0] == '\n') {
        printf("Value will not be modified\n");
	    extract_tmd_field(startingBit, fieldSize, NULL);
    } else {
        char *endptr;
        uint64_t payload = strtoull(input, &endptr, 10);
        if (*endptr != '\0' && *endptr != '\n') {
            fprintf(stderr, "Error: invalid characters found in the input\n");
            return 1;
        }

        uint64_t *p_payload = &payload;
        printf("Value will be modified to: %" PRIu64 "\n", *p_payload);
	    extract_tmd_field(544, 18, p_payload);
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
