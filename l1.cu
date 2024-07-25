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
	printf("Allocating %lu bytes\n", bytes);
	SAFE(cudaMalloc(&d_a, bytes));
	SAFE(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
	
	// flag for CPU synchronization
	int *flag;
	SAFE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped));
	*flag = 1;

	// launch kernel
	extract_tmd_field(544, 18, NULL);
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
