#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "testbench.h"
#include "tmdOld.h"
#include <unistd.h>
#define PERCENTAGE_SHARED 1

// Global pointer to payload for TMD field
uint64_t* p_payload = NULL;
uint64_t payload;

__global__ void vecAdd(int* x, int* y, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int z[];
    if (i < n) {
        z[i] = x[i] + y[i];
        y[i] = z[i];
    }
}


int main() {
    // For checking errors via SAFE
    cudaError_t err;

    // Vector lengths
    int n = 8000;

    // Allocate host memory for x
    int* h_x = (int*) malloc(sizeof(int) * n);
    if (h_x == NULL) {
        fprintf(stderr, "Couldn't allocate host memory for x\n");
        return 1;
    }

    // Allocate host memory for y
    int* h_y = (int*) malloc(sizeof(int) * n);
    if (h_y == NULL) {
        fprintf(stderr, "Couldn't allocate host memory for y\n");
        return 1;
    }

    // Initialize vectors
    for (int i=0; i<n; i++) {
        h_x[i] = i;
        h_y[i] = i;
    }

    // Allocate device memory for x
    int* d_x;
    SAFE(cudaMalloc(&d_x, sizeof(int)*n));

    // Allocate device memory for y
    int* d_y;
    SAFE(cudaMalloc(&d_y, sizeof(int)*n));

    // Copy memory from host to device
    SAFE(cudaMemcpy(d_x, h_x, sizeof(int)*n, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_y, h_y, sizeof(int)*n, cudaMemcpyHostToDevice));

    // User will input their bit index
    uint64_t userValue;
    uint64_t fieldSize;

    // Prompt the user for input
    printf("Enter TMD field starting bit number: ");

    // Read user input
    if (scanf("%" SCNu64, &userValue) != 1) {
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

    // Partition the L1 cache to allocate the desired amount of shared memory
    int carveout = (int) 100.0 * PERCENTAGE_SHARED;
    SAFE(cudaFuncSetAttribute(vecAdd, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

    // Check if the user wants to modify the TMD field
    if (input[0] == '\n') {
        printf("Value will not be modified\n");

        // Print TMD field
        print_tmd_field(userValue, fieldSize, NULL);
    } else {
        printf("Value will be modified\n");

        // Convert the input to uint64_t
        char *endptr;
        payload = strtoull(input, &endptr, 10);
        p_payload = &payload;

        // Check for conversion errors
        if (*endptr != '\0' && *endptr != '\n') {
            fprintf(stderr, "Invalid input. Please enter a valid uint64_t.\n");
            return 1;
        }

        // Print TMD field
        print_tmd_field(userValue, fieldSize, p_payload);
    }



    // Launch vector addition kernel
    vecAdd<<<8, 1024, n*sizeof(int)>>>(d_x, d_y, n);

    // Copy memory from device to host
    SAFE(cudaMemcpy(h_y, d_y, sizeof(int)*n, cudaMemcpyDeviceToHost));

    // Print first 5 results
    for (int i=0; i<5; i++) {
      printf("y[%d]=%d\n", i, h_y[i]);
    }

    // Free host memory
    free(h_x);
    free(h_y);
    return 0;   
}
