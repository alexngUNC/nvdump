#include <stdio.h>
#include <sys/wait.h>
#include "test.h"
#include "testbench.h"
#include <unistd.h>
#define PERCENTAGE_SHARED 1


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
    
    // Print TMD struct field
    pid_t child_pid;

    // Fork the process
    if ((child_pid = fork()) == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (child_pid == 0) {
        // Execute the child TMD printing process
        char* startArgs[2];
        startArgs[0] = "start";
        startArgs[1] = NULL;
        if (execvp("./start", startArgs) == -1) {
            perror("execvp");
            exit(EXIT_FAILURE);
        }
        exit(EXIT_SUCCESS);
    } else {
        // Wait for the TMD print process to complete
        waitpid(child_pid, NULL, 0);
    }

    // Partition the L1 cache to allocate the desired amount of shared memory
    int carveout = (int) 100.0 * PERCENTAGE_SHARED;
    SAFE(cudaFuncSetAttribute(vecAdd, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

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
