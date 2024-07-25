#include "testbench.h"
#include "tmdOld.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>

// Global pointer to payload for TMD field
uint64_t* p_payload = NULL;
uint64_t payload;

void __global__ trigger_callback() {}

int
main(void)
{
    // User will input their bit index
    uint64_t userValue;
    uint64_t fieldSize;

    prompt:
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
    
        // Check if the user wants to modify the value of the TMD field
        getchar();
    
        char input[20];
    
        printf("Enter a uint64_t value: ");
        if (fgets(input, sizeof(input), stdin) == NULL) {
            // Handle error or end-of-file condition
            fprintf(stderr, "Error reading input\n");
            return 1;
        }
    
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

    trigger_callback<<<1,1>>>();
    cudaDeviceSynchronize();
    goto prompt;

    return 0;
}
