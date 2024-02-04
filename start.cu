#include "test.h"
#include "testbench.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>

void __global__ trigger_callback() {}

int
main(void)
{
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

    // Print TMD field
    print_tmd_field(userValue, fieldSize);
    trigger_callback<<<1,1>>>();

    return 0;
}
