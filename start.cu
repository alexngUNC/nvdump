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

    // Prompt the user for input
    printf("Enter a 64-bit unsigned integer: ");

    // Use scanf to read the input
    if (scanf("%" SCNu64, &userValue) != 1) {
        // Handle invalid input
        printf("Invalid input. Please enter a valid 64-bit unsigned integer.\n");
        return 1;
    }

    // Print TMD field
    print_tmd_field(userValue);
    trigger_callback<<<1,1>>>();

    return 0;
}
