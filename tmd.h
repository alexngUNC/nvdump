#include <cuda.h>

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <dlfcn.h>

// In functions that do not return an error code, we favor terminating with an
// error rather than merely printing a warning and continuing.
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3

//#define DEBUG 1

// Boolean if the field should be modified
int g_modify;

// Global bit index into TMD struct
uint64_t g_bit_index;

// Global field size in bits
uint64_t g_field_size;

// Global payload to change target field to
uint64_t g_payload;

// Check if TMD has been accessed
int g_tmd_accessed = 0;

// Global TMD pointer
char tmd_val = 'a';
char *g_tmd = &tmd_val;

void printTMDField(void);
void modifyTMDField(void);

static void launchCallback(void *ukwn, int domain, int cbid, const void *in_params) {
    // The third 8-byte element in `in_parms` is a pointer to the stream struct.
    // This exists even when in_params < 0x50. This could be used to implement
    // stream masking without the manual offsets specified elsewhere (store a
    // table of stream pointers to masks and do a lookup here).
    // It could also be used (although not as easily) to support global and next
    // masking on old CUDA versions, but that would require hooking earlier in the
    // launch process (before the stream mask is applied).
    if (*(uint32_t*)in_params < 0x50)
        abort(1, 0, "Unsupported CUDA version for callback-based SM masking. Aborting...");
    // The eighth 8-byte element in `in_params` is a pointer to a struct which
    // contains a pointer to the TMD as its first element. Note that this eighth
    // pointer must exist---it only exists when the first 8-byte element of
    // `in_params` is at least 0x50 (checked above).
    char* tmd = (char*) (**((uintptr_t***)in_params + 8));
    if (!tmd)
        abort(1, 0, "TMD allocation appears NULL; likely forward-compatibilty issue.\n");

    //fprintf(stderr, "cta: %lx\n", *(uint64_t*)(tmd + 74));
    // TODO: Check for supported QMD version (>XXX, <4.00)
    // TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 16 bytes (rather than 8 bytes) wide. It also requires an enable bit at +31bits.
    // uint32_t *lower_ptr = (uint32_t *) (tmd + 84);
    // uint32_t *upper_ptr = (uint32_t *) (tmd + 88);

    //  if (g_next_sm_mask) {
    //      *lower_ptr = (uint32_t)g_next_sm_mask;
    //      *upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
    //      g_next_sm_mask = 0;
    //  } else if (!*lower_ptr && !*upper_ptr){
    //      // Only apply the global mask if a per-stream mask hasn't been set
    //      *lower_ptr = (uint32_t)g_sm_mask;
    //      *upper_ptr = (uint32_t)(g_sm_mask >> 32);
    //  }
    g_tmd = tmd;
    if (g_modify) {
        modifyTMDField();
    } else {
        printTMDField();
    }
}

void printTMDField(void) {
    // Set TMD pointer
    char *tmd = g_tmd;

    // Initialize memory to store whatever is at location target
    uint64_t a = 0;
    uint64_t* target_ptr = &a;

    // Specify address of target
    int target = g_bit_index;

    // Specify number of bits that target is
    int length = g_field_size;

    // Find the closest multiple of 8 <= target
    int bottom = (target / 8) * 8;

    // Find offset in size of 8-bits
    int offset = bottom / 8;

    // Grab 64 bits starting at bottom
    //uint64_t* target_addr = ((uint64_t*)((uint32_t*)(**((char***)in_params + 8) + offset)));
    *target_ptr = *((uint64_t*)(tmd + offset));
    //*target_ptr = *((uint64_t*)((uint32_t*)(**((char***)in_params + 8) + offset)));

    // Shift right until you get the desired starting address
    int right = target - bottom;
    *target_ptr = (*target_ptr) >> right;

    // Only take the desired number of bits
    uint64_t desired_bits = 0xFFFFFFFFFFFFFFFF;
    desired_bits = desired_bits >> (64 - length);
    *target_ptr = (*target_ptr) & desired_bits;

    // Print value before modifying
    fprintf(stdout, "Previous value: %lu\n", *target_ptr);
}

void modifyTMDField(void) {
    #ifdef DEBUG
    fprintf(stdout, "target_addr: %p\t*target_addr: %lx\n", target_addr, *target_addr);
    #endif
    char *tmd = g_tmd;
    uint64_t buf = 0;
    uint64_t *target_ptr = &buf;
    int target = g_bit_index;
    int length = g_field_size;
    uint64_t payload = g_payload;
    int bottom = (target / 8) * 8;
    int offset = bottom / 8;

    uint64_t *target_addr = (uint64_t *) (tmd + offset);
    *target_ptr = *((uint64_t *) (tmd + offset));

    int right = target - bottom;
    *target_ptr = (*target_ptr) >> right;

    uint64_t desired_bits = 0xFFFFFFFFFFFFFFFF;
    desired_bits = desired_bits >> (64 - length);
    *target_ptr = (*target_ptr) & desired_bits;

    // Print value before modifying
    fprintf(stdout, "Previous value: %lu\n", *target_ptr);

    // Payload of 64-bits; need to preserve left and right segments
    // that are not to be modified
    uint64_t full_payload = 0;

    // Find number of bits to keep on left of target
    int left = 64 - (length + right);
    #ifdef DEBUG
    printf("left: %u\tright: %u\n", left, right);
    #endif

    // Preserve left segment
    uint64_t left_bits = 0xFFFFFFFFFFFFFFFF;
    if (left > 0) {
        left_bits = (left_bits >> (length + right));
        left_bits <<= (length + right);
        full_payload = (*target_addr) & left_bits;
    }

    // Preserve bits to the right of target
    uint64_t right_part;
    uint64_t right_bits = 0xFFFFFFFFFFFFFFFF;
    if (right > 0) {
        right_bits = right_bits >> (left + length);
        right_part = (*target_addr) & right_bits;
    } else {
        right_part = 0;
    }
    full_payload |= right_part;

    // Shift payload to the left
    payload = payload << right;

    // Put payload in middle of full 64-bit value
    full_payload |= payload;

    // Modify value at target_addr
    *target_addr = full_payload;
    
    // Go to the modified value
    *target_ptr = *((uint64_t*)(tmd + offset));

    // Shift right until you get the desired starting address
    *target_ptr = (*target_ptr) >> right;

    // Only take the desired number of bits
    desired_bits = 0xFFFFFFFFFFFFFFFF  >> (64 - length);
    *target_ptr = (*target_ptr) & desired_bits;

    // Print segments
    #ifdef DEBUG
    fprintf(stdout, "Right part: %lx\n", right_part);
    fprintf(stdout, "Full payload: %lx\n", full_payload);

    // Print only the target value
    fprintf(stdout, "target_addr: %p\t*target_addr: %lx\n", target_addr, *target_addr);
    #endif
    fprintf(stdout, "Modified value: %lu\n", *target_ptr);
}

void set_bit_index(uint64_t bit_index) {
    g_bit_index = bit_index;
}

void set_field_size(uint64_t bit_length) {
    g_field_size = bit_length;
}

static char extract_tmd_field_called = 0;
static void extract_tmd_field(uint64_t bit_index, uint64_t bit_length, uint64_t* p_payload) {
    int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
    int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
    uintptr_t* tbl_base;
    uint32_t my_hndl;
    
        
    // Set the global bit index
    set_bit_index(bit_index);
    set_field_size(bit_length);

    // Avoid race conditions (setup can only be called once)
    //if (__atomic_test_and_set(&print_tmd_field_called, __ATOMIC_SEQ_CST))
    //    printTMDField();
    //    return;
    if (p_payload != NULL) {
        g_payload = *p_payload;
        g_modify = 1;
    } else {
        g_modify = 0;
    }
    
    cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
    uintptr_t subscribe_func_addr = *(tbl_base + 3);
    uintptr_t enable_func_addr = *(tbl_base + 6);
    subscribe = (typeof(subscribe))subscribe_func_addr;
    enable = (typeof(enable))enable_func_addr;
    int res = 0;
    res = subscribe(&my_hndl, launchCallback, NULL);
    // subscribe to the launch callback
    if (res)
        abort(1, 0, "Error subscribing to launch callback. CUDA returned error code %d.", res);
    res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
    if (res)
        abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
}
