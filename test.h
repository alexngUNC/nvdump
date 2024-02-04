#include <cuda.h>

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <dlfcn.h>

#include "libsmctrl.h"
// In functions that do not return an error code, we favor terminating with an
// error rather than merely printing a warning and continuing.
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};
#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3
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
    void* tmd = **((uintptr_t***)in_params + 8);
    if (!tmd)
        abort(1, 0, "TMD allocation appears NULL; likely forward-compatibilty issue.\n");

    //fprintf(stderr, "cta: %lx\n", *(uint64_t*)(tmd + 74));
    // TODO: Check for supported QMD version (>XXX, <4.00)
    // TODO: Support QMD version 4 (Hopper), where offset starts at +304 (rather than +84) and is 16 bytes (rather than 8 bytes) wide. It also requires an enable bit at +31bits.
    uint32_t *lower_ptr = (uint32_t *) (tmd + 84);
    uint32_t *upper_ptr = (uint32_t *) (tmd + 88);

    //  if (g_next_sm_mask) {
    //      *lower_ptr = (uint32_t)g_next_sm_mask;
    //      *upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
    //      g_next_sm_mask = 0;
    //  } else if (!*lower_ptr && !*upper_ptr){
    //      // Only apply the global mask if a per-stream mask hasn't been set
    //      *lower_ptr = (uint32_t)g_sm_mask;
    //      *upper_ptr = (uint32_t)(g_sm_mask >> 32);
    //  }

    // ---------- Added by Alex ----------
    // Initialize memory to store whatever is at location target
    uint64_t a = 0;
    uint64_t* my_ptr = &a;

    // Specify address of target
    int target = 544;

    // Specify number of bits that target is
    int length = 18;

    // Create payload that is only the length of target
    uint64_t payload = 49152;

    // Find the closest multiple of 8 <= target
    int floor = (target / 8) * 8;

    // Find offset in size of 8-bits
    int offset = (int) floor / 8;

    // Grab 64 bits starting at floor
    uint64_t* target_addr = ((uint64_t*)((uint32_t*)(**((char***)in_params + 8) + offset)));
    *my_ptr = *((uint64_t*)((uint32_t*)(**((char***)in_params + 8) + offset)));

    // Shift right until you get the desired starting address
    int right = target - floor;
    *my_ptr = (*my_ptr) >> right;

    // Only take the desired number of bits
    uint64_t desired_bits = 0xFFFFFFFFFFFFFFFF;
    desired_bits = desired_bits >> (64 - length);
    *my_ptr = (*my_ptr) & desired_bits;

    // Print value before modifying
    fprintf(stdout, "Previous value: %lu\n", *my_ptr);
    fprintf(stdout, "target_addr: %p\t*target_addr: %lx\n", target_addr, *target_addr);

    // Payload of 64-bits; need to preserve left and right segments
    // that are not to be modified
    uint64_t full_payload = 0;

    // Find number of bits to keep on left of target
    int left = 64 - (length + right);
    printf("left: %u\tright: %u\n", left, right);

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
    // *target_addr = full_payload;

    // Print what is stored at bit number target
    fprintf(stdout, "lower_ptr: %p\t *lower_ptr: %u\n", lower_ptr, *lower_ptr);
    fprintf(stdout, "upper_ptr: %p\t *upper_ptr: %u\n", upper_ptr, *upper_ptr);

    // Currently just prints 64 bits at target address
    *my_ptr = *((uint64_t*)((uint32_t*)(**((char***)in_params + 8) + offset)));

    // Shift right until you get the desired starting address
    *my_ptr = (*my_ptr) >> right;

    // Only take the desired number of bits
    desired_bits = 0xFFFFFFFFFFFFFFFF  >> (64 - length);
    *my_ptr = (*my_ptr) & desired_bits;

    // Print segments
    fprintf(stdout, "Right part: %lx\n", right_part);
    fprintf(stdout, "Full payload: %lx\n", full_payload);

    // Print only the target value
    fprintf(stdout, "target_addr: %p\t*target_addr: %lx\n", target_addr, *target_addr);
    fprintf(stdout, "Modified value: %lu\n", *my_ptr);
}
static void setup_sm_control_11() {
    int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
    int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
    uintptr_t* tbl_base;
    uint32_t my_hndl;
    // Avoid race conditions (setup can only be called once)
    //if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
    //    return;

    cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
    uintptr_t subscribe_func_addr = *(tbl_base + 3);
    uintptr_t enable_func_addr = *(tbl_base + 6);
    subscribe = (typeof(subscribe))subscribe_func_addr;
    enable = (typeof(enable))enable_func_addr;
    int res = 0;
    res = subscribe(&my_hndl, launchCallback, NULL);
    if (res)
        abort(1, 0, "Error subscribing to launch callback. CUDA returned error code %d.", res);
    res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
    if (res)
        abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
}
