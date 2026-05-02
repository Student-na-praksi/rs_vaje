#include <stdio.h>
#include "hip/hip_runtime.h"

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}

/*
 * Add corresponding elements of arrays A and B, and write to array C.
 */
template <typename T>
__global__ void
vector_add(T *C_d, const T *A_d, const T *B_d, size_t N)
{
    size_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < N) {
        C_d[i] = A_d[i] + B_d[i];
    }
}


int main(int argc, char *argv[])
{
    float *A_h, *B_h, *C_h;
    size_t N = 1 << 16;
    size_t Nbytes = N * sizeof(float);

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0 /*deviceID*/));
    printf("info: running on device %s\n", props.name);
    #ifdef __HIP_PLATFORM_HCC__
      printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
    #endif
    printf("info: allocate host and device mem (%6.2f MB)\n", 3 * Nbytes / 1024.0 / 1024.0);
    CHECK(hipHostMalloc(&A_h, Nbytes));
    CHECK(hipHostMalloc(&B_h, Nbytes));
    CHECK(hipHostMalloc(&C_h, Nbytes));

    // Fill A_h and B_h with values
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i; 
        B_h[i] = 2.718f + i;
    }

    printf("Number of elements: %zu\n", N);
    const unsigned threadsPerBlock = 256;
    const unsigned blocks = N / threadsPerBlock;
;

    printf("info: launch 'vector_add' kernel\n");
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_h, A_h, B_h, N);
    hipDeviceSynchronize();

    printf("info: check result\n");
    for (size_t i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] + B_h[i]) {
            CHECK(hipErrorUnknown);
        }
    }
    printf("PASSED!\n");

    // Free memory
    CHECK(hipHostFree(A_h));
    CHECK(hipHostFree(B_h));
    CHECK(hipHostFree(C_h));

    return 0;
}
