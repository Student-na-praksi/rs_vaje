#include <cuda_fp16.h>
#include <stdio.h>
#define N 16*16

__global__ void wmma_load_test(float* output, float *input, float scalar)
{       
    asm volatile(
        ".reg .f32 x<8>;\n\t"
        ".reg .u64 rd<2>;\n\t"
        "mov.u64 rd0, %0;\n\t"
        "mov.u64 rd1, %1;\n\t"
        "wmma.load.c.sync.aligned.m16n16k16.global.row.f32 {x0, x1, x2, x3, x4, x5, x6, x7}, [rd1];\n\t"
        "mul.f32 x0, x0, %2;\n\t"
        "mul.f32 x1, x1, %2;\n\t"
        "mul.f32 x2, x2, %2;\n\t"
        "mul.f32 x3, x3, %2;\n\t"
        "mul.f32 x4, x4, %2;\n\t"
        "mul.f32 x4, x4, %2;\n\t"
        "mul.f32 x5, x5, %2;\n\t"
        "mul.f32 x6, x6, %2;\n\t"
        "mul.f32 x7, x7, %2;\n\t"
        "wmma.store.d.sync.aligned.m16n16k16.global.row.f32 [rd0], {x0, x1, x2, x3, x4, x5, x6, x7};\n\t"
        : "+l" (output) :"l" (input), "f"(scalar));
}



int main() {
    float *h_x, *h_y; // Host vectors
    float *d_x, *d_y; // Device vectors

    // Allocate memory on the host
    h_x = (float*)malloc(N * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));

    // Allocate memory on the device
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = i+1;
        h_y[i] = 2 * (i+1);
    }

    // Copy host vectors to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = 1;
    float a = 10; // Scalar value
    wmma_load_test<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_x, a);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display result
    printf("Resultant vector:\n");
    for (int i = 0; i < N; ++i) {
        printf("%.2f ", (float)h_y[i]);
    }
    printf("\n");

    // Free memory
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}