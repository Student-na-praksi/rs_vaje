#include <cuda_fp16.h>
#include <stdio.h>
#define N 16

__global__ void wmma_test(half* a, half *b, float *c, float *d)
{       
    asm volatile(
        ".reg .b32 a<8>, b<8>;\n\t"
        ".reg .b32 c<8>;\n\t"
        ".reg .b32 d<8>;\n\t"
        ".reg .u64 rd<4>;\n\t"
        // load address of a,b,c,d 
        "mov.u64 rd0, %0;\n\t" // d
        "mov.u64 rd1, %1;\n\t" // a
        "mov.u64 rd2, %2;\n\t" // b
        "mov.u64 rd3, %3;\n\t" // c
        // wmma load, note there are three different load instructions
        // row major
        "wmma.load.a.sync.aligned.m16n16k16.global.row.f16 {a0, a1, a2, a3, a4, a5, a6, a7}, [rd1];\n\t"
        // col major
        "wmma.load.b.sync.aligned.m16n16k16.global.col.f16 {b0, b1, b2, b3, b4, b5, b6, b7}, [rd2];\n\t"
        // row major
        "wmma.load.c.sync.aligned.m16n16k16.global.row.f32 {c0, c1, c2, c3, c4, c5, c6, c7}, [rd3];\n\t"
        // multiply
        "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 {d0, d1, d2, d3, d4, d5, d6, d7}, {a0, a1, a2, a3, a4, a5, a6, a7}, {b0, b1, b2, b3, b4, b5, b6, b7}, {c0, c1, c2, c3, c4, c5, c6, c7};\n\t"
        // wmma store
        "wmma.store.d.sync.aligned.m16n16k16.global.row.f32 [rd0], {d0, d1, d2, d3, d4, d5, d6, d7};\n\t"
        : "+l" (d) : "l" (a), "l" (b), "l" (c) );
            // +d because we are writting and reading 
}



int main() {
    half *mat_a, *mat_b;
    float *mat_c;
    float *mat_d, *mat_d_check;

    mat_a = (half*)malloc(N * N * sizeof(half));
    mat_b = (half*)malloc(N * N * sizeof(half));
    mat_c = (float*)malloc(N * N * sizeof(float));
    mat_d = (float*)malloc(N * N * sizeof(float));
    mat_d_check = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mat_a[i * N + j] = __float2half(1);
            mat_b[i * N + j] = __float2half(1);
            mat_c[i * N + j] = 1.0;
        }
    }

    // instantiate buffers on the device
    half *d_mat_a, *d_mat_b;
    float *d_mat_c, *d_mat_d;
    cudaMalloc(&d_mat_a, N * N * sizeof(half));
    cudaMalloc(&d_mat_b, N * N * sizeof(half));
    cudaMalloc(&d_mat_c, N * N * sizeof(float));
    cudaMalloc(&d_mat_d, N * N * sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_mat_a, mat_a, N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_b, mat_b, N * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_c, mat_c, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = 1;
    wmma_test<<<blocksPerGrid, threadsPerBlock>>>(d_mat_a, d_mat_b, d_mat_c, d_mat_d);

    // copy data from device to host
    cudaMemcpy(mat_d, d_mat_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);  

    // scalar version of matrix multiplication
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += __half2float(mat_a[i * N + k]) * __half2float(mat_b[k * N + j]);
            }
            mat_d_check[i * N + j] = sum + mat_c[i * N + j];
        }
    }

    // display result

    int Error = 0;
    // check result
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (mat_d[i * N + j] != mat_d_check[i * N + j]) {
                Error++;
            }
        }
    }

    if (Error == 0) {
        printf("Resultant matrix:\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%.2f ", mat_d[i * N + j]);
            }
            printf("\n");
        }
    }
    else {
        printf("Test failed!\n");
    }
    // free memory
    free(mat_a);
    free(mat_b);
    free(mat_c);
    free(mat_d);
    free(mat_d_check);
    cudaFree(d_mat_a);
    cudaFree(d_mat_b);
    cudaFree(d_mat_c);
    cudaFree(d_mat_d);

    return 0;
}