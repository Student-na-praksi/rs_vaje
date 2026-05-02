
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"


#define TILE_SIZE 16

// Kernel for matrix multiplication
__global__ void matrixMul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel for matrix multiplication using shared memory
__global__ void matrixMul_shared(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform computation on the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


// Function to initialize a matrix with random values
void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = 1; // Random values between 0 and 9
    }
}

// Function to print a matrix
void printMatrix(const float* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%6.2f ", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    const int N = 64; // Size of the matrix (N x N)
    const int matrixSize = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(matrixSize);
    float* h_B = (float*)malloc(matrixSize);
    float* h_C = (float*)malloc(matrixSize);

    // Initialize matrices A and B
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    printf("Matrix A:\n");
    //printMatrix(h_A, N);

    printf("\nMatrix B:\n");
    //printMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc((void**)&d_A, matrixSize);
    hipMalloc((void**)&d_B, matrixSize);
    hipMalloc((void**)&d_C, matrixSize);

    // Copy matrices A and B to device memory
    hipMemcpy(d_A, h_A, matrixSize, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, matrixSize, hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel for shared memory matrix multiplication
    hipLaunchKernelGGL(matrixMul_shared, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N);

    // Copy the result matrix C back to host memory
    hipMemcpy(h_C, d_C, matrixSize, hipMemcpyDeviceToHost);

    printf("\nMatrix C (Result):\n");
    printMatrix(h_C, N);

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}