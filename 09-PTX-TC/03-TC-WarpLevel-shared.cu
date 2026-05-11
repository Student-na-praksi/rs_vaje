#include <cuda_fp16.h>
#include <stdio.h>
#define N 64
#define BLOCKSIZE 16
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define WARP_SIZE 32


void init_matrices(half *a, half *b, float *c, int matsize) {
    for (int i = 0; i < matsize; ++i) {
        for (int j = 0; j < matsize; ++j) {
            a[i * matsize + j] = __float2half(1.0);
            b[i * matsize + j] = __float2half(1.0);
            c[i * matsize + j] = 1.0;
        }
    }
}

__global__ void mma_block(int mat_size, half *A, half *B, float *C, float *D) {
    // Tile using a 2D grid
    int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpY = (blockIdx.y * blockDim.y + threadIdx.y);
    //printf("warpM = %d, warpN = %d\n", warpX, warpY);
    int block_size = mat_size ;
    int aRow = warpX * BLOCKSIZE;
    int aCol = warpY * BLOCKSIZE;
    int bRow = warpX * BLOCKSIZE;
    int bCol = warpY * BLOCKSIZE;
    int cRow = warpX * BLOCKSIZE;
    int cCol = warpY * BLOCKSIZE;
    int dRow = warpX * BLOCKSIZE;
    int dCol = warpY * BLOCKSIZE;

    int threadCol = threadIdx.x % BLOCKSIZE;
    int threadRow = threadIdx.x / BLOCKSIZE;

    // load A, B, C    
    half* A_block = A + aRow * mat_size + aCol;
    half* B_block = B + bRow * mat_size + bCol;
    float* C_block = C + cRow * mat_size + cCol;
    float* D_block = D + dRow * mat_size + dCol;
    // shared memory for A, B, C and Ds 
    __shared__ half As[BLOCKSIZE * BLOCKSIZE];
    __shared__ half Bs[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Cs[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Ds[BLOCKSIZE * BLOCKSIZE];
    __shared__ float accumulator[BLOCKSIZE * BLOCKSIZE];

    // Need to convert from generic to shared memory space using cvta
    long unsigned addrAs, addrBs, addrCs, addrDs, addrAcc;
    addrAs = static_cast< long unsigned>(__cvta_generic_to_shared(&As[0]));
    addrBs = static_cast< long unsigned>(__cvta_generic_to_shared(&Bs[0]));
    addrCs = static_cast< long unsigned>(__cvta_generic_to_shared(&Cs[0]));
    addrDs = static_cast< long unsigned>(__cvta_generic_to_shared(&Ds[0]));
    addrAcc = static_cast< long unsigned>(__cvta_generic_to_shared(&accumulator[0]));


    for (int j = 0; j < BLOCKSIZE*BLOCKSIZE/WARP_SIZE; j++){
        As[threadRow * BLOCKSIZE + threadCol + j*2*BLOCKSIZE] = A_block[threadRow * mat_size + threadCol + j*2*mat_size];
        Bs[threadRow * BLOCKSIZE + threadCol + j*2*BLOCKSIZE] = B_block[threadRow * mat_size + threadCol + j*2*mat_size];
        Cs[threadRow * BLOCKSIZE + threadCol + j*2*BLOCKSIZE] = C_block[threadRow * mat_size + threadCol + j*2*mat_size];
    }
    
    __syncthreads(); 

    
    
    asm volatile(
    ".reg .b32 a<8>, b<8>;\n\t"
    ".reg .b32 c<8>;\n\t"
    ".reg .b32 d<8>;\n\t"
    ".reg .u64 rd<4>;\n\t"
    ".reg .u32 r<2>;\n\t"
    // load address of a,b,c,d 
    "mov.u64 rd0, %0;\n\t" // d
    "mov.u64 rd1, %1;\n\t" // a
    "mov.u64 rd2, %2;\n\t" // b
    "mov.u64 rd3, %3;\n\t" // c
    "mov.u32 r0, %4;\n\t" // stride - number of elements between each row

    // convert to shared memory
    // wmma load, note there are three different load instructions
    // row major
    "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16 {a0, a1, a2, a3, a4, a5, a6, a7}, [rd1];\n\t"
    // col major
    "wmma.load.b.sync.aligned.m16n16k16.shared.col.f16 {b0, b1, b2, b3, b4, b5, b6, b7}, [rd2];\n\t"
    // row major
    "wmma.load.c.sync.aligned.m16n16k16.shared.row.f32 {c0, c1, c2, c3, c4, c5, c6, c7}, [rd3];\n\t"
    // multiply
    "wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 {d0, d1, d2, d3, d4, d5, d6, d7}, {a0, a1, a2, a3, a4, a5, a6, a7}, {b0, b1, b2, b3, b4, b5, b6, b7}, {c0, c1, c2, c3, c4, c5, c6, c7};\n\t"
    // wmma store
    "wmma.store.d.sync.aligned.m16n16k16.shared.row.f32 [rd0], {d0, d1, d2, d3, d4, d5, d6, d7};\n\t"
    : "+l" (addrDs) : "l" (addrAs), "l" (addrAs), "l" (addrCs), "r" (mat_size)); // employ shared memory pointers to access the shared memory
    // 
    // store D  
    
    for (int j = 0; j < BLOCKSIZE*BLOCKSIZE/WARP_SIZE; j++){
        D_block[threadRow * mat_size +threadCol + j*2*mat_size] = Ds[threadRow * BLOCKSIZE + threadCol + j*2*BLOCKSIZE];
    }
        
    
    __syncthreads();
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

    // init matrices 
    init_matrices(mat_a, mat_b, mat_c, N);

    
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
   

   // First: using WMM
    dim3 gridDim;
    dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
    blockDim.x = 32;
    blockDim.y = 1;
    gridDim.x = CEIL_DIV(N, BLOCKSIZE * blockDim.x / 32); 
    gridDim.y = CEIL_DIV(N, BLOCKSIZE * blockDim.y);
    printf("Running grid %d %d\n", gridDim.x, gridDim.y);
    printf("Running block %d %d\n", blockDim.x, blockDim.y);
    mma_block<<<gridDim, blockDim>>>(N, d_mat_a, d_mat_b, d_mat_c, d_mat_d);

    printf("Running gemm...\n");
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
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         if (mat_d[i * N + j] != mat_d_check[i * N + j]) {
    //             Error++;
    //         }
    //     }
    // }
    float acc = 0.0;
    if (Error == 0) {
        printf("Resultant matrix:\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%f ", mat_d[i * N + j]);
            }
            printf("\n");
        }
    }
    else {
        printf("Test failed!\n");
    }
    printf("acc: %f\n", acc);

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