#include <stdio.h>

#define N 1000 // Size of vectors

__global__ void daxpy(double *x, double *y, double a) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        y[tid] = a * x[tid] + y[tid];
    }
}

__global__ void daxpy_ptx(double *x, double *y, double a) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // load x[tid] and y[tid] into registers
    if (tid < N) {
        asm volatile(
        ".reg .f64 fd<6>;\n\t"
        ".reg .u64 rd<6>;\n\t"
        ".reg .u32 r<6>;\n\t"
        // load parameters 
        "mov.u64 rd0, %0;\n\t" // Address of y 
        "mov.u64 rd1, %1;\n\t" // Address of x 
        "mov.f64 fd2, %2;\n\t" // Value of a
        /// Calculate tid 
        "mov.u32 r0, %ctaid.x;\n\t" // blockIdx.x
        "mov.u32 r1, %ntid.x;\n\t" // blockDim.x
        "mov.u32 r2, %tid.x;\n\t" // threadIdx.x;
        "mad.lo.u32 r2, r1, r0, r2;\n\t"
        // load x[tid] and y[tid] into registers 
        "mul.wide.u32 rd3, r2, 8;\n\t" // Double data 
        "add.u64 rd0, rd0, rd3;\n\t" // Address of y[tid]
        "add.u64 rd1, rd1, rd3;\n\t" // Address of x[tid]
        // Perform DAXPY
        "ld.global.f64 fd0, [rd0];\n\t"
        "ld.global.f64 fd1, [rd1];\n\t"
        "mad.rp.f64 fd4, fd2, fd1, fd0;\n\t"
        // Store the result in y[tid]
        "st.global.f64 [rd0], fd4;\n\t"   
        :"+l"(y) // %0  
        : "l"(x), "d"(a));  // x-> %1, a-> %2 
        // l - u64, d - f64  
           // +d because we are writting and reading 
    }
}


__global__ void daxpy_debug(double *x, double *y, double a) {
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    volatile double a_local;
    volatile long thread_id;
    double y_local;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N) {
        asm volatile(
        ".reg .f64 fd<6>;\n\t"
        ".reg .u64 rd<6>;\n\t"
        ".reg .u32 r<6>;\n\t"
        // This works well ≠≠
        "mov.u32 r0, %ctaid.x;\n\t"
        "mov.u32 r1, %ntid.x;\n\t"
        "mov.u32 r2, %tid.x;\n\t"
        "mad.lo.u32 r1, r1, r0, r2;\n\t"
        // load parameters 
        "mov.u64 rd0, %0;\n\t" // Address of y 
        // load x[tid] and y[tid] into registers 
        "mul.wide.u32 rd3, r1, 8;\n\t" // Double data 
        "add.u64 rd4, rd0, rd3;\n\t"
        "cvt.rp.f64.u32 fd0, r1;\n\t"
        "mov.f64 %1, fd0;\n\t"
        "st.global.f64 [rd4], fd0;\n\t"
        //"mov.u64 %1, rd0;\n\t"
        :"+l"(y),"=d"(y_local): "l"(x), "l"(&a));        
        printf("SMID = %lf\n", y_local);

    }
}



int main() {
    double *h_x, *h_y; // Host vectors
    double *d_x, *d_y; // Device vectors

    // Allocate memory on the host
    h_x = (double*)malloc(N * sizeof(double));
    h_y = (double*)malloc(N * sizeof(double));

    // Allocate memory on the device
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_x[i] = i+1;
        h_y[i] = 2 * (i+1);
    }

    // Copy host vectors to device
    cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    double a = 1.5; // Scalar value
    daxpy_ptx<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, a);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Display result
    printf("Resultant vector:\n");
    for (int i = 0; i < N; ++i) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    // Free memory
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
