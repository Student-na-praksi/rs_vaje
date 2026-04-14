#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>

void axpy_scalar(float alpha, const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; i++)
        y[i] = alpha * x[i] + y[i];
}
 
/*
 * Strip-mining pattern:
 *   - vsetvli sets vl (actual vector length) and configures vtype
 *   - loop advances by vl each iteration
 *   - handles tail automatically (vl < VLEN at last iteration)
 */
void axpy_rvv(float alpha, const float *x, float *y, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) { // ceil(n/vl) iterations
        /* Set vl = min(n-i, VLMAX) for e32 (32-bit float), m1 grouping */
        vl = __riscv_vsetvl_e32m1(n - i);
 
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);  /* unit-stride load */
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y + i, vl);
 
        /* fused multiply-add: vy = alpha * vx + vy */
        vy = __riscv_vfmacc_vf_f32m1(vy, alpha, vx, vl);
 
        __riscv_vse32_v_f32m1(y + i, vy, vl);                 /* unit-stride store */
    }
}


int main() {
    const size_t n = 521; // Example vector length
    float alpha = 2.0f;
    float x[n], y[n], y_ref[n];
 
    // Initialize x and y with example values
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = (float)(n - i);
        y_ref[i] = (float)(n - i);

    }
 
    axpy_scalar(alpha, x, y_ref, n); // Call scalar version

 
    axpy_rvv(alpha, x, y, n); // Call RVV version
 
    // check results
    for (size_t i = 0; i < n; i++) {
        if (y[i] != y_ref[i]) {
            printf("Mismatch at index %zu: got %f, expected %f\n", i, y[i], y_ref[i]);
            return 1;
        }
    }  
    printf("AXPY results match reference implementation.\n");
 
 
    return 0;
}