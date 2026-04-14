/*
 * rvv_roofline_single.c
 *
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include "gem5/m5ops.h"

/* ------------------------------------------------------------------ */
/* Parameters                                                          */
/* ------------------------------------------------------------------ */
#ifndef REPEAT_OPS
#define REPEAT_OPS  64      /* tune this to sweep intensity */
#endif

#define N  65536            /* large enough to exceed L2   */
#define ALPHA 1.3f
#define BETA  0.7f

/* ------------------------------------------------------------------ */
/* Kernel                                                              */
/*                                                                     */
/* Each strip:                                                         */
/*   - 1 unit-stride load  (fixed memory traffic)                    */
/*   - REPEAT_OPS × vfmacc (variable compute)                        */
/*   - 1 unit-stride store (fixed memory traffic)                    */
/*                                                                     */
/* Using two alternating scalars (ALPHA, BETA) prevents the compiler  */
/* from folding repeated fmacc into a single multiply.                */
/* ------------------------------------------------------------------ */
void roofline_kernel(const float *x, float *y, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        /* One load — fixed bandwidth cost */
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /* REPEAT_OPS × fmacc — variable compute cost */
        vfloat32m4_t vacc = vx;
        for (int r = 0; r < REPEAT_OPS; r++) {
            /* Alternate scalars to prevent compiler optimisation */
            float s = (r % 2 == 0) ? ALPHA : BETA;
            vacc = __riscv_vfmacc_vf_f32m4(vacc, s, vx, vl);
        }

        /* One store — fixed bandwidth cost */
        __riscv_vse32_v_f32m4(y + i, vacc, vl);
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {

    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));

    for (size_t i = 0; i < N; i++)
        x[i] = 1.0f + (float)i * 0.0001f;

    /* Theoretical intensity for this configuration */
    double flop  = 2.0 * N * REPEAT_OPS;
    double bytes = 2.0 * N * sizeof(float);
    double I     = flop / bytes;

    printf("RVV Roofline Single Kernel\n");
    printf("===========================\n");
    printf("N           = %d\n",   N);
    printf("REPEAT_OPS  = %d\n",   REPEAT_OPS);
    printf("FLOP        = %.0f\n", flop);
    printf("Bytes       = %.0f\n", bytes);
    printf("Intensity I = %.3f FLOP/Byte\n\n", I);

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif   
    roofline_kernel(x, y, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif  

    printf("y[0]=%.4f y[N-1]=%.4f\n", y[0], y[N-1]);
    printf("\nExtract from GEM5:\n");
    printf("  perf = %.0f / numCycles  [FLOP/cycle]\n", flop);
    printf("  Plot point: (%.3f, perf)\n", I);

    free(x);
    free(y);
    return 0;
}