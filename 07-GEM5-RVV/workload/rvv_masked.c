/*
 * rvv_masked.c
 *
 * Masked conditional execution examples in RVV 1.0.
 *
 * Kernels:
 *   1. Absolute value:  y[i] = |x[i]|
 *   2. Clamp:           y[i] = clip(x[i], lo, hi)
 *   3. ReLU:            y[i] = max(x[i], 0.0f)
 *
 * Compile:
 *   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gcv -mabi=lp64d \
 *       -static -o rvv_masked rvv_masked.c
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>
#include <gem5/m5ops.h>

/* ------------------------------------------------------------------ */
/* Parameters                                                          */
/* ------------------------------------------------------------------ */
#define N      1024
#define LO    -0.5f
#define HI     0.5f

/* ================================================================== */
/* Kernel 1: Absolute value                                           */
/* ================================================================== */

void abs_scalar(const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] < 0.0f ? -x[i] : x[i];
}

void abs_rvv(const float *x, float *y, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /*
         * vmflt: floating-point less-than compare, produces mask.
         * m0[element] = 1 if vx[element] < 0.0
         */
        vbool8_t m0 = __riscv_vmflt_vf_f32m4_b8(vx, 0.0f, vl);

        /*
         * vfneg: negate all elements unconditionally.
         * The mask will select which result to keep.
         */
        vfloat32m4_t vneg = __riscv_vfneg_v_f32m4(vx, vl);

        /*
         * vmerge: merge two sources using mask m0.
         *   dst[i] = m0[i] ? vneg[i] : vx[i]
         * elements where x < 0 get the negated value;
         * all other elements keep the original.
         */
        vfloat32m4_t vy = __riscv_vmerge_vvm_f32m4(vx, vneg, m0, vl);

        __riscv_vse32_v_f32m4(y + i, vy, vl);
    }
}

/* ================================================================== */
/* Kernel 2: Clamp (clip)                                             */
/*                                                                     */
/* Scalar:                                                            */
/*   if      (x[i] < lo) y[i] = lo                                  */
/*   else if (x[i] > hi) y[i] = hi                                  */
/*   else                y[i] = x[i]                                 */
/* ================================================================== */

void clamp_scalar(const float *x, float *y, float lo, float hi, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if      (x[i] < lo) y[i] = lo;
        else if (x[i] > hi) y[i] = hi;
        else                y[i] = x[i];
    }
}

void clamp_rvv(const float *x, float *y, float lo, float hi, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /*
         * vfmax: element-wise max(vx, lo).
         * Any element below lo is replaced by lo.
         */
        vfloat32m4_t vlo = __riscv_vfmax_vf_f32m4(vx, lo, vl);

        /*
         * vfmin: element-wise min(vlo, hi).
         * Any element above hi is replaced by hi.
         */
        vfloat32m4_t vy  = __riscv_vfmin_vf_f32m4(vlo, hi, vl);

        __riscv_vse32_v_f32m4(y + i, vy, vl);
    }
}

/*
 * Alternative clamp using explicit masks — shows the mask mechanism
 */
void clamp_rvv_masked(const float *x, float *y,
                      float lo, float hi, size_t n)
{
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /* Masks for out-of-range elements */
        vbool8_t mlo = __riscv_vmflt_vf_f32m4_b8(vx, lo, vl);
        vbool8_t mhi = __riscv_vmfgt_vf_f32m4_b8(vx, hi, vl);

        /*
         * First merge: replace elements below lo with the scalar lo.
         * vfmerge selects the scalar when mask=1, vector when mask=0.
         */
        vfloat32m4_t vy = __riscv_vfmerge_vfm_f32m4(vx, lo, mlo, vl);

        /*
         * Second merge: replace elements above hi with the scalar hi.
         * Applied to vy so that the lo-clamped values are preserved.
         */
        vy = __riscv_vfmerge_vfm_f32m4(vy, hi, mhi, vl);

        __riscv_vse32_v_f32m4(y + i, vy, vl);
    }
}

/* ================================================================== */
/* Kernel 3: ReLU — max(x, 0)                                        */
/* ================================================================== */

void relu_scalar(const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

/* Strategy A: vfmax */
void relu_rvv_fmax(const float *x, float *y, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /* max(x, 0.0): negative elements become 0, positives unchanged */
        vfloat32m4_t vy = __riscv_vfmax_vf_f32m4(vx, 0.0f, vl);

        __riscv_vse32_v_f32m4(y + i, vy, vl);
    }
}

/* Strategy B: explicit mask + merge */
void relu_rvv_masked(const float *x, float *y, size_t n) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /* vzero: broadcast 0.0 into all elements */
        vfloat32m4_t vzero = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        /*
         * vmfgt: mask = 1 where vx > 0.0
         * Negative and zero elements get mask = 0.
         */
        vbool8_t mpos = __riscv_vmfgt_vf_f32m4_b8(vx, 0.0f, vl);

        /*
         * vmerge: select vx where mpos=1, vzero where mpos=0.
         * Equivalent to: y[i] = mpos[i] ? vx[i] : 0.0f
         */
        vfloat32m4_t vy = __riscv_vmerge_vvm_f32m4(vzero, vx, mpos, vl);

        __riscv_vse32_v_f32m4(y + i, vy, vl);
    }
}

/* ------------------------------------------------------------------ */
/* Data initialisation                                                 */
/* ------------------------------------------------------------------ */
static void init_float_signed(float *a, size_t n) {
    /* Fill with values in range [-1.0, 1.0] */
    for (size_t i = 0; i < n; i++)
        a[i] = -1.0f + (float)i * (2.0f / (float)(n - 1));
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {

    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));

    init_float_signed(x, N);

    /* ---- Kernel 1: Absolute value ---- */
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    abs_scalar(x, y, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    abs_rvv(x, y, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    /* ---- Kernel 2: Clamp ---- */
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    clamp_scalar(x, y, LO, HI, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif    
    clamp_rvv(x, y, LO, HI, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif    
    clamp_rvv_masked(x, y, LO, HI, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    /* ---- Kernel 3: ReLU ---- */
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif   
    relu_scalar(x, y, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    relu_rvv_fmax(x, y, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    relu_rvv_masked(x, y, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    free(x);
    free(y);
    return 0;
}