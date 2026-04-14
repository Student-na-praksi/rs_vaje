/*
 * rvv_quant_matmul.c
 *
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>
/* ------------------------------------------------------------------ */
/* Parameters                                                          */
/* ------------------------------------------------------------------ */
#define M      64      /* matrix rows    */
#define N      256     /* matrix columns */
#define SCALE  7       /* right-shift after accumulation              */

/* ------------------------------------------------------------------ */
/* Utility: saturating clip to int8 range                             */
/* ------------------------------------------------------------------ */
static inline int8_t clip_i8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

static inline int16_t clip_i16(int32_t v) {
    if (v >  (1<<15) - 1) return  (1<<15) - 1;
    if (v < -(1<<15)) return -(1<<15);
    return (int16_t)v;
}


/* ------------------------------------------------------------------ */
/* Scalar baseline                                                     */
/* ------------------------------------------------------------------ */
void matvec_scalar(const int8_t  *A,   /* M×N, row-major              */
                   const int8_t  *x,   /* N                           */
                   int8_t        *y,   /* M                           */
                   int M_, int N_)
{
    for (int i = 0; i < M_; i++) {
        int32_t acc = 0;
        for (int j = 0; j < N_; j++)
            acc += (int32_t)A[i * N_ + j] * (int32_t)x[j];
        y[i] = clip_i8(acc >> SCALE);
    }
}

/* ================================================================== */
/* Variant 1: input i8, accumulator i32                               */
/* ================================================================== */
void matvec_i8_i32(const int8_t *A,
                   const int8_t *x,
                   int8_t       *y,
                   int M_, int N_)
{
    for (int i = 0; i < M_; i++) {
        const int8_t *row = A + i * N_;

        // Accumulator: i32m8, zero-initialised                       
        size_t vlmax = __riscv_vsetvlmax_e32m8();
        vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vlmax);

        size_t vl;
        for (int j = 0; j < N_; j += (int)vl) {
            // Request vl elements at e8m2.
            vl = __riscv_vsetvl_e8m2(N_ - j);

            // Load i8 strip from row and x 
            vint8m2_t va = __riscv_vle8_v_i8m2(row + j, vl);
            vint8m2_t vx = __riscv_vle8_v_i8m2(x   + j, vl);

            
            // vwmul: i8m2 × i8m2 → i16m4
            vint16m4_t vprod = __riscv_vwmul_vv_i16m4(va, vx, vl);

            
            // vwmacc: i16m4 accumulates into i32m8
            vl = __riscv_vsetvl_e16m4(vl); /* switch vl to e16 domain */
            // __riscv_vmv_v_x_i16m4(1, vl) creates a vector of 1s to sum vprod into vacc.
            vacc = __riscv_vwmacc_vv_i32m8(vacc, vprod, __riscv_vmv_v_x_i16m4(1, vl), vl);
        }

        // horizontal reduction: i32m8 → scalar 
        vl = __riscv_vsetvlmax_e32m8();
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t vsum  = __riscv_vredsum_vs_i32m8_i32m1(vacc, vzero, vl);
        int32_t    sum   = __riscv_vmv_x_s_i32m1_i32(vsum);

        // scale and narrow: i32 → i8 
        y[i] = clip_i8(sum >> SCALE);
    }
}

/* ================================================================== */
/* Variant 2: input i16, accumulator i32                              */
/* ================================================================== */
void matvec_i16_i32(const int16_t *A,
                    const int16_t *x,
                    int8_t        *y,
                    int M_, int N_)
{
    for (int i = 0; i < M_; i++) {
        const int16_t *row = A + i * N_;

        size_t vlmax = __riscv_vsetvlmax_e32m4();
        vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax);

        size_t vl;
        for (int j = 0; j < N_; j += (int)vl) {
            // e16m2: VLMAX(e16m2) = VLEN/16 * 2
            
            vl = __riscv_vsetvl_e16m2(N_ - j);

            vint16m2_t va = __riscv_vle16_v_i16m2(row + j, vl);
            vint16m2_t vx = __riscv_vle16_v_i16m2(x   + j, vl);

            // vwmul: i16m2 × i16m2 → i32m4
            vint32m4_t vprod = __riscv_vwmul_vv_i32m4(va, vx, vl);

            // Accumulate widened products 
            vl = __riscv_vsetvl_e32m4(vl);
            vacc = __riscv_vadd_vv_i32m4(vacc, vprod, vl);
        }

        /* ---- horizontal reduction ---- */
        vl = __riscv_vsetvlmax_e32m4();
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vint32m1_t vsum  = __riscv_vredsum_vs_i32m4_i32m1(vacc, vzero, vl);
        int32_t    sum   = __riscv_vmv_x_s_i32m1_i32(vsum);

        y[i] = clip_i16(sum >> SCALE);
    }
}

/* ================================================================== */
/* Variant 3: input i16, accumulator i64                              */
/* ================================================================== */
void matvec_i16_i64(const int16_t *A,
                    const int16_t *x,
                    int8_t        *y,
                    int M_, int N_)
{
    for (int i = 0; i < M_; i++) {
        const int16_t *row = A + i * N_;

        size_t vlmax = __riscv_vsetvlmax_e64m4();
        vint64m4_t vacc = __riscv_vmv_v_x_i64m4(0, vlmax);

        size_t vl;
        for (int j = 0; j < N_; j += (int)vl) {
            // e16m1: VLMAX(e16m1) = VLEN/16
            vl = __riscv_vsetvl_e16m1(N_ - j);

            vint16m1_t va = __riscv_vle16_v_i16m1(row + j, vl);
            vint16m1_t vx = __riscv_vle16_v_i16m1(x   + j, vl);

            // First widening: i16m1 × i16m1 → i32m2 
            vint32m2_t vprod = __riscv_vwmul_vv_i32m2(va, vx, vl);

            
            // Second widening accumulate: i32m2 → i64m4
            vl = __riscv_vsetvl_e32m2(vl);
            vacc = __riscv_vwadd_wv_i64m4(vacc, vprod, vl);
        }

        /* ---- horizontal reduction ---- */
        vl = __riscv_vsetvlmax_e64m4();
        vint64m1_t vzero64 = __riscv_vmv_v_x_i64m1(0, 1);
        vint64m1_t vsum64  = __riscv_vredsum_vs_i64m4_i64m1(vacc, vzero64, vl);
        int64_t    sum64   = __riscv_vmv_x_s_i64m1_i64(vsum64);

        y[i] = clip_i16((int32_t)(sum64 >> SCALE));
    }
}

/* ------------------------------------------------------------------ */
/* Data initialisation                                                 */
/* ------------------------------------------------------------------ */
static void init_i8(int8_t *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = (int8_t)((i * 3 + 1) % 127);
}
static void init_i16(int16_t *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = (int16_t)((i * 3 + 1) % 127);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {

    /* ---- i8 buffers ---- */
    int8_t *A8  = malloc(M * N * sizeof(int8_t));
    int8_t *x8  = malloc(N     * sizeof(int8_t));
    int8_t *y_scalar = malloc(M * sizeof(int8_t));
    int8_t *y_v1     = malloc(M * sizeof(int8_t));

    /* ---- i16 buffers ---- */
    int16_t *A16 = malloc(M * N * sizeof(int16_t));
    int16_t *x16 = malloc(N     * sizeof(int16_t));
    int8_t  *y_v2 = malloc(M * sizeof(int8_t));
    int8_t  *y_v3 = malloc(M * sizeof(int8_t));

    init_i8(A8,  M * N);  init_i8(x8,   N);
    init_i16(A16, M * N); init_i16(x16, N);

    /* ---- scalar baseline ---- */
    printf("[SCALAR i8×i8→i32] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    matvec_scalar(A8, x8, y_scalar, M, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    /* ---- Variant 1: i8 input, i32 accumulator ---- */
    printf("[RVV v1: i8×i8→i32] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    matvec_i8_i32(A8, x8, y_v1, M, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    /* ---- Variant 2: i16 input, i32 accumulator ---- */
    printf("[RVV v2: i16×i16→i32] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    matvec_i16_i32(A16, x16, y_v2, M, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    /* ---- Variant 3: i16 input, i64 accumulator ---- */
    printf("[RVV v3: i16×i16→i64] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    matvec_i16_i64(A16, x16, y_v3, M, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    free(A8); free(x8); free(y_scalar); free(y_v1);
    free(A16); free(x16); free(y_v2); free(y_v3);
    return 0;
}