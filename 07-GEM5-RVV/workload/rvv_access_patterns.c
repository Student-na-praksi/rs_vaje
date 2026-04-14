/*
 * rvv_access_patterns.c
 *
 * Sparse vector update: y[idx[i]] += alpha * x[i]
 * Variants:
 *   scalar          : plain C scalar loop
 *   unit_stride     : RVV unit-stride (regular layout, no indirection)
 *   strided         : RVV strided load (fixed stride, no indirection)
 *   gather_scatter  : RVV indexed load/store (full indirection via idx[])
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
#define N       1024        /* number of source elements              */
#define Y_SIZE  4096        /* size of destination vector y           */
#define ALPHA   2.0f        /* scale factor                           */
#define STRIDE  4           /* element stride for strided variant     */

/* ------------------------------------------------------------------ */
/* Variant 0: Scalar baseline                                          */
/* ------------------------------------------------------------------ */
void update_scalar(const float    *x,
                   const uint32_t *idx,
                   float          *y,
                   float           alpha,
                   size_t          n)
{
    for (size_t i = 0; i < n; i++)
        y[idx[i]] += alpha * x[i];
}

/* ================================================================== */
/* Variant 1: Unit-stride                                             */
/* ================================================================== */
void update_unit_stride(const float *x,
                        float       *y,
                        float        alpha,
                        size_t       n)
{
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        vfloat32m4_t vx  = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vy  = __riscv_vle32_v_f32m4(y + i, vl);

        /* y[i] += alpha * x[i] */
        vy = __riscv_vfmacc_vf_f32m4(vy, alpha, vx, vl);

        __riscv_vse32_v_f32m4(y + i, vy, vl);
    }
}

/* ================================================================== */
/* Variant 2: Strided                                                 */
/* ================================================================== */
void update_strided(const float *x,
                    float       *y,
                    float        alpha,
                    size_t       n)   
{
    /* Byte stride between consecutive accessed elements of y */
    ptrdiff_t byte_stride = STRIDE * sizeof(float);

    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);

        /* Unit-stride load of x */
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);

        /* Strided load of y: y[i*STRIDE], y[(i+1)*STRIDE], ... */
        vfloat32m4_t vy = __riscv_vlse32_v_f32m4(
                              y + i * STRIDE, byte_stride, vl);

        vy = __riscv_vfmacc_vf_f32m4(vy, alpha, vx, vl);

        /* Strided store back to y */
        __riscv_vsse32_v_f32m4(y + i * STRIDE, byte_stride, vy, vl);
    }
}

/* ================================================================== */
/* Variant 3: Indexed gather / scatter                                */
/* ================================================================== */
void update_gather_scatter(const float    *x,
                           const uint32_t *idx,
                           float          *y,
                           float           alpha,
                           size_t          n)
{
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);

        /* Unit-stride load of x and idx */
        vfloat32m2_t  vx   = __riscv_vle32_v_f32m2(x + i, vl);
        vuint32m2_t   vidx = __riscv_vle32_v_u32m2(idx + i, vl);

        /*
         * Convert element indices to byte offsets:
         *   byte_offset = idx[i] * sizeof(float) = idx[i] << 2
         */
        vuint32m2_t vbytes = __riscv_vsll_vx_u32m2(vidx, 2, vl);

        /*
         * Gather: load y[idx[i]] for each active lane.
         * vloxei32: indexed load with 32-bit byte offsets.
         */
        vfloat32m2_t vy = __riscv_vloxei32_v_f32m2(y, vbytes, vl);

        /* Accumulate: y[idx[i]] += alpha * x[i] */
        vy = __riscv_vfmacc_vf_f32m2(vy, alpha, vx, vl);

        /*
         * Scatter: store back to y[idx[i]] for each active lane.
         * vsoxei32: indexed store with 32-bit byte offsets.
         */
        __riscv_vsoxei32_v_f32m2(y, vbytes, vy, vl);
    }
}

/* ------------------------------------------------------------------ */
/* Index generator: unique indices within each strip of width vl      */
/* ------------------------------------------------------------------ */
static void gen_indices(uint32_t *idx, size_t n, size_t y_size) {
    uint32_t gap = 64; // (uint32_t)(y_size / n) + 1;
    for (size_t i = 0; i < n; i++)
        idx[i] = (uint32_t)((i * gap) % y_size);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {

    float    *x      = malloc(N       * sizeof(float));
    float    *y_ref  = malloc(Y_SIZE  * sizeof(float));
    float    *y_work = malloc(Y_SIZE  * sizeof(float));
    uint32_t *idx    = malloc(N       * sizeof(uint32_t));

    /* Initialise x */
    for (size_t i = 0; i < N; i++)
        x[i] = 1.0f + (float)i * 0.001f;

    /* Initialise y */
    for (size_t i = 0; i < Y_SIZE; i++)
        y_ref[i] = 0.5f;

    /* Generate non-aliasing indices */
    gen_indices(idx, N, Y_SIZE);

    // print idx 
    for (size_t i = 0; i < N; i++)
        printf("idx[%zu] = %u\n", i, idx[i]);
    /* ---- Scalar baseline ---- */
    memcpy(y_work, y_ref, Y_SIZE * sizeof(float));
    printf("[SCALAR] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    update_scalar(x, idx, y_work, ALPHA, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    /* ---- Variant 1: unit-stride ---- */
    memcpy(y_work, y_ref, Y_SIZE * sizeof(float));
    printf("[UNIT-STRIDE] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    update_unit_stride(x, y_work, ALPHA, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    /* ---- Variant 2: strided ---- */
    memcpy(y_work, y_ref, Y_SIZE * sizeof(float));
    printf("[STRIDED] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    update_strided(x, y_work, ALPHA, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    /* ---- Variant 3: gather/scatter ---- */
    memcpy(y_work, y_ref, Y_SIZE * sizeof(float));
    printf("[GATHER/SCATTER] start\n");
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    update_gather_scatter(x, idx, y_work, ALPHA, N);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    free(x); free(y_ref); free(y_work); free(idx);
    return 0;
}