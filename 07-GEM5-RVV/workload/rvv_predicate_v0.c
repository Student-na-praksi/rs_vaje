/*
 * rvv_v0_predicate.c
 *
 * Illustrates the role of v0 as the RVV predicate register.
 *
 * Three progressively complex examples:
 *   Ex 1: Simple masked add — show which lanes are written
 *   Ex 2: Mask from comparison — branch-free conditional update
 *   Ex 3: Mask composition — AND of two conditions
 *
 * Compile:
 *   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gcv -mabi=lp64d \
 *       -static -o v0_predicate rvv_v0_predicate.c
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <riscv_vector.h>

/* ------------------------------------------------------------------ */
/* Print a float vector — shows both active and inactive lanes        */
/* ------------------------------------------------------------------ */
static void print_vec(const char *label, const float *v,
                      const int *active, size_t n)
{
    printf("%-20s [ ", label);
    for (size_t i = 0; i < n; i++) {
        if (active == NULL || active[i])
            printf("%+.1f ", v[i]);
        else
            printf("  _   ");       /* inactive lane marker */
    }
    printf("]\n");
}

static void print_mask(const char *label, const int *mask, size_t n)
{
    printf("%-20s [ ", label);
    for (size_t i = 0; i < n; i++)
        printf("  %d   ", mask[i]);
    printf("]\n");
}

/* ================================================================== */
/* Example 1: Masked add with a hand-crafted mask                    */
/* ================================================================== */
static void example1_masked_add(void)
{
    printf("\n");
    printf("=======================================================\n");
    printf(" Example 1: Masked add with hand-crafted mask          \n");
    printf("=======================================================\n");

    /* Small N so every lane fits on one line */
    const size_t N = 8;

    float x_data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float y_data[8] = { 10, 10, 10, 10, 10, 10, 10, 10 };
    float result[8];

    /*
     * Build the mask manually as a uint8 bitmask.
     * Bit k controls lane k. Mask = 0b01001101 = 0x4D
     *   bit0=1 (element 0 active)
     *   bit1=0 (element 1 inactive)
     *   bit2=1 (element 2 active)
     *   bit3=1 (element 3 active)
     *   bit4=0 (element 4 inactive)
     *   bit5=0 (element 5 inactive)
     *   bit6=1 (element 6 active)
     *   bit7=0 (element 7 inactive)
     */
    uint8_t raw_mask[1] = { 0x4D };   /* 0100 1101 */

    /* For pretty printing */
    int active[8] = { 1, 0, 1, 1, 0, 0, 1, 0 };

    size_t vl = __riscv_vsetvl_e32m1(N);

    /* Load data */
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(x_data, vl);
    vfloat32m1_t vy = __riscv_vle32_v_f32m1(y_data, vl);

    /*
     * Load the raw bitmask into a mask register (vbool32_t for m1/e32).
     */
    vbool32_t mask = __riscv_vlm_v_b32(raw_mask, vl);

    /*
     * Masked add: tail-undisturbed, mask-undisturbed (_tumu).
     */
    vfloat32m1_t vresult = __riscv_vfadd_vv_f32m1_tumu(
                               mask, vy, vx, vy, vl);

    __riscv_vse32_v_f32m1(result, vresult, vl);

    /* Print lane-by-lane to make v0's effect visible */
    printf("\n");
    print_vec ("x          :", x_data, NULL,   N);
    print_vec ("y (before) :", y_data, NULL,   N);
    print_mask("v0 mask    :", active,          N);
    print_vec ("y (after)  :", result, NULL,  N);
}

/* ================================================================== */
/* Example 2: Mask produced by a comparison                          */
/* ================================================================== */
static void example2_comparison_mask(void)
{
    printf("\n");
    printf("=======================================================\n");
    printf(" Example 2: Mask from comparison (vmflt)               \n");
    printf("=======================================================\n");

    const size_t N = 8;
    float x_data[8] = { -3, 5, -1, 8, -7, 2, 0, -4 };
    float result[8];
    int   active[8];

    size_t vl = __riscv_vsetvl_e32m1(N);

    vfloat32m1_t vx = __riscv_vle32_v_f32m1(x_data, vl);

    /*
     * vmflt_vf: compare vx < 0.0 element-wise.
     */
    vbool32_t mneg = __riscv_vmflt_vf_f32m1_b32(vx, 0.0f, vl);

    /*
     * Masked add scalar: add 100.0 only to negative lanes.
     * _tumu: tail-undisturbed, mask-undisturbed.
     */
    vfloat32m1_t vresult = __riscv_vfadd_vf_f32m1_tumu(
                               mneg, vx, vx, 100.0f, vl);

    __riscv_vse32_v_f32m1(result, vresult, vl);

    /* Build active[] for printing from the mask */
    for (size_t i = 0; i < N; i++)
        active[i] = (x_data[i] < 0.0f) ? 1 : 0;

    printf("\n");
    print_vec ("x           :", x_data, NULL,   N);
    print_mask("v0 = x<0    :", active,          N);
    print_vec ("x + 100 (m) :", result, NULL,  N);
}

/* ================================================================== */
/* Example 3: Mask composition — AND of two conditions               */
/* ================================================================== */
static void example3_mask_composition(void)
{
    printf("\n");
    printf("=======================================================\n");
    printf(" Example 3: Mask composition (vmand)                   \n");
    printf("=======================================================\n");

    const size_t N = 8;
    float x_data[8] = { -5, 2, -1, -8, 3, -4, 0, -2 };
    float result[8];
    int m_neg[8], m_lt3[8], m_both[8];

    size_t vl = __riscv_vsetvl_e32m1(N);

    vfloat32m1_t vx = __riscv_vle32_v_f32m1(x_data, vl);

    /* Condition A: x < 0 */
    vbool32_t mneg = __riscv_vmflt_vf_f32m1_b32(vx, 0.0f, vl);

    /* Condition B: x < -3 */
    vbool32_t mlt3 = __riscv_vmflt_vf_f32m1_b32(vx, -3.0f, vl);

    /*
     * Compose: m_both = m_neg AND m_lt3.
     * vmand_mm operates on two mask registers and writes the result
     * back as a mask. The compiler keeps this in v0 or a temp mask reg.
     */
    vbool32_t mboth = __riscv_vmand_mm_b32(mneg, mlt3, vl);

    /*
     * vfmerge: for each lane,
     *   if mboth[i] = 1 → write scalar 0.0 into result
     *   if mboth[i] = 0 → keep x[i] unchanged
     *
     * vfmerge always uses the mask argument (no _tumu needed here
     * because vfmerge is inherently a select operation).
     */
    vfloat32m1_t vresult = __riscv_vfmerge_vfm_f32m1(vx, 0.0f, mboth, vl);

    __riscv_vse32_v_f32m1(result, vresult, vl);

    /* Build arrays for printing */
    for (size_t i = 0; i < N; i++) {
        m_neg[i]  = (x_data[i] <  0.0f) ? 1 : 0;
        m_lt3[i]  = (x_data[i] < -3.0f) ? 1 : 0;
        m_both[i] = m_neg[i] & m_lt3[i];
    }

    printf("\n");
    print_vec ("x              :", x_data, NULL,   N);
    print_mask("m_neg (x<0)    :", m_neg,           N);
    print_mask("m_lt3 (x<-3)   :", m_lt3,           N);
    print_mask("m_both (AND)   :", m_both,           N);
    print_vec ("result         :", result, NULL,   N);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("RVV v0 Predicate Register Illustration\n");
    printf("=======================================\n");
    printf("VLEN=256, SEW=32, LMUL=1 -> VLMAX=8\n");

    example1_masked_add();
    example2_comparison_mask();
    example3_mask_composition();

    return 0;
}