#include <stdio.h>
#include <stddef.h>
#include <riscv_vector.h>

int main(void) {
    /* Query VLMAX for each configuration without processing any data */
    size_t vl;

    printf("VLEN-dependent vector lengths\n");
    printf("==============================\n\n");

    /* SEW = 8 */
    vl = __riscv_vsetvlmax_e8m1();
    // asm volatile("vsetvli %0, %1, "vtype", m1 "tail_mask" " : "=r"(max_vlen) : "r"(req_vlen));
    printf("SEW =  8b, LMUL =  1  ->  VLMAX = %zu\n", vl);

    /* SEW = 16, LMUL = 1 */
    vl = __riscv_vsetvlmax_e16m1();
    printf("SEW = 16b, LMUL =  1  ->  VLMAX = %zu\n", vl);

    /* SEW = 32, LMUL = 1 */
    vl = __riscv_vsetvlmax_e32m1();
    printf("SEW = 32b, LMUL =  1  ->  VLMAX = %zu\n", vl);

    /* SEW = 64, LMUL = 1 */
    vl = __riscv_vsetvlmax_e64m1();
    printf("SEW = 64b, LMUL =  1  ->  VLMAX = %zu\n", vl);

    printf("\n");

    /* SEW = 32, LMUL = 2 */
    vl = __riscv_vsetvlmax_e32m2();
    printf("SEW = 32b, LMUL =  2  ->  VLMAX = %zu\n", vl);

    /* SEW = 32, LMUL = 4 */
    vl = __riscv_vsetvlmax_e32m4();
    printf("SEW = 32b, LMUL =  4  ->  VLMAX = %zu\n", vl);

    /* SEW = 32, LMUL = 8 */
    vl = __riscv_vsetvlmax_e32m8();
    printf("SEW = 32b, LMUL =  8  ->  VLMAX = %zu\n", vl);

    printf("\n");

    /* set custom vector length */

    int avl = 6; // Application Vector Length
    vl = __riscv_vsetvl_e32m1(avl);
    printf("Requested AVL = %d, SEW = 32b, LMUL = 1 -> Set VLMAX = %zu\n", avl, vl);

    int avl2 = 20; // Application Vector Length
    vl = __riscv_vsetvl_e32m1(avl2);
    printf("Requested AVL = %d, SEW = 32b, LMUL = 1 -> Set VLMAX = %zu\n", avl2, vl);


    return 0;
}