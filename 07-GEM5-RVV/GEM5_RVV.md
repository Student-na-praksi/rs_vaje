# Benchmarking RISC-V Vector Extension (RVV) in gem5   

## Vectorised Fixed-Point Arithmetic — Widening, Narrowing, and SEW


Neural network inference on embedded processors frequently uses quantised
weights and activations to reduce memory bandwidth and arithmetic cost. A
typical pattern is:

1. Load narrow operands (e.g. `int8`)
2. Widen and multiply into a wider accumulator (e.g. `int32`)
3. Apply a scale factor and narrow the result back to `int8`

RVV supports this natively through widening multiply (`vwmul`), which doubles
*SEW* in the destination, and narrowing shift-right-round (`vnsra`), which
halves *SEW* in the destination.

### Kernel description

All variants compute the same operation:

```
y[i] = clip( (A[i,:] · x) >> scale, -128, 127 )
```

where `A` is an `M × N` matrix, `x` is a vector of length `N`, and the result
`y` is clipped to `int8` range. The dot product accumulates in a wider type to
avoid overflow.

### Employed RVV instructions

| Intrinsic | Assembly mnemonic | SEW in | SEW out | LMUL in | LMUL out | Operation |
|---|---|---|---|---|---|---|
| `__riscv_vmv_v_x_i16m4(scalar, vl)` | `vmv.v.x` | — | 16b | — | 4 | Broadcast scalar integer to all `vl` lanes |
| `__riscv_vwmul_vv_i16m4(vs2, vs1, vl)` | `vwmul.vv` | 8b | 16b | 2 | 4 | Widening element-wise multiply: `dst[i] = vs2[i] × vs1[i]` |
| `__riscv_vwmacc_vv_i32m8(vd, vs1, vs2, vl)` | `vwmacc.vv` | 16b | 32b | 4 | 8 | Widening multiply-accumulate: `vd[i] += vs1[i] × vs2[i]` |


## Performance of Memory Access Patterns

Memory access pattern is one of the most significant factors determining
vector processor performance. RVV supports three distinct access modes —
unit-stride, strided, and indexed (gather/scatter) — each with different
hardware requirements and cache behaviour. This exercise implements the same
indirect vector update kernel in four variants to isolate the cost of
increasingly irregular memory access patterns.


### Kernel description

All variants compute the sparse vector update:

```
y[idx[i]] += alpha * x[i],   for i = 0..N-1
```

where `idx` contains indices into `y`, which may be non-contiguous and
unordered. This pattern appears in sparse matrix  operations, graph algorithms,
and other irregular computations.

### Employed RVV instructions

| Intrinsic | Mnemonic | Pattern | Operands | Description |
|---|---|---|---|---|
| `__riscv_vle<sew>_v_<type>(ptr, vl)` | `vle<sew>.v` | Unit-stride load | `ptr`: base address | Load `vl` consecutive elements of width `sew` bits |
| `__riscv_vse<sew>_v_<type>(ptr, vs, vl)` | `vse<sew>.v` | Unit-stride store | `ptr`: base address, `vs`: source vector | Store `vl` consecutive elements of width `sew` bits |
| `__riscv_vlse<sew>_v_<type>(ptr, stride, vl)` | `vlse<sew>.v` | Strided load | `ptr`: base address, `stride`: byte stride (ptrdiff_t) | Load `vl` elements separated by `stride` bytes |
| `__riscv_vsse<sew>_v_<type>(ptr, stride, vs, vl)` | `vsse<sew>.v` | Strided store | `ptr`: base address, `stride`: byte stride, `vs`: source vector | Store `vl` elements separated by `stride` bytes |
| `__riscv_vloxei<sew>_v_<type>(base, offsets, vl)` | `vloxei<sew>.v` | Indexed load (ordered) | `base`: base address, `offsets`: vector of byte offsets | Gather: load `base[offsets[i]]` per lane; indices must be unique |
| `__riscv_vluxei<sew>_v_<type>(base, offsets, vl)` | `vluxei<sew>.v` | Indexed load (unordered) | `base`: base address, `offsets`: vector of byte offsets | Gather: like `vloxei` but indices need not be unique; may reorder accesses |
| `__riscv_vsoxei<sew>_v_<type>(base, offsets, vs, vl)` | `vsoxei<sew>.v` | Indexed store (ordered) | `base`: base address, `offsets`: vector of byte offsets, `vs`: source vector | Scatter: store `vs[i]` to `base[offsets[i]]` per lane; indices must be unique |
| `__riscv_vsuxei<sew>_v_<type>(base, offsets, vs, vl)` | `vsuxei<sew>.v` | Indexed store (unordered) | `base`: base address, `offsets`: vector of byte offsets, `vs`: source vector | Scatter: like `vsoxei` but indices need not be unique; may reorder accesses |
| `__riscv_vlm_v_b<n>(ptr, vl)` | `vlm.v` | Mask load | `ptr`: base address | Load `vl` mask bits from memory into a mask register |
| `__riscv_vsm_v_b<n>(ptr, vs, vl)` | `vsm.v` | Mask store | `ptr`: base address, `vs`: source mask register | Store `vl` mask bits from a mask register to memory |

## The `v0` Register as a Predicate

In RVV 1.0, `v0` has a special architectural role: it is the **only**
register that can serve as a mask (predicate) for vector instructions.
While all other vector registers (`v1`–`v31`) hold data, `v0` holds a
bitmask where each bit controls whether the corresponding lane is active.

This is expressed in assembly as the `.v0.t` suffix on any maskable
instruction:

```asm
vfadd.vv  v2, v3, v4, v0.t   # add only where v0 bit = 1
```

In intrinsics, the mask is passed as an explicit argument of type
`vbool<n>_t`, and the compiler always maps it to `v0`.


### How `v0` bits map on elements

For SEW=32 and VLEN=256, VLMAX=8. The `v0` register holds one bit per
element. Bit 0 controls element 0, bit 1 controls element 1, and so on:

```
v0  (mask register):   [ 1  0  1  1  0  0  1  0 ]
                         |     |  |        |
                         ↓     ↓  ↓        ↓
vx  (source data) :   [ a  b  c  d  e  f  g  h ]
                         |     |  |        |
                         ↓     ↓  ↓        ↓
vy  (destination) :   [ *  _  *  *  _  _  *  _ ]

* = computed result written
_ = element inactive (undisturbed or agnostic depending on vta/vma)
```

## Masked Conditional Execution

Scalar code frequently uses branches to handle conditional operations —
different elements in an array may require different computation depending
on their value. On a vector processor, branches cannot be taken
per-element. RVV resolves this through **mask registers**: a dedicated
boolean vector where each bit corresponds to one active element. Instructions
can then operate selectively on elements where the mask bit is set, leaving
other elements either undisturbed or set to zero depending on the merge
policy.

In this part several workload implementations are provided to demonstrate the use of masked operations in RVV, showing how three kernels that would require per-element branches in scalar code can be expressed branch-free using RVV mask instructions.


### Kernels

1. **Absolute value** — `y[i] = |x[i]|`
2. **Vector clamp** — `y[i] = clip(x[i], lo, hi)`
3. **ReLU** — `y[i] = max(x[i], 0)` (common neural network activation)

Each is first implemented as a scalar reference, then as a masked RVV
intrinsic version.

## Roofline Model

The roofline model is a visual performance model that relates the achieved
floating-point throughput of a kernel to the memory bandwidth and compute
capacity of the processor it runs on. It was introduced by Williams, Waterman,
and Patterson in 2009 as a practical tool for identifying the primary
bottleneck of a kernel without requiring detailed microarchitectural
knowledge.

### Core idea

Every processor has two fundamental limits:

- **Peak compute throughput** (*π*) — the maximum number of floating-point
  operations it can complete per unit time, measured in FLOP/cycle. This is
  determined by the number and width of the floating-point execution units.

- **Peak memory bandwidth** (*β*) — the maximum rate at which data can be
  transferred between memory and the processor, measured in Bytes/cycle. This
  is determined by the memory bus width, the cache hierarchy, and the memory
  controller.

A kernel cannot exceed either limit. Its achievable performance is therefore
bounded by whichever limit is more restrictive given how much data it moves
relative to how much computation it performs.

---

### Arithmetic intensity

The key property of a kernel that determines which limit applies is its
**arithmetic intensity** (*I*), defined as the ratio of floating-point
operations to bytes transferred from memory:

```
I = FLOP / Bytes   [FLOP/Byte]
```

A kernel that performs many operations on a small amount of data has high
arithmetic intensity. A kernel that moves large amounts of data but performs
little computation on each element has low arithmetic intensity.

Some representative examples for RVV kernels at SEW=32:

| Kernel | FLOP | Bytes | I [FLOP/Byte] |
|---|---|---|---|
| Vector copy | 0 | 2 × N × 4 | 0 |
| AXPY | 2 × N | 3 × N × 4 | 0.17 |
| Dot product | 2 × N | 2 × N × 4 | 0.25 |
| SpMV CSR | 2 × NNZ | ≥ 3 × NNZ × 4 | ≤ 0.17 |
| Quant matvec (i8) | 2 × M × N | M × N × 1 | ≈ 2.0 |


### The roofline equation

The achievable performance *P* of a kernel is bounded by:

```
P = min( π,  β × I )   [FLOP/cycle]
```

This equation defines two straight lines on a log-log plot:

- The **memory roof** — a line of slope 1 passing through the origin:
  `P = β × I`. Performance grows linearly with intensity as long as the
  kernel is memory-bound.

- The **compute roof** — a horizontal line at `P = π`. Once the kernel
  provides enough data reuse to keep the execution units busy, performance
  is capped by compute capacity and no longer grows with intensity.

The intersection of the two lines defines the **ridge point**:

```
I_ridge = π / β   [FLOP/Byte]
```

Kernels with `I < I_ridge` are **memory-bound**: the execution units are
starved of data and cannot operate at full throughput. Increasing compute
width or adding more execution units will not improve performance. The only
way to improve is to increase arithmetic intensity (data reuse) or increase
memory bandwidth.

Kernels with `I > I_ridge` are **compute-bound**: the memory system delivers
data faster than the execution units can consume it. Increasing memory
bandwidth will not improve performance. The only way to improve is to
increase peak compute throughput or reduce the operation count.


### Roofline on a log-log plot

The roofline is conventionally plotted on a log-log axis with arithmetic
intensity on the x-axis and performance on the y-axis:

```
P
[FLOP/cycle]
    |                              π ──────────────────── compute roof
    |                           ./
    |                        ./
    |                     ./        ← memory slope (β)
    |                  ./
    |               ./
    |            ./
    +─────────────────────────────── I [FLOP/Byte]
                  ↑
              I_ridge
```

Each kernel appears as a single point on this plot. A point that lies on the
memory slope is memory-bound; a point that lies on the compute roof is
compute-bound. The vertical distance between a kernel's point and the roofline
is its **performance gap** — how much performance is left on the table due to
inefficiency beyond the primary bottleneck.



### Application to RVV

For an RVV processor, the roofline model takes on additional dimensions
because both *π* and *β* depend on the vector configuration:

**Effect of VLEN on π**

Doubling VLEN doubles the number of elements processed per instruction,
which doubles peak throughput assuming the execution unit scales accordingly:

```
π = (VLEN / SEW) × LMUL × FMA_units × 2 FLOP/cycle
```

**Effect of VLEN on β**

Doubling VLEN also doubles the width of unit-stride memory transactions,
which can increase effective memory bandwidth if the memory bus is wide
enough to sustain it.

**Effect of SEW on I**

Narrower element widths (e.g. i8 vs f32) increase arithmetic intensity
because the same number of FLOP requires fewer bytes transferred. This is
why quantised kernels (SEW=8) sit much further right on the roofline plot
than their f32 equivalents — they are more likely to be compute-bound.



