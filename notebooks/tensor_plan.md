# tensor.ipynb — Detailed Plan

---

## Section 0 — Recap: what a Tensor is

**Markdown cell.**

Abstract model (already stubbed):

```
T = E ∘ L
T[X]  =  value_map( E + L(X) )
```

- `E` — **engine**: a random-access iterator. It is just a typed pointer into some memory region, or a counting iterator for coordinate tensors.
- `L` — **layout**: maps logical coord `X` → physical offset `L(X)`.
- The engine defines *where* the data lives (which address space), the layout defines *how coordinates map to offsets within that space*.

Address spaces (`cute.AddressSpace`):

| Enum value | Space | Who owns it |
|---|---|---|
| `generic (0)` | compiler-chosen | flexible |
| `gmem (1)` | global DRAM | host-allocated or device-allocated |
| `smem (3)` | shared SRAM | per-CTA, lifetime = kernel |
| `rmem` | registers | per-thread, SSA-valued in IR |

There is no `AddressSpace.rmem` enum — rmem tensors are a special IR type (`TensorSSA`) created with `make_rmem_tensor`, not via `make_ptr`.

---

## Section 1 — gmem tensor

**Goal:** wrap a raw device pointer + layout into a `cute.Tensor`.

**Kernel sketch:**

```python
@cute.kernel
def gmem_tensor_demo(A_ptr, M: cute.Constexpr, N: cute.Constexpr):
    # engine
    ptr  = cute.make_ptr(cute.Float32, A_ptr, cute.AddressSpace.gmem)
    # layout: row-major (M, N) matrix
    layout = cute.make_layout((M, N), stride=(N, 1))
    # tensor
    A = cute.make_tensor(ptr, layout)

    cute.printf(A.layout)          # print the layout
    cute.printf(cute.size(A))      # total elements
```

**What to observe:**
- `A.layout` echoes back the layout you gave it — no magic yet.
- `A[i, j]` would dereference `ptr + i*N + j`.
- Column-major: flip `stride=(1, M)`.

**Launch:** single thread block, 1 thread is enough (we're just printing structure).

---

## Section 2 — Coordinate tensor (identity tensor)

**Goal:** a tensor whose *values are its own coordinates*, not data from memory. The engine is a counting integer, not a pointer.

Two ways to make one:

```python
# way 1: explicit
layout = cute.make_layout((M, N), stride=(N, 1))
crd_tensor = cute.make_tensor(0, layout)   # engine = counting from 0

# way 2: helper (uses identity layout under the hood)
crd_tensor = cute.make_identity_tensor((M, N))
```

**What to observe:**
- `crd_tensor[i, j]` returns the *coordinate tuple* `(i, j)`, not a memory value.
- Use case: after `local_partition`, each thread's slice of the coord tensor tells it exactly which `(row, col)` in the global tile it owns — critical for predication and TMA coordinate passing.
- Hierarchical shape: `make_identity_tensor(((BM, BN), (TM, TN)))` returns nested coordinate tuples.

**Kernel sketch:** print `crd_tensor(i, j)` for a small `(4, 4)` shape.

---

## Section 3 — smem tensor

**Goal:** allocate smem inside a kernel, wrap it as a `cute.Tensor`.

Two allocation modes:

```python
# static allocation (shape known at compile time)
smem_ptr = cute.arch.smem.alloc_smem(cute.Float32, BM * BK)

# dynamic allocation (kernel launched with smem_size=... bytes)
smem_ptr = cute.arch.smem.get_dyn_smem(cute.Float32)
```

Both return a `Pointer` with `AddressSpace.smem`. Then:

```python
layout = cute.make_layout((BM, BK), stride=(BK, 1))
sA     = cute.make_tensor(smem_ptr, layout)
```

**What to observe:**
- `sA.layout` identical to `gA.layout` if you used the same shape — the layout is just a mapping, it doesn't care about address space.
- smem pointer arithmetic is the same as gmem — `sA[i, k]` = `smem_ptr + i*BK + k`.
- For swizzled smem (to avoid bank conflicts), you'd pass a `ComposedLayout` (swizzle ∘ layout) — that's a later topic.

**Kernel sketch:** static alloc a `(32, 16)` fp32 smem tile. Thread 0 writes `threadIdx.x` to `sA[0, 0]`, then reads it back and prints it to confirm smem is live.

---

## Section 4 — rmem tensor (TensorSSA)

**Goal:** allocate a register fragment — the accumulator or operand tile owned entirely by one thread.

```python
# from an explicit layout
rC = cute.make_rmem_tensor(cute.make_layout((MMA_M, MMA_N)), cute.Float32)

# mirroring the shape/dtype of another tensor
rC = cute.make_rmem_tensor_like(some_tensor)
```

**What makes rmem special:**
- Returns a `TensorSSA` — this is an MLIR SSA value, not a pointer. It has no address in the conventional sense.
- Lives in the register file. The compiler maps it to `float` / `float4` / etc. locals.
- You cannot pass its address to `cp.async` or TMA — you can only read/write elements, or use `cute.copy` to move data into/out of it.
- Alignment: `make_rmem_tensor` uses 32-byte alignment so `.128` load/store instructions are legal.

**Kernel sketch:** allocate a `(4, 4)` fp32 rmem tensor, fill it with `threadIdx.x * 100 + i*4 + j`, print it.

---

## Section 5 — local_tile and local_partition

**Goal:** carve up a gmem tensor so each CTA / thread sees only its slice.

### `local_tile`

```python
# gA: (M, K) gmem tensor
# tile shape: (BM, BK), CTA coord: (block_m, block_k)
gA_cta = cute.local_tile(gA, (BM, BK), (block_m, block_k))
# result shape: (BM, BK) — the CTA's tile of gA
```

Under the hood this is `zipped_divide` + coord indexing. `gA_cta[i, k]` still reads from global memory, just offset to the right CTA tile.

### `local_partition`

```python
# sA: (BM, BK) smem tensor
# tiled_copy.layout_tv: thread-value layout (tells each thread which elements it owns)
# tidx: Int32 thread index
sA_thr = cute.local_partition(sA, tiled_copy.layout_tv, tidx)
# result: each thread's slice of sA
```

`local_partition` uses `logical_divide` on the layout: the "thread" mode is consumed by `tidx`, the "value" mode remains as the per-thread iteration space.

**Kernel sketch:**
- Create a `(128, 64)` gmem tensor.
- `local_tile` it with `(32, 16)` tile, pick CTA `(0, 0)`.
- Print the resulting layout to confirm it's a `(32, 16)` view starting at offset 0.
- Do it again for CTA `(1, 0)` and confirm the offset shifts by `32*64`.

---

## Section 6 — Thread-register vectorized copy (CopyUniversalOp)

**Goal:** the "normal" SIMT copy path. Each thread loads a vector of elements from gmem into rmem (or smem).

### Building the atom

```python
op   = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(op, cute.Float32, num_bits_per_copy=128)
# 128-bit → 4 fp32 elements per thread per instruction (LDG.128)
```

### Tiling it into a `TiledCopy`

```python
# layout_tv: (threads, values_per_thread)  — how threads & values tile the (BM, BN) block
tiled_copy = cute.make_tiled_copy(
    atom,
    layout_tv  = cute.make_layout((32, 4)),    # 32 threads × 4 vals
    tiler_mn   = cute.make_layout((BM, BN)),   # tile shape
)
```

### Partitioning and copying

```python
thr_copy = tiled_copy.get_thread_slice(tidx)
tgA = thr_copy.partition_S(gA_cta)   # source: thread's slice of gmem tile
trA = thr_copy.partition_D(rA)        # dest:   thread's slice of rmem frag

cute.copy(tiled_copy, tgA, trA)       # or: cute.autovec_copy(tgA, trA)
```

**What to observe:**
- `partition_S(gA_cta)` returns a tensor of shape `(CPY_VEC, CPY_M, CPY_N)` — the first dim is the vector (4 floats), the others iterate over the tile.
- `autovec_copy` is the lazy version — it infers the best vector width automatically.
- The actual PTX emitted is `LDG.E.128` for 128-bit loads.

**Kernel sketch:** copy a `(64, 64)` fp32 gmem tile into a `(64, 64)` rmem fragment using `CopyUniversalOp` + 128-bit, verify by summing elements.

---

## Section 7 — cp.async: non-bulk async gmem → smem (CopyG2SOp)

**Goal:** asynchronous copy that bypasses the register file entirely — data goes directly from L2/DRAM into smem via the LSU.

```python
op   = cute.nvgpu.cpasync.CopyG2SOp()            # cp.async.ca (cache all)
atom = cute.make_copy_atom(op, cute.Float16, num_bits_per_copy=128)
# 128-bit = 8 fp16 elements per thread

tiled_copy = cute.make_tiled_copy(atom, layout_tv, tiler_mn)
thr_copy   = tiled_copy.get_thread_slice(tidx)

tgA = thr_copy.partition_S(gA_cta)   # gmem source
tsA = thr_copy.partition_D(sA)        # smem destination   ← no rmem intermediate

cute.copy(tiled_copy, tgA, tsA)       # issues cp.async

# commit + wait
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)       # wait for all groups
cute.arch.sync_threads()
```

**Key differences from Section 6:**
- `partition_D` targets an **smem** tensor, not rmem.
- The copy is *asynchronous* — you must `commit_group` + `wait_group` before consuming smem.
- `wait_group(N)` means "wait until at most N async groups are pending" — `wait_group(0)` = wait all.
- `CacheMode.GLOBAL` variant: `CopyG2SOp(cache_mode=LoadCacheMode.ALWAYS)` (default) vs `LoadCacheMode.GLOBAL`.

**Kernel sketch:** async-copy a `(128, 64)` bf16 gmem matrix into a `(128, 64)` smem buffer, synchronize, then read back thread 0's element to verify.

---

## Section 8 — TMA: bulk async gmem → smem (CopyBulkTensorTileG2SOp)

**Goal:** the TMA unit copies an entire tile from gmem to smem in one hardware instruction, issued by a *single elected thread*, with mbarrier synchronization.

### Host side — build the TMA atom (outside the kernel)

```python
import torch

A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

# gmem tensor (host, just for descriptor construction)
gmem_layout = cute.make_layout((M, K), stride=(K, 1))
gA = cute.make_tensor(A.data_ptr(), gmem_layout)   # pointer as int, not inside jit

smem_layout = cute.make_layout((BM, BK), stride=(BK, 1))

tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
    cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
    gA,
    smem_layout,
    cta_tiler=(BM, BK),
)
# tma_tensor: has basis-stride elements so the layout can produce TMA coordinates
```

### Device side — partition and issue the copy

```python
@cute.kernel
def tma_load_demo(tma_atom, tma_tensor, ...):
    mbar = cute.arch.mbar.alloc_mbarrier(1)        # 1 mbarrier
    cute.arch.mbar.mbarrier_init(mbar, 1)          # arrive count = 1

    smem_ptr = cute.arch.smem.get_dyn_smem(cute.BFloat16)
    sA = cute.make_tensor(smem_ptr, smem_layout)

    # partition: tma_partition returns (smem_view, tma_coord_view)
    tDsD, tDgD = cute.nvgpu.cpasync.tma_partition(
        tma_atom,
        cta_coord   = (block_m, block_k),
        cta_layout  = cute.make_layout(1),
        smem_tensor = sA,
        gmem_tensor = tma_tensor,
    )

    # elected thread issues the load
    if cute.arch.elect_one():
        cute.arch.mbar.mbarrier_arrive_and_expect_tx(mbar, cute.size_in_bytes(sA))
        cute.copy(tma_atom, tDgD, tDsD)

    # all threads wait
    cute.arch.mbar.mbarrier_try_wait(mbar, phase=0)
    cute.arch.sync_threads()
```

**Key ideas:**
- TMA descriptor (`tma_tensor`) is built on the host and passed to the kernel as a kernel arg — it encodes the global tensor shape, strides, box size, and swizzle. The hardware reads it to figure out what to copy.
- `tma_partition` gives you a smem view and a *coordinate-keyed* view of the tma_tensor; the coord view is what you pass to `cute.copy`.
- Only **one** thread per CTA calls `cute.copy` for TMA (the `elect_one()` guard). The others still participate in the mbarrier arrive (`mbarrier_arrive`).
- `mbarrier_arrive_and_expect_tx` tells the barrier to also wait for `N` bytes of async data to arrive — this is what makes it synchronize with TMA completion.
- `mbarrier_try_wait(phase=0)` spins until the barrier is satisfied.

**Kernel sketch:** load a `(64, 64)` bf16 tile from a `(256, 256)` gmem matrix using TMA, print element `(0, 0)` of the smem tile for CTA `(0,0)` and `(1, 0)` to confirm correct tiling.

---

## Ordering of cells per section

Each section follows this pattern:

1. **Markdown** — abstract idea + the key API names
2. **Code** — build/print the layout/tensor structure (cheap, often `@cute.jit`)
3. **Code** — the kernel (`@cute.kernel`) that actually launches and verifies

---

## Open questions to decide before writing

1. For smem swizzle (Section 3 extension) — cover it here or in a separate `swizzle.ipynb`?
2. For cp.async double buffering — show it inline in Section 7 or defer to the matmul notebook?
3. TMA multicast (`CopyBulkTensorTileG2SMulticastOp`) — Section 8 extension or separate?
4. `autovec_copy` vs explicit `CopyUniversalOp` — just mention the difference or demo both?
