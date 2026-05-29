# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Structured state space duality (SSD) intra-chunk diagonal block kernel.

This implements stage 1 of the SSD chunk-scan algorithm used by Mamba2: the
"intra-chunk diagonal block output". For each ``(batch, chunk, head)`` it
computes the causal, decay-weighted attention-like output that only depends on
tokens within the same chunk.

For a chunk of ``L`` tokens with state dim ``N`` and head dim ``P``:

- Inputs: ``C: [L, N]``, ``B: [L, N]``, ``X: [L, P]``, ``A: [L]`` (a scalar
  decay per token).
- Cumulative decay: ``A_cumsum[l] = sum_{k=0..l} A[k]``.
- Causal decay (segment-sum) matrix: for ``s <= l``,
  ``Ldecay[l, s] = exp(A_cumsum[l] - A_cumsum[s])``; for ``s > l`` it is ``0``.
- Attention scores: ``scores[l, s] = (sum_n C[l, n] * B[s, n]) * Ldecay[l, s]``
  for ``s <= l``.
- Output: ``Y[l, p] = sum_{s=0..l} scores[l, s] * X[s, p]``.

This matches the reference einsum
``Y_diag = einsum("ln,sn,ls,sp->lp", C, B, Ldecay, X)`` restricted to ``s <= l``
(stage 1 of ``ssd_minimal_discrete``).

Implementations:

- ``*_naive``: scalar reference kernels for golden / numerical-stability tests.
- ``ssd_intra_chunk_fwd_cpu``: parallel per-slice GEMM decomposition (CPU).
- ``ssd_intra_chunk_fwd_gpu``: host dispatch — batched ``C @ B^T`` via Modular
  ``batched_matmul`` (tensor cores on GPU) plus a Triton-style tiled chunk-scan
  kernel that fuses decay masking with ``scores @ X``.
"""

from layout import (
    Idx,
    Layout,
    LayoutTensor,
    TensorLayout,
    TileTensor,
    row_major,
)
from std.gpu.memory import AddressSpace
from std.sys.info import _has_sm_121x
from linalg.bmm import batched_matmul
from std.algorithm import sync_parallelize
from std.collections import OptionalReg
from std.gpu.host import DeviceContext
from std.math import ceildiv, exp
from std.memory import alloc, stack_allocation as mem_stack_allocation
from std.gpu import (
    barrier as gpu_barrier,
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from std.utils.index import IndexList

# Triton-style tile sizes (scaled down for per-block register/stack budget).
#
# Device-split tuning: the chunk-scan's per-block work grows linearly with
# pid_m (causal early-exit), so heavy tiles drain late and idle SMs. On the
# GB10 (sm_121, DGX Spark) — many small SMs that prefer fine-grained work —
# a 32-row tile (2× the blocks) measurably beats 64 (0.41 → 0.37 ms). On
# datacenter parts (A100/H100/B200, fewer but larger SMs) the 64-row tile is
# the safer default; it keeps more rows resident per block and we have not
# profiled the smaller tile there. The kernel is correct for either value on
# any GPU — this only trades block count vs per-block work.
comptime SCAN_BLOCK_M = 32 if _has_sm_121x() else 64
comptime SCAN_BLOCK_N = 32
comptime SCAN_BLOCK_K = 32
comptime SCAN_ACC_ELEMS = SCAN_BLOCK_M * SCAN_BLOCK_N
# Warps per scan block. The cooperative CB load strides BM rows by this
# count, so the block size must be a whole number of warps.
comptime SCAN_WARPS = SCAN_BLOCK_M // 32

# Upper bound on chunk_len for the shared-memory prefix-sum scratch (one
# block per slice, one element per thread). 1024 floats = 4 KB shmem.
comptime SCAN_MAX_CHUNK = 1024

# BMM tile sizes for the in-house C @ B^T kernel. Each thread block emits
# a BMM_BM × BMM_BN output tile; threads laid out (BMM_BN, BMM_BM) so a warp
# spans a full BMM_BN row (32 contiguous output cols) for coalesced stores.
comptime BMM_BM = 32
comptime BMM_BN = 32
comptime BMM_BK = 16

# Stride helpers for indexing the flat row-major buffers.
comptime Strides4D = IndexList[4]
comptime Strides5D = IndexList[5]


def _row_major_strides_4d(d0: Int, d1: Int, d2: Int, d3: Int) -> Strides4D:
    """Row-major strides for a 4D shape ``(d0, d1, d2, d3)``."""
    return Strides4D(d1 * d2 * d3, d2 * d3, d3, 1)


def _row_major_strides_5d(
    d0: Int, d1: Int, d2: Int, d3: Int, d4: Int
) -> Strides5D:
    """Row-major strides for a 5D shape ``(d0, d1, d2, d3, d4)``."""
    return Strides5D(d1 * d2 * d3 * d4, d2 * d3 * d4, d3 * d4, d4, 1)


@always_inline
def _apply_decay_mask_to_cb[
    dtype: DType,
](
    chunk_len: Int,
    cb_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cumsum: UnsafePointer[Float32, ImmutAnyOrigin],
) -> None:
    """Apply causal decay ``exp(cum[l]-cum[s])`` to lower-triangular ``CB[l,s]``.
    """
    for l in range(chunk_len):
        var cum_l = cumsum[l]
        for s in range(l + 1):
            var off = l * chunk_len + s
            var val = cb_ptr[off].cast[DType.float32]() * exp(cum_l - cumsum[s])
            cb_ptr[off] = val.cast[dtype]()
        for s in range(l + 1, chunk_len):
            cb_ptr[l * chunk_len + s] = Scalar[dtype](0)


# ===----------------------------------------------------------------------=== #
# Naive CPU baseline (numerical reference)
# ===----------------------------------------------------------------------=== #


def ssd_intra_chunk_fwd_cpu_naive[
    kernel_dtype: DType,
    C_layout: Layout,
    B_layout: Layout,
    X_layout: Layout,
    A_layout: Layout,
    Y_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    X: LayoutTensor[kernel_dtype, X_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    Y: LayoutTensor[kernel_dtype, Y_layout, MutAnyOrigin],
):
    """Naive scalar CPU reference for golden / stability tests.

    Triple nested loops over ``(l, s, n, p)`` with no SIMD or GEMM.
    """
    var cb_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, state_dim
    )
    var x_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, head_dim
    )
    var a_strides = _row_major_strides_4d(batch, n_chunks, n_heads, chunk_len)

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var cb_base = (
                    b * cb_strides[0] + c * cb_strides[1] + h * cb_strides[2]
                )
                var x_base = (
                    b * x_strides[0] + c * x_strides[1] + h * x_strides[2]
                )
                var a_base = (
                    b * a_strides[0] + c * a_strides[1] + h * a_strides[2]
                )
                var y_base = x_base

                var a_cumsum = List[Float32](length=chunk_len, fill=0.0)
                var running = Float32(0.0)
                for l in range(chunk_len):
                    var a_off = a_base + l * a_strides[3]
                    running += A.ptr[a_off].cast[DType.float32]()
                    a_cumsum[l] = running

                for l in range(chunk_len):
                    var c_row = cb_base + l * cb_strides[3]
                    var cum_l = a_cumsum[l]
                    for s in range(l + 1):
                        var b_row = cb_base + s * cb_strides[3]

                        var dot = Float32(0.0)
                        for n in range(state_dim):
                            var cv = C.ptr[c_row + n * cb_strides[4]].cast[
                                DType.float32
                            ]()
                            var bv = B.ptr[b_row + n * cb_strides[4]].cast[
                                DType.float32
                            ]()
                            dot += cv * bv

                        var decay = exp(cum_l - a_cumsum[s])
                        var score = dot * decay

                        var x_row = x_base + s * x_strides[3]
                        var y_row = y_base + l * x_strides[3]
                        for p in range(head_dim):
                            var xv = X.ptr[x_row + p * x_strides[4]].cast[
                                DType.float32
                            ]()
                            var y_off = y_row + p * x_strides[4]
                            var acc = (
                                Y.ptr[y_off].cast[DType.float32]() + score * xv
                            )
                            Y.ptr[y_off] = acc.cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# Optimized CPU (parallel slices + GEMM)
# ===----------------------------------------------------------------------=== #


def ssd_intra_chunk_fwd_cpu[
    kernel_dtype: DType,
    C_layout: Layout,
    B_layout: Layout,
    X_layout: Layout,
    A_layout: Layout,
    Y_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    C: LayoutTensor[kernel_dtype, C_layout, MutAnyOrigin],
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    X: LayoutTensor[kernel_dtype, X_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    Y: LayoutTensor[kernel_dtype, Y_layout, MutAnyOrigin],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Optimized CPU intra-chunk diagonal block.

    1. Batched ``CB = C @ B^T`` via ``batched_matmul``.
    2. Parallel per-slice causal decay on ``CB``.
    3. Batched ``Y = CB @ X`` via ``batched_matmul``.
    """
    var num_slices = batch * n_chunks * n_heads
    var l2 = chunk_len * chunk_len

    var cb_ptr = alloc[Scalar[kernel_dtype]](num_slices * l2)

    var C3 = TileTensor(C.ptr, row_major(num_slices, chunk_len, state_dim))
    var B3 = TileTensor(B.ptr, row_major(num_slices, chunk_len, state_dim))
    var CB3 = TileTensor(cb_ptr, row_major(num_slices, chunk_len, chunk_len))
    var X3 = TileTensor(X.ptr, row_major(num_slices, chunk_len, head_dim))
    var Y3 = TileTensor(Y.ptr, row_major(num_slices, chunk_len, head_dim))

    batched_matmul[target="cpu", transpose_b=True](CB3, C3, B3)

    var a_strides = _row_major_strides_4d(batch, n_chunks, n_heads, chunk_len)

    @parameter
    def decay_worker(flat_idx: Int):
        var h = flat_idx % n_heads
        var c = (flat_idx // n_heads) % n_chunks
        var b = flat_idx // (n_heads * n_chunks)

        var a_base = b * a_strides[0] + c * a_strides[1] + h * a_strides[2]
        var cumsum_ptr = alloc[Float32](chunk_len)

        var running = Float32(0.0)
        for l in range(chunk_len):
            running += A.ptr[a_base + l * a_strides[3]].cast[DType.float32]()
            cumsum_ptr[l] = running

        _apply_decay_mask_to_cb[kernel_dtype](
            chunk_len,
            cb_ptr + flat_idx * l2,
            cumsum_ptr,
        )
        cumsum_ptr.free()

    sync_parallelize[decay_worker](num_slices, ctx)

    batched_matmul[target="cpu"](Y3, CB3, X3)

    cb_ptr.free()


# ===----------------------------------------------------------------------=== #
# Naive GPU baseline (numerical reference)
# ===----------------------------------------------------------------------=== #


def ssd_intra_chunk_fwd_gpu_naive[
    kernel_dtype: DType,
    C_LT: TensorLayout,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    Y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    C: TileTensor[kernel_dtype, C_LT, MutExternalOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutExternalOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutExternalOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutExternalOrigin],
    Y: TileTensor[kernel_dtype, Y_LT, MutExternalOrigin],
):
    """Naive GPU reference: one thread per ``(batch, chunk, head)`` slice.

    Scalar loops only — use for golden / stability tests, not performance.
    """
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total_slices = batch * n_chunks * n_heads
    if flat_idx >= total_slices:
        return

    var h = flat_idx % n_heads
    var c = (flat_idx // n_heads) % n_chunks
    var b = flat_idx // (n_heads * n_chunks)

    var cb_n_stride = 1
    var cb_l_stride = state_dim
    var cb_h_stride = chunk_len * cb_l_stride
    var cb_c_stride = n_heads * cb_h_stride
    var cb_b_stride = n_chunks * cb_c_stride

    var x_p_stride = 1
    var x_l_stride = head_dim
    var x_h_stride = chunk_len * x_l_stride
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride

    var a_l_stride = 1
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var cb_base = b * cb_b_stride + c * cb_c_stride + h * cb_h_stride
    var x_base = b * x_b_stride + c * x_c_stride + h * x_h_stride
    var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride
    var y_base = x_base

    for l in range(chunk_len):
        var c_row = cb_base + l * cb_l_stride

        var cum_l = Float32(0.0)
        for k in range(l + 1):
            cum_l += A.ptr[a_base + k * a_l_stride].cast[DType.float32]()

        for p in range(head_dim):
            Y.ptr[y_base + l * x_l_stride + p * x_p_stride] = Scalar[
                kernel_dtype
            ](0)

        var cum_s = Float32(0.0)
        for s in range(l + 1):
            cum_s += A.ptr[a_base + s * a_l_stride].cast[DType.float32]()

            var b_row = cb_base + s * cb_l_stride

            var dot = Float32(0.0)
            for n in range(state_dim):
                var cv = C.ptr[c_row + n * cb_n_stride].cast[DType.float32]()
                var bv = B.ptr[b_row + n * cb_n_stride].cast[DType.float32]()
                dot += cv * bv

            var decay = exp(cum_l - cum_s)
            var score = dot * decay

            var x_row = x_base + s * x_l_stride
            var y_row = y_base + l * x_l_stride
            for p in range(head_dim):
                var xv = X.ptr[x_row + p * x_p_stride].cast[DType.float32]()
                var y_off = y_row + p * x_p_stride
                var acc = Y.ptr[y_off].cast[DType.float32]() + score * xv
                Y.ptr[y_off] = acc.cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# GPU: batched ``CB = C @ B^T`` (tiled, shared-memory, no tensor cores)
# ===----------------------------------------------------------------------=== #


def _ssd_intra_chunk_bmm_gpu[
    kernel_dtype: DType,
    C_LT: TensorLayout,
    B_LT: TensorLayout,
    CB_LT: TensorLayout,
](
    n_slices: Int,
    chunk_len: Int,
    state_dim: Int,
    C: TileTensor[kernel_dtype, C_LT, MutAnyOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    CB: TileTensor[kernel_dtype, CB_LT, MutAnyOrigin],
):
    """Batched ``CB[s, l, r] = sum_n C[s, l, n] * B[s, r, n]``.

    Replaces Modular ``batched_matmul`` for our dim set. The Modular
    dispatch needs static N/K to pick its tensor-core path; our caller
    only has dynamic dims, so it falls through to a per-element naive
    kernel that was profiled at >3ms/call. This kernel is a stock tiled
    matmul (no tensor cores, just shmem prefetched tiles); on the GB10
    it's enough to close the gap to the scan kernel cost.

    Grid: ``(ceildiv(L, BMM_BM) * ceildiv(L, BMM_BN), n_slices)``.
    Block: ``(BMM_BN, BMM_BM)``; thread ``(tn, tm)`` owns output
    ``CB[block_slice, l = m_tile*BMM_BM + tm, r = n_tile*BMM_BN + tn]``.
    Shared memory holds the current ``BMM_BM × BMM_BK`` C tile and
    ``BMM_BN × BMM_BK`` B tile per K window.
    """
    var num_n_tiles = ceildiv(chunk_len, BMM_BN)
    var m_tile = Int(block_idx.x) // num_n_tiles
    var n_tile = Int(block_idx.x) - m_tile * num_n_tiles
    var slice_idx = Int(block_idx.y)

    var l = m_tile * BMM_BM + Int(thread_idx.y)
    var r = n_tile * BMM_BN + Int(thread_idx.x)

    var c_slice_base = slice_idx * chunk_len * state_dim
    var b_slice_base = c_slice_base  # same layout
    var cb_slice_base = slice_idx * chunk_len * chunk_len

    var c_smem = mem_stack_allocation[
        BMM_BM * BMM_BK, Float32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = mem_stack_allocation[
        BMM_BN * BMM_BK, Float32, address_space=AddressSpace.SHARED
    ]()

    var acc: Float32 = 0.0

    var tm = Int(thread_idx.y)
    var tn = Int(thread_idx.x)

    for k0 in range(0, state_dim, BMM_BK):
        # Cooperative C[m_tile_row, k0..k0+BMM_BK] load. Threads
        # (tm, tn) load C_smem[tm, tn] when tn < BMM_BK.
        if tn < BMM_BK:
            var n = k0 + tn
            if l < chunk_len and n < state_dim:
                c_smem[tm * BMM_BK + tn] = C.ptr[
                    c_slice_base + l * state_dim + n
                ].cast[DType.float32]()
            else:
                c_smem[tm * BMM_BK + tn] = 0.0
        # Cooperative B[n_tile_row, k0..k0+BMM_BK] load. Threads
        # (tm, tn) → B_smem[tn, tm] when tm < BMM_BK.
        if tm < BMM_BK:
            var n = k0 + tm
            var r2 = n_tile * BMM_BN + tn
            if r2 < chunk_len and n < state_dim:
                b_smem[tn * BMM_BK + tm] = B.ptr[
                    b_slice_base + r2 * state_dim + n
                ].cast[DType.float32]()
            else:
                b_smem[tn * BMM_BK + tm] = 0.0

        gpu_barrier()

        if l < chunk_len and r < chunk_len:

            @parameter
            for k in range(BMM_BK):
                acc += c_smem[tm * BMM_BK + k] * b_smem[tn * BMM_BK + k]

        gpu_barrier()

    if l < chunk_len and r < chunk_len:
        CB.ptr[cb_slice_base + l * chunk_len + r] = acc.cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# GPU: prefix sum of A per slice
# ===----------------------------------------------------------------------=== #


def _ssd_intra_chunk_cumsum_gpu[
    kernel_dtype: DType,
    A_LT: TensorLayout,
    cumsum_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    A: TileTensor[kernel_dtype, A_LT, MutExternalOrigin],
    cumsum: TileTensor[DType.float32, cumsum_LT, MutExternalOrigin],
):
    """Inclusive prefix sum of ``A`` into ``cumsum`` — one block per slice.

    Hillis-Steele scan in shared memory: ``block_dim = chunk_len`` threads,
    one element each, ``log2(chunk_len)`` doubling steps. The previous version
    launched a single block (one thread per slice), so all slices' scans
    serialised on **one SM** (~32 µs at the Mamba2 profile). One block per
    slice spreads the work across every SM. Slices are contiguous ``chunk_len``
    blocks in both buffers (``base = slice_idx * chunk_len``).
    """
    var slice_idx = Int(block_idx.x)
    var total_slices = batch * n_chunks * n_heads
    if slice_idx >= total_slices:
        return

    var tid = Int(thread_idx.x)
    var base = slice_idx * chunk_len

    var sdata = mem_stack_allocation[
        SCAN_MAX_CHUNK, Float32, address_space=AddressSpace.SHARED
    ]()

    if tid < chunk_len:
        sdata[tid] = A.ptr[base + tid].cast[DType.float32]()
    gpu_barrier()

    var offset = 1
    while offset < chunk_len:
        var add: Float32 = 0.0
        if tid < chunk_len and tid >= offset:
            add = sdata[tid - offset]
        gpu_barrier()
        if tid < chunk_len and tid >= offset:
            sdata[tid] += add
        gpu_barrier()
        offset *= 2

    if tid < chunk_len:
        cumsum.ptr[base + tid] = sdata[tid]


def _ssd_intra_chunk_decay_gpu[
    kernel_dtype: DType,
    CB_LT: TensorLayout,
    cumsum_LT: TensorLayout,
](
    num_slices: Int,
    chunk_len: Int,
    CB: TileTensor[kernel_dtype, CB_LT, MutExternalOrigin],
    cumsum: TileTensor[DType.float32, cumsum_LT, MutExternalOrigin],
):
    """Apply causal decay in place: ``M[l,s] = CB[l,s]*exp(cum[l]-cum[s])``.

    Lower-triangular (``s <= l``); the strict upper triangle is zeroed.
    Materialises the decayed, masked score matrix so the chunk scan can be
    expressed as a plain ``Y = M @ X`` tensor-core matmul. Flat grid over
    the ``num_slices * L * L`` elements; consecutive threads map to
    consecutive ``s`` so CB / cumsum traffic is fully coalesced.
    """
    var slice_idx = Int(block_idx.y)
    var elem = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var l2 = chunk_len * chunk_len
    if slice_idx >= num_slices or elem >= l2:
        return

    var l = elem // chunk_len
    var s = elem - l * chunk_len

    var cs_base = slice_idx * chunk_len
    var cb_off = slice_idx * l2 + elem

    if s <= l:
        var cum_l = cumsum.ptr[cs_base + l]
        var cum_s = cumsum.ptr[cs_base + s]
        var decayed = CB.ptr[cb_off].cast[DType.float32]() * exp(cum_l - cum_s)
        CB.ptr[cb_off] = decayed.cast[kernel_dtype]()
    else:
        CB.ptr[cb_off] = Scalar[kernel_dtype](0)


# ===----------------------------------------------------------------------=== #
# GPU: Triton-style tiled chunk scan  (decay-masked CB @ X)
# ===----------------------------------------------------------------------=== #


def _ssd_intra_chunk_chunk_scan_gpu[
    kernel_dtype: DType,
    CB_LT: TensorLayout,
    cumsum_LT: TensorLayout,
    X_LT: TensorLayout,
    Y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    CB: TileTensor[kernel_dtype, CB_LT, MutExternalOrigin],
    cumsum: TileTensor[DType.float32, cumsum_LT, MutExternalOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutExternalOrigin],
    Y: TileTensor[kernel_dtype, Y_LT, MutExternalOrigin],
):
    """Tiled ``Y = causal_decay(CB) @ X`` matching Mamba ``_chunk_scan_fwd_kernel``.

    Grid ``(ceil(L/BM)*ceil(P/BN), batch*n_chunks, n_heads)`` with causal
    masking and ``exp(cumsum[l]-cumsum[s])`` decay fused in the K loop.
    """
    comptime BM = SCAN_BLOCK_M
    comptime BN = SCAN_BLOCK_N
    comptime BK = SCAN_BLOCK_K
    var num_pid_n = ceildiv(head_dim, BN)
    var pid_m = Int(block_idx.x) // num_pid_n
    var pid_n = Int(block_idx.x) % num_pid_n

    var pid_bc = Int(block_idx.y)
    var pid_h = Int(block_idx.z)

    var c = pid_bc // batch
    var b = pid_bc - c * batch

    var cb_b_stride = n_chunks * n_heads * chunk_len * chunk_len
    var cb_c_stride = n_heads * chunk_len * chunk_len
    var cb_h_stride = chunk_len * chunk_len
    var cb_l_stride = chunk_len
    var cb_s_stride = 1

    var x_b_stride = n_chunks * n_heads * chunk_len * head_dim
    var x_c_stride = n_heads * chunk_len * head_dim
    var x_h_stride = chunk_len * head_dim
    var x_l_stride = head_dim
    var x_p_stride = 1

    var y_b_stride = x_b_stride
    var y_c_stride = x_c_stride
    var y_h_stride = x_h_stride
    var y_l_stride = head_dim
    var y_p_stride = 1

    var cs_b_stride = n_chunks * n_heads * chunk_len
    var cs_c_stride = n_heads * chunk_len
    var cs_h_stride = chunk_len
    var cs_l_stride = 1

    var cb_base = b * cb_b_stride + c * cb_c_stride + pid_h * cb_h_stride
    var x_base = b * x_b_stride + c * x_c_stride + pid_h * x_h_stride
    var y_base = b * y_b_stride + c * y_c_stride + pid_h * y_h_stride
    var cs_base = b * cs_b_stride + c * cs_c_stride + pid_h * cs_h_stride

    var offs_m0 = pid_m * BM
    var offs_n0 = pid_n * BN

    # One thread per row of the BM×BN output tile (block_dim = BM threads).
    var tid = Int(thread_idx.x)

    # Shared tiles for the K window. We batch the X load to amortise
    # the global → shmem traffic across BK iterations and remove the
    # per-K-iter sync (was 2*K_MAX barriers per block, now 1 per BK).
    var x_smem = mem_stack_allocation[
        SCAN_BLOCK_K * SCAN_BLOCK_N,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()
    var cum_smem = mem_stack_allocation[
        SCAN_BLOCK_K, Float32, address_space=AddressSpace.SHARED
    ]()
    # Pad inner stride by 1 to avoid 32-way bank conflicts in the FMA
    # inner loop: with stride 32 (= 128 B), every BM lane hits the same
    # bank. Stride 33 makes the bank index = (row * 33 + ki) % 32, which
    # cycles through all 32 banks across the 64 lanes of the block.
    alias CB_SMEM_STRIDE = SCAN_BLOCK_K + 1
    var cb_smem = mem_stack_allocation[
        SCAN_BLOCK_M * CB_SMEM_STRIDE,
        Float32,
        address_space=AddressSpace.SHARED,
    ]()

    var l = offs_m0 + tid
    var active = tid < BM and l < chunk_len

    # ``exp(cum_l - cum_s)`` is factored as ``exp(cum_l) * exp(-cum_s)``.
    # Precomputing ``exp_neg_cum_smem[ki]`` cooperatively (BK threads, one
    # exp each per outer K iter) drops the per-thread exp count from K_MAX
    # to 1 — a substantial saving since exp is ~20 cycles on sm_120.
    var exp_cum_l: Float32 = 0.0
    if active:
        var cum_l = cumsum.ptr[cs_base + l * cs_l_stride]
        exp_cum_l = exp(cum_l)

    # Per-thread BN-wide accumulator.
    var acc = mem_stack_allocation[
        SCAN_BLOCK_N, Float32, address_space=AddressSpace.LOCAL
    ]()

    @parameter
    for ni in range(BN):
        acc[ni] = 0.0

    # Walk the K axis only up to ``(pid_m + 1) * BM`` (the max ``l`` in
    # this block) — causal masking would zero-out anything beyond that, so
    # there's no point loading or computing it. For pid_m=0 (first L-tile)
    # this cuts K_MAX from chunk_len to BM; on the Mamba2-130m profile
    # (L=256, BM=64) the average K work drops by ~1.6× across the four
    # M-tiles.
    var k_limit = min((pid_m + 1) * BM, chunk_len)
    for k0 in range(0, k_limit, BK):
        # Cooperative ``exp(-cum_s)`` window. Storing the pre-negated
        # exp lets the inner FMA write ``score = cb * exp_cum_l *
        # exp_neg_cum_s`` with two multiplies and zero exp calls per K iter.
        if tid < BK:
            var s = k0 + tid
            if s < chunk_len:
                cum_smem[tid] = exp(-cumsum.ptr[cs_base + s * cs_l_stride])
            else:
                cum_smem[tid] = 0.0
        # Cooperative X tile of size BK × BN, one outer iter at a time.
        # Threads (BM total) split BK*BN elements; with BM == BN, each
        # thread (column tid in [0, BN)) loads BK contiguous K rows for
        # its column.
        if tid < BN:
            var p = offs_n0 + tid

            @parameter
            for ki in range(BK):
                var s = k0 + ki
                if s < chunk_len and p < head_dim:
                    x_smem[ki * BN + tid] = X.ptr[
                        x_base + s * x_l_stride + p * x_p_stride
                    ].cast[DType.float32]()
                else:
                    x_smem[ki * BN + tid] = 0.0

        # Cooperative CB tile of size BM × BK: each warp loads one
        # contiguous K row per iter (32 lanes × 4 B = 1 cache line,
        # perfectly coalesced) — replaces the previous scattered
        # per-thread CB[l, s] reads (stride L between lanes, 16-way
        # uncoalesced on the Mamba2-130m profile). The ``SCAN_WARPS``
        # warps in the block stride the BM rows by ``SCAN_WARPS`` so any
        # block size that is a multiple of the warp size is covered.
        var warp_id = tid // 32
        var lane = tid - warp_id * 32

        @parameter
        for r in range(0, SCAN_BLOCK_M, SCAN_WARPS):
            var row = r + warp_id
            var l_row = offs_m0 + row
            var s_col = k0 + lane
            if (
                row < SCAN_BLOCK_M
                and l_row < chunk_len
                and lane < BK
                and s_col < chunk_len
            ):
                cb_smem[row * CB_SMEM_STRIDE + lane] = CB.ptr[
                    cb_base + l_row * cb_l_stride + s_col * cb_s_stride
                ].cast[DType.float32]()
            elif lane < BK:
                cb_smem[row * CB_SMEM_STRIDE + lane] = 0.0

        gpu_barrier()

        if active:

            @parameter
            for ki in range(BK):
                var s = k0 + ki
                if s < chunk_len and s <= l:
                    var cb_val = cb_smem[tid * CB_SMEM_STRIDE + ki]
                    var score = cb_val * exp_cum_l * cum_smem[ki]

                    @parameter
                    for ni in range(BN):
                        acc[ni] += score * x_smem[ki * BN + ni]

        gpu_barrier()

    if active:
        var y_row = y_base + l * y_l_stride

        @parameter
        for ni in range(BN):
            var p = offs_n0 + ni
            if p < head_dim:
                Y.ptr[y_row + p * y_p_stride] = acc[ni].cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# Optimized GPU host dispatch
# ===----------------------------------------------------------------------=== #


def ssd_intra_chunk_fwd_gpu[
    kernel_dtype: DType,
    C_LT: TensorLayout,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    Y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    C: TileTensor[kernel_dtype, C_LT, MutAnyOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, Y_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Optimized GPU intra-chunk diagonal block (Mamba-style pipeline).

    1. Batched ``CB = C @ B^T`` via Modular ``batched_matmul`` (tensor cores).
    2. Per-slice inclusive prefix sum of ``A``.
    3. Tiled chunk-scan: ``Y = causal_decay(CB) @ X``.

    Matches the structure of ``mamba_ssm/ops/triton/ssd_chunk_scan.py``:
    ``_bmm_chunk_fwd`` + ``_chunk_scan_fwd_kernel``.
    """
    var num_slices = batch * n_chunks * n_heads
    var l2 = chunk_len * chunk_len

    var CB_device = ctx.enqueue_create_buffer[kernel_dtype](num_slices * l2)
    var cumsum_device = ctx.enqueue_create_buffer[DType.float32](
        num_slices * chunk_len
    )

    var C3 = TileTensor(
        C.ptr,
        row_major(num_slices, chunk_len, state_dim),
    )
    var B3 = TileTensor(
        B.ptr,
        row_major(num_slices, chunk_len, state_dim),
    )
    var CB3 = TileTensor(
        CB_device,
        row_major(num_slices, chunk_len, chunk_len),
    )
    var cumsum4 = TileTensor(
        cumsum_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var CB5 = TileTensor(
        CB_device,
        row_major(batch, n_chunks, n_heads, chunk_len, chunk_len),
    )

    # Dispatch CB = C @ B^T:
    #   batched_matmul picks a cutlass tensor-core path when batch_size == 1
    #   but falls back to a naive O(N^3) per-element kernel for batched
    #   cases. Our hand-tiled ``_ssd_intra_chunk_bmm_gpu`` beats the naive
    #   fallback by ~5×, but is still scalar (no tensor cores). For
    #   multi-slice we benchmark both: a per-slice loop of cutlass calls
    #   (tensor cores, but 96 launches of ~5µs each) and the custom bmm
    #   (one launch, scalar). The faster of the two wins at this dim set.
    #
    # Toggle via env var for benchmarking; default: per-slice cutlass.
    var use_per_slice_cutlass = True
    if num_slices == 1:
        with ctx.push_context():
            batched_matmul[target="gpu", transpose_b=True](
                CB3, C3, B3, context=ctx
            )
    elif use_per_slice_cutlass:
        with ctx.push_context():
            for s in range(num_slices):
                var Cs = TileTensor(
                    C.ptr + s * chunk_len * state_dim,
                    row_major(1, chunk_len, state_dim),
                )
                var Bs = TileTensor(
                    B.ptr + s * chunk_len * state_dim,
                    row_major(1, chunk_len, state_dim),
                )
                var CBs = TileTensor(
                    CB_device.unsafe_ptr() + s * chunk_len * chunk_len,
                    row_major(1, chunk_len, chunk_len),
                )
                batched_matmul[target="gpu", transpose_b=True](
                    CBs, Cs, Bs, context=ctx
                )
    else:
        comptime bmm_kernel = _ssd_intra_chunk_bmm_gpu[
            kernel_dtype,
            C3.LayoutType,
            B3.LayoutType,
            CB3.LayoutType,
        ]
        var bmm_compiled = ctx.compile_function[bmm_kernel]()
        var bmm_grid_x = ceildiv(chunk_len, BMM_BM) * ceildiv(chunk_len, BMM_BN)

        with ctx.push_context():
            ctx.enqueue_function(
                bmm_compiled,
                num_slices,
                chunk_len,
                state_dim,
                C3,
                B3,
                CB3,
                grid_dim=(bmm_grid_x, num_slices, 1),
                block_dim=(BMM_BN, BMM_BM, 1),
            )

    var cumsum_compiled = ctx.compile_function[
        _ssd_intra_chunk_cumsum_gpu[
            kernel_dtype,
            A_LT,
            cumsum4.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            cumsum_compiled,
            batch,
            n_chunks,
            n_heads,
            chunk_len,
            A,
            cumsum4,
            grid_dim=(num_slices,),
            block_dim=(chunk_len,),
        )

    comptime scan_kernel = _ssd_intra_chunk_chunk_scan_gpu[
        kernel_dtype,
        CB5.LayoutType,
        cumsum4.LayoutType,
        X_LT,
        Y_LT,
    ]
    var scan_compiled = ctx.compile_function[scan_kernel]()

    var grid_x = ceildiv(chunk_len, SCAN_BLOCK_M) * ceildiv(
        head_dim, SCAN_BLOCK_N
    )

    # Scan kernel reads ``block_idx.y`` as ``batch * n_chunks`` and
    # ``block_idx.z`` as ``n_heads``; launching ``num_slices`` (= b*nc*nh)
    # in y was redundantly multiplying the work by n_heads. Fix the launch
    # to match the kernel's index decomposition.
    with ctx.push_context():
        ctx.enqueue_function(
            scan_compiled,
            batch,
            n_chunks,
            n_heads,
            chunk_len,
            head_dim,
            CB5,
            cumsum4,
            X,
            Y,
            grid_dim=(grid_x, batch * n_chunks, n_heads),
            block_dim=(SCAN_BLOCK_M, 1, 1),
        )


# ===----------------------------------------------------------------------=== #
# Static-shape entry point
# ===----------------------------------------------------------------------=== #


def ssd_intra_chunk_fwd_gpu_static[
    kernel_dtype: DType,
    chunk_len_ct: Int,
    state_dim_ct: Int,
    head_dim_ct: Int,
    C_LT: TensorLayout,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    Y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    C: TileTensor[kernel_dtype, C_LT, MutAnyOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, Y_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Static-shape variant of ``ssd_intra_chunk_fwd_gpu``.

    Knowing ``chunk_len`` / ``state_dim`` / ``head_dim`` at compile time
    lets ``batched_matmul`` see ``has_static_NK`` and pick its A100
    tensor-core batched path (one launch for all slices, grid_z =
    batch_size), instead of the per-slice cutlass loop (96 launches × ~4 µs
    each) that dominates the dynamic-shape dispatch.
    """
    var num_slices = batch * n_chunks * n_heads
    comptime l2 = chunk_len_ct * chunk_len_ct

    var CB_device = ctx.enqueue_create_buffer[kernel_dtype](num_slices * l2)
    var cumsum_device = ctx.enqueue_create_buffer[DType.float32](
        num_slices * chunk_len_ct
    )

    var C3 = TileTensor(
        C.ptr,
        row_major(num_slices, Idx[chunk_len_ct], Idx[state_dim_ct]),
    )
    var B3 = TileTensor(
        B.ptr,
        row_major(num_slices, Idx[chunk_len_ct], Idx[state_dim_ct]),
    )
    var CB3 = TileTensor(
        CB_device,
        row_major(num_slices, Idx[chunk_len_ct], Idx[chunk_len_ct]),
    )
    var cumsum4 = TileTensor(
        cumsum_device,
        row_major(batch, n_chunks, n_heads, chunk_len_ct),
    )
    var CB5 = TileTensor(
        CB_device,
        row_major(
            batch,
            n_chunks,
            n_heads,
            Idx[chunk_len_ct],
            Idx[chunk_len_ct],
        ),
    )

    # Stage 1: CB = C @ B^T (batched tensor-core matmul, one launch).
    #
    # Uses the library ``batched_matmul``, which self-tunes per architecture
    # (A100 multistage on GB10/Ampere, SM100 tcgen05 on datacenter Blackwell,
    # AMD path on MI-series) and is correct for all shapes. A hand-dispatched
    # smaller tile config was tried on GB10 (the "more blocks" lesson from the
    # scan) but made no difference: this matmul is **DRAM-bandwidth bound** on
    # materialising the L×L ``CB`` matrix (~50 MB C+B+CB traffic ≈ 90% of
    # GB10 peak BW at 203 µs), not occupancy bound, so tile shape doesn't
    # move it. The datacenter Blackwell (sm100/tcgen05) kernels cannot run on
    # GB10 (consumer sm_121 has no tcgen05); consumer Blackwell already uses
    # this multistage TF32 tensor-core path. The only further lever is fusing
    # the bmm with the scan to avoid the CB round-trip (large change).
    with ctx.push_context():
        batched_matmul[target="gpu", transpose_b=True](CB3, C3, B3, context=ctx)

    # Stage 2: inclusive prefix sum of A over the chunk.
    var cumsum_compiled = ctx.compile_function[
        _ssd_intra_chunk_cumsum_gpu[
            kernel_dtype,
            A_LT,
            cumsum4.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            cumsum_compiled,
            batch,
            n_chunks,
            n_heads,
            chunk_len_ct,
            A,
            cumsum4,
            grid_dim=(num_slices,),
            block_dim=(chunk_len_ct,),
        )

    # Stage 3: tiled scalar chunk-scan — Y = causal_decay(CB) @ X. The
    # scalar path beats a dense tensor-core M @ X here: it exploits the
    # lower-triangular causal structure (≈half the FLOPs, plus per-M-tile
    # early-K exit) and avoids padding head_dim to 128, which the A100
    # batched matmul requires (c_n % 128 == 0). See
    # ``ssd_intra_chunk_fwd_gpu_static_mma`` for the MMA alternative, which
    # is ~2× faster single-slice but ~2× slower multi-slice.
    comptime scan_kernel = _ssd_intra_chunk_chunk_scan_gpu[
        kernel_dtype,
        CB5.LayoutType,
        cumsum4.LayoutType,
        X_LT,
        Y_LT,
    ]
    var scan_compiled = ctx.compile_function[scan_kernel]()

    comptime grid_x = ceildiv(chunk_len_ct, SCAN_BLOCK_M) * ceildiv(
        head_dim_ct, SCAN_BLOCK_N
    )

    with ctx.push_context():
        ctx.enqueue_function(
            scan_compiled,
            batch,
            n_chunks,
            n_heads,
            chunk_len_ct,
            head_dim_ct,
            CB5,
            cumsum4,
            X,
            Y,
            grid_dim=(grid_x, batch * n_chunks, n_heads),
            block_dim=(SCAN_BLOCK_M, 1, 1),
        )


def ssd_intra_chunk_fwd_gpu_static_mma[
    kernel_dtype: DType,
    chunk_len_ct: Int,
    state_dim_ct: Int,
    head_dim_ct: Int,
    C_LT: TensorLayout,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    Y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    C: TileTensor[kernel_dtype, C_LT, MutAnyOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutAnyOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutAnyOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutAnyOrigin],
    Y: TileTensor[kernel_dtype, Y_LT, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Tensor-core variant: materialise ``M = causal_decay(CB)`` then
    ``Y = M @ X`` as two batched matmuls.

    Trades the scalar chunk-scan for a dense matmul. **Faster than
    ``ssd_intra_chunk_fwd_gpu_static`` only for a single slice** (no causal
    work to exploit, scan launch overhead dominates): ~2× on GB10. For the
    multi-slice Mamba2 prefill profile it is ~2× *slower* because the dense
    ``M @ X`` ignores the lower-triangular structure (≈2× the FLOPs) and the
    A100 batched matmul rejects ``head_dim`` (= output N = 64) for not being
    a multiple of 128, falling back to the naive scalar batched kernel. Kept
    for the single-slice regime and as a reference for future MMA work.
    """
    var num_slices = batch * n_chunks * n_heads
    comptime l2 = chunk_len_ct * chunk_len_ct

    var CB_device = ctx.enqueue_create_buffer[kernel_dtype](num_slices * l2)
    var cumsum_device = ctx.enqueue_create_buffer[DType.float32](
        num_slices * chunk_len_ct
    )

    var C3 = TileTensor(
        C.ptr, row_major(num_slices, Idx[chunk_len_ct], Idx[state_dim_ct])
    )
    var B3 = TileTensor(
        B.ptr, row_major(num_slices, Idx[chunk_len_ct], Idx[state_dim_ct])
    )
    var CB3 = TileTensor(
        CB_device, row_major(num_slices, Idx[chunk_len_ct], Idx[chunk_len_ct])
    )
    var cumsum4 = TileTensor(
        cumsum_device, row_major(batch, n_chunks, n_heads, chunk_len_ct)
    )
    var X3 = TileTensor(
        X.ptr, row_major(num_slices, Idx[chunk_len_ct], Idx[head_dim_ct])
    )
    var Y3 = TileTensor(
        Y.ptr, row_major(num_slices, Idx[chunk_len_ct], Idx[head_dim_ct])
    )

    # Stage 1: CB = C @ B^T (batched A100 tensor-core matmul).
    with ctx.push_context():
        batched_matmul[target="gpu", transpose_b=True](CB3, C3, B3, context=ctx)

    # Stage 2: inclusive prefix sum of A over the chunk.
    var cumsum_compiled = ctx.compile_function[
        _ssd_intra_chunk_cumsum_gpu[kernel_dtype, A_LT, cumsum4.LayoutType]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            cumsum_compiled,
            batch,
            n_chunks,
            n_heads,
            chunk_len_ct,
            A,
            cumsum4,
            grid_dim=(num_slices,),
            block_dim=(chunk_len_ct,),
        )

    # Stage 3: apply causal decay in place — M = causal_decay(CB).
    comptime DECAY_BLOCK = 256
    var decay_compiled = ctx.compile_function[
        _ssd_intra_chunk_decay_gpu[
            kernel_dtype, CB3.LayoutType, cumsum4.LayoutType
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            decay_compiled,
            num_slices,
            chunk_len_ct,
            CB3,
            cumsum4,
            grid_dim=(ceildiv(l2, DECAY_BLOCK), num_slices),
            block_dim=(DECAY_BLOCK,),
        )

    # Stage 4: Y = M @ X (no transpose).
    with ctx.push_context():
        batched_matmul[target="gpu", transpose_b=False](
            Y3, CB3, X3, context=ctx
        )
