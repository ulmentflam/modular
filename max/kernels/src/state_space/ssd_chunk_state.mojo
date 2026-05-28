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
"""Structured state space duality (SSD) chunk-state kernel.

This implements stage 2 of the SSD chunk-scan algorithm used by Mamba2: the
"chunk end-state". For each ``(batch, chunk, head)`` it reduces the chunk down
to a single ``[head_dim, state_dim]`` end-state by summing the decay-weighted
outer products of the per-token key-like and value-like projections.

For a chunk of ``L`` tokens with state dim ``N`` and head dim ``P``:

- Inputs: ``B: [L, N]``, ``X: [L, P]``, ``A: [L]`` (a scalar decay per token).
- Cumulative decay: ``A_cumsum[l] = sum_{k=0..l} A[k]``.
- Per-token decay toward the chunk end:
  ``decay[l] = exp(A_cumsum[L-1] - A_cumsum[l])``. Note ``decay[L-1] = 1``.
- End-state: ``state[p, n] = sum_{l=0..L-1} B[l, n] * decay[l] * X[l, p]``.

This matches the reference einsum
``state = einsum("ln,l,lp->pn", B, decay, X)`` (stage 2 of
``ssd_minimal_discrete``), producing one end-state per ``(batch, chunk, head)``.
"""

from layout import Layout, LayoutTensor, TensorLayout, TileTensor
from std.math import exp
from std.gpu import (
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from std.utils.index import IndexList

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


def ssd_chunk_state_fwd_cpu[
    kernel_dtype: DType,
    B_layout: Layout,
    X_layout: Layout,
    A_layout: Layout,
    state_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    B: LayoutTensor[kernel_dtype, B_layout, MutAnyOrigin],
    X: LayoutTensor[kernel_dtype, X_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, A_layout, MutAnyOrigin],
    state: LayoutTensor[kernel_dtype, state_layout, MutAnyOrigin],
):
    """Compute the SSD chunk end-state on CPU.

    Operates per ``(batch, chunk, head)`` over the supplied row-major tensors,
    reducing each chunk down to a single ``[head_dim, state_dim]`` end-state.

    Tensor shapes (row-major):
        - ``B``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``X``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        B_layout: Layout of the ``B`` tensor.
        X_layout: Layout of the ``X`` tensor.
        A_layout: Layout of the ``A`` tensor.
        state_layout: Layout of the ``state`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        state_dim: State dimension (``N``).
        head_dim: Head dimension (``P``).
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        state: Output end-state, shape ``[batch, n_chunks, n_heads, P, N]``.
    """
    # Row-major strides for the flat backing buffers.
    var b_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, state_dim
    )
    var x_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, head_dim
    )
    var a_strides = _row_major_strides_4d(batch, n_chunks, n_heads, chunk_len)
    var state_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, head_dim, state_dim
    )

    for bi in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                # Base offsets for this (batch, chunk, head) slice.
                var b_base = (
                    bi * b_strides[0] + c * b_strides[1] + h * b_strides[2]
                )
                var x_base = (
                    bi * x_strides[0] + c * x_strides[1] + h * x_strides[2]
                )
                var a_base = (
                    bi * a_strides[0] + c * a_strides[1] + h * a_strides[2]
                )
                var state_base = (
                    bi * state_strides[0]
                    + c * state_strides[1]
                    + h * state_strides[2]
                )

                # Inclusive prefix sum of the per-token decay: A_cumsum[l].
                var a_cumsum = List[Float32](length=chunk_len, fill=0.0)
                var running = Float32(0.0)
                for l in range(chunk_len):
                    var a_off = a_base + l * a_strides[3]
                    running += A.ptr[a_off].cast[DType.float32]()
                    a_cumsum[l] = running

                # decay[l] = exp(A_cumsum[L-1] - A_cumsum[l]); decay[L-1] = 1.
                var cum_end = a_cumsum[chunk_len - 1]
                var decay = List[Float32](length=chunk_len, fill=0.0)
                for l in range(chunk_len):
                    decay[l] = exp(cum_end - a_cumsum[l])

                # state[p, n] = sum_l B[l, n] * decay[l] * X[l, p]
                for p in range(head_dim):
                    for n in range(state_dim):
                        var acc = Float32(0.0)
                        for l in range(chunk_len):
                            var bv = B.ptr[
                                b_base + l * b_strides[3] + n * b_strides[4]
                            ].cast[DType.float32]()
                            var xv = X.ptr[
                                x_base + l * x_strides[3] + p * x_strides[4]
                            ].cast[DType.float32]()
                            acc += bv * decay[l] * xv
                        var state_off = (
                            state_base
                            + p * state_strides[3]
                            + n * state_strides[4]
                        )
                        state.ptr[state_off] = acc.cast[kernel_dtype]()


# ===----------------------------------------------------------------------=== #
# GPU Implementation
# ===----------------------------------------------------------------------=== #


def ssd_chunk_state_fwd_gpu[
    kernel_dtype: DType,
    B_LT: TensorLayout,
    X_LT: TensorLayout,
    A_LT: TensorLayout,
    state_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    B: TileTensor[kernel_dtype, B_LT, MutExternalOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutExternalOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutExternalOrigin],
    state: TileTensor[kernel_dtype, state_LT, MutExternalOrigin],
):
    """Compute the SSD chunk end-state on GPU.

    This is a straightforward parallel port of ``ssd_chunk_state_fwd_cpu``: one
    GPU thread per ``(batch, chunk, head)`` slice. Each thread reduces the full
    ``[chunk_len, ...]`` chunk to its ``[head_dim, state_dim]`` end-state. It is
    not tensor-core optimized — correctness and parity with the CPU/reference
    numerics are the goal.

    The same math as the CPU kernel:
        - ``A_cumsum[l] = sum_{k<=l} A[k]``
        - ``decay[l] = exp(A_cumsum[L-1] - A_cumsum[l])``
        - ``state[p, n] = sum_l B[l, n] * decay[l] * X[l, p]``

    Tensor shapes (row-major):
        - ``B``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``X``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        B_LT: Layout of the ``B`` tensor.
        X_LT: Layout of the ``X`` tensor.
        A_LT: Layout of the ``A`` tensor.
        state_LT: Layout of the ``state`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        state_dim: State dimension (``N``).
        head_dim: Head dimension (``P``).
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        state: Output end-state, shape ``[batch, n_chunks, n_heads, P, N]``.
    """
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total_slices = batch * n_chunks * n_heads
    if flat_idx >= total_slices:
        return

    # Decompose the flat slice index into (batch, chunk, head).
    var h = flat_idx % n_heads
    var c = (flat_idx // n_heads) % n_chunks
    var bi = flat_idx // (n_heads * n_chunks)

    # Row-major strides for the flat backing buffers.
    var b_n_stride = 1
    var b_l_stride = state_dim
    var b_h_stride = chunk_len * b_l_stride
    var b_c_stride = n_heads * b_h_stride
    var b_b_stride = n_chunks * b_c_stride

    var x_p_stride = 1
    var x_l_stride = head_dim
    var x_h_stride = chunk_len * x_l_stride
    var x_c_stride = n_heads * x_h_stride
    var x_b_stride = n_chunks * x_c_stride

    var a_l_stride = 1
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    var state_n_stride = 1
    var state_p_stride = state_dim
    var state_h_stride = head_dim * state_p_stride
    var state_c_stride = n_heads * state_h_stride
    var state_b_stride = n_chunks * state_c_stride

    var b_base = bi * b_b_stride + c * b_c_stride + h * b_h_stride
    var x_base = bi * x_b_stride + c * x_c_stride + h * x_h_stride
    var a_base = bi * a_b_stride + c * a_c_stride + h * a_h_stride
    var state_base = (
        bi * state_b_stride + c * state_c_stride + h * state_h_stride
    )

    # cum_end = A_cumsum[L-1] = sum_{k=0..L-1} A[k].
    var cum_end = Float32(0.0)
    for k in range(chunk_len):
        cum_end += A.ptr[a_base + k * a_l_stride].cast[DType.float32]()

    # state[p, n] = sum_l B[l, n] * exp(cum_end - cum[l]) * X[l, p]
    # where cum[l] = sum_{k<=l} A[k]. We recompute the inclusive prefix sum
    # cum[l] on the fly to avoid needing per-thread scratch memory.
    for p in range(head_dim):
        for n in range(state_dim):
            var acc = Float32(0.0)
            var cum_l = Float32(0.0)
            for l in range(chunk_len):
                cum_l += A.ptr[a_base + l * a_l_stride].cast[DType.float32]()
                var bv = B.ptr[b_base + l * b_l_stride + n * b_n_stride].cast[
                    DType.float32
                ]()
                var xv = X.ptr[x_base + l * x_l_stride + p * x_p_stride].cast[
                    DType.float32
                ]()
                var decay = exp(cum_end - cum_l)
                acc += bv * decay * xv
            var state_off = state_base + p * state_p_stride + n * state_n_stride
            state.ptr[state_off] = acc.cast[kernel_dtype]()
