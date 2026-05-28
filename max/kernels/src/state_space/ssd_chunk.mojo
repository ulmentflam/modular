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
  Note ``Ldecay[l, l] = exp(0) = 1``.
- Attention scores: ``scores[l, s] = (sum_n C[l, n] * B[s, n]) * Ldecay[l, s]``
  for ``s <= l``.
- Output: ``Y[l, p] = sum_{s=0..l} scores[l, s] * X[s, p]``.

This matches the reference einsum
``Y_diag = einsum("ln,sn,ls,sp->lp", C, B, Ldecay, X)`` restricted to ``s <= l``
(stage 1 of ``ssd_minimal_discrete``).
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
):
    """Compute the SSD intra-chunk diagonal block output on CPU.

    Operates per ``(batch, chunk, head)`` over the supplied row-major tensors.

    Tensor shapes (row-major):
        - ``C``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``B``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``X``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``Y``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        C_layout: Layout of the ``C`` tensor.
        B_layout: Layout of the ``B`` tensor.
        X_layout: Layout of the ``X`` tensor.
        A_layout: Layout of the ``A`` tensor.
        Y_layout: Layout of the ``Y`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        state_dim: State dimension (``N``).
        head_dim: Head dimension (``P``).
        C: Query-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
    """
    # Row-major strides for the flat backing buffers.
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
                # Base offsets for this (batch, chunk, head) slice.
                var cb_base = (
                    b * cb_strides[0] + c * cb_strides[1] + h * cb_strides[2]
                )
                var x_base = (
                    b * x_strides[0] + c * x_strides[1] + h * x_strides[2]
                )
                var a_base = (
                    b * a_strides[0] + c * a_strides[1] + h * a_strides[2]
                )
                var y_base = x_base  # Y has the same shape/strides as X.

                # Inclusive prefix sum of the per-token decay: A_cumsum[l].
                var a_cumsum = List[Float32](length=chunk_len, fill=0.0)
                var running = Float32(0.0)
                for l in range(chunk_len):
                    var a_off = a_base + l * a_strides[3]
                    running += A.ptr[a_off].cast[DType.float32]()
                    a_cumsum[l] = running

                # Y[l, p] = sum_{s<=l} (C[l,:]·B[s,:]) * exp(cum[l]-cum[s]) * X[s,p]
                for l in range(chunk_len):
                    var c_row = cb_base + l * cb_strides[3]
                    var cum_l = a_cumsum[l]
                    for s in range(l + 1):
                        var b_row = cb_base + s * cb_strides[3]

                        # dot(C[l,:], B[s,:]) over the state dimension.
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

                        # Accumulate score * X[s, :] into Y[l, :].
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
# GPU Implementation
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
    C: TileTensor[kernel_dtype, C_LT, MutExternalOrigin],
    B: TileTensor[kernel_dtype, B_LT, MutExternalOrigin],
    X: TileTensor[kernel_dtype, X_LT, MutExternalOrigin],
    A: TileTensor[kernel_dtype, A_LT, MutExternalOrigin],
    Y: TileTensor[kernel_dtype, Y_LT, MutExternalOrigin],
):
    """Compute the SSD intra-chunk diagonal block output on GPU.

    This is a straightforward parallel port of ``ssd_intra_chunk_fwd_cpu``:
    one GPU thread per ``(batch, chunk, head)`` slice. Each thread computes the
    full ``[chunk_len, head_dim]`` output for its slice, reusing the inclusive
    prefix sum of the per-token decay across all output positions. It is not
    tensor-core optimized — correctness and parity with the CPU/reference
    numerics are the goal.

    The same math as the CPU kernel:
        - ``A_cumsum[l] = sum_{k<=l} A[k]``
        - ``Ldecay[l, s] = exp(A_cumsum[l] - A_cumsum[s])`` for ``s <= l``
        - ``scores[l, s] = (sum_n C[l, n] * B[s, n]) * Ldecay[l, s]``
        - ``Y[l, p] = sum_{s<=l} scores[l, s] * X[s, p]``

    Tensor shapes (row-major):
        - ``C``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``B``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``X``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``Y``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        C_LT: Layout of the ``C`` tensor.
        B_LT: Layout of the ``B`` tensor.
        X_LT: Layout of the ``X`` tensor.
        A_LT: Layout of the ``A`` tensor.
        Y_LT: Layout of the ``Y`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        state_dim: State dimension (``N``).
        head_dim: Head dimension (``P``).
        C: Query-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        B: Key-like projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        X: Value-like input, shape ``[batch, n_chunks, n_heads, L, P]``.
        A: Per-token scalar decay, shape ``[batch, n_chunks, n_heads, L]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
    """
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total_slices = batch * n_chunks * n_heads
    if flat_idx >= total_slices:
        return

    # Decompose the flat slice index into (batch, chunk, head).
    var h = flat_idx % n_heads
    var c = (flat_idx // n_heads) % n_chunks
    var b = flat_idx // (n_heads * n_chunks)

    # Row-major strides for the flat backing buffers.
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
    var y_base = x_base  # Y has the same shape/strides as X.

    # Y[l, p] = sum_{s<=l} (C[l,:]·B[s,:]) * exp(cum[l]-cum[s]) * X[s,p]
    # where cum[l] = sum_{k<=l} A[k]. We recompute the inclusive prefix sums
    # cum[l] and cum[s] on the fly to avoid needing per-thread scratch memory.
    for l in range(chunk_len):
        var c_row = cb_base + l * cb_l_stride

        # cum[l] = sum_{k=0..l} A[k].
        var cum_l = Float32(0.0)
        for k in range(l + 1):
            cum_l += A.ptr[a_base + k * a_l_stride].cast[DType.float32]()

        # Initialize this output row to zero before accumulating.
        for p in range(head_dim):
            Y.ptr[y_base + l * x_l_stride + p * x_p_stride] = Scalar[
                kernel_dtype
            ](0)

        var cum_s = Float32(0.0)
        for s in range(l + 1):
            cum_s += A.ptr[a_base + s * a_l_stride].cast[DType.float32]()

            var b_row = cb_base + s * cb_l_stride

            # dot(C[l,:], B[s,:]) over the state dimension.
            var dot = Float32(0.0)
            for n in range(state_dim):
                var cv = C.ptr[c_row + n * cb_n_stride].cast[DType.float32]()
                var bv = B.ptr[b_row + n * cb_n_stride].cast[DType.float32]()
                dot += cv * bv

            var decay = exp(cum_l - cum_s)
            var score = dot * decay

            # Accumulate score * X[s, :] into Y[l, :].
            var x_row = x_base + s * x_l_stride
            var y_row = y_base + l * x_l_stride
            for p in range(head_dim):
                var xv = X.ptr[x_row + p * x_p_stride].cast[DType.float32]()
                var y_off = y_row + p * x_p_stride
                var acc = Y.ptr[y_off].cast[DType.float32]() + score * xv
                Y.ptr[y_off] = acc.cast[kernel_dtype]()
