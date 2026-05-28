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
"""Structured state space duality (SSD) output-recombination kernel.

This implements stage 4 (the final "output recombination") of the SSD
chunk-scan algorithm used by Mamba2. After stage 3 (``ssd_chunk_scan``) has
threaded the inter-chunk recurrence to produce the state *entering* each chunk,
this stage combines that inter-chunk contribution with the per-chunk diagonal
output to form the final per-token output.

For a single ``(batch, chunk, head)`` slice with chunk length ``L``, state dim
``N`` and head dim ``P``:

- Inputs:
  - ``C: [b, nc, h, L, N]`` (the per-token C projection).
  - ``entering_state: [b, nc, h, P, N]`` (state entering this chunk, the
    stage-3 output).
  - ``A: [b, nc, h, L]`` (per-token log-decay within the chunk).
  - ``Y_diag: [b, nc, h, L, P]`` (the intra-chunk diagonal output, from
    stage 1).
- Compute (per token ``l`` and head-dim index ``p``):
  ```
  A_cumsum[l]       = sum_{k <= l} A[k]           # inclusive prefix sum
  state_decay_out[l]= exp(A_cumsum[l])
  Y_off[l, p]       = state_decay_out[l] * sum_n C[l, n] * entering_state[p, n]
  Y[l, p]           = Y_diag[l, p] + Y_off[l, p]
  ```
- Output:
  - ``Y: [b, nc, h, L, P]``.

The computation is fully element-wise parallel over ``(b, nc, h, l, p)``: one
GPU thread owns a single output scalar ``Y[b, nc, h, l, p]``. The only sequential
dependency is the per-``(b, nc, h, l)`` inclusive prefix sum ``A_cumsum[l]``,
which each thread recomputes up to its own ``l`` (``l`` is at most the chunk
length, so this is cheap and avoids any cross-thread synchronization).
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


def ssd_output_recombination_fwd_cpu[
    kernel_dtype: DType,
    c_layout: Layout,
    entering_layout: Layout,
    a_layout: Layout,
    y_diag_layout: Layout,
    y_layout: Layout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    C: LayoutTensor[kernel_dtype, c_layout, MutAnyOrigin],
    entering_state: LayoutTensor[kernel_dtype, entering_layout, MutAnyOrigin],
    A: LayoutTensor[kernel_dtype, a_layout, MutAnyOrigin],
    Y_diag: LayoutTensor[kernel_dtype, y_diag_layout, MutAnyOrigin],
    Y: LayoutTensor[kernel_dtype, y_layout, MutAnyOrigin],
):
    """Compute the SSD output recombination (stage 4) on CPU.

    Combines the intra-chunk diagonal output ``Y_diag`` with the inter-chunk
    contribution (``C`` projected against the entering state, decayed by the
    in-chunk cumulative decay) to form the final per-token output ``Y``.

    Tensor shapes (row-major):
        - ``C``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``entering_state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``Y_diag``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``Y``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        c_layout: Layout of the ``C`` tensor.
        entering_layout: Layout of the ``entering_state`` tensor.
        a_layout: Layout of the ``A`` tensor.
        y_diag_layout: Layout of the ``Y_diag`` tensor.
        y_layout: Layout of the ``Y`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        head_dim: Head dimension (``P``).
        state_dim: State dimension (``N``).
        C: Per-token C projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        entering_state: State entering this chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        A: Per-token in-chunk log-decay,
            shape ``[batch, n_chunks, n_heads, L]``.
        Y_diag: Intra-chunk diagonal output,
            shape ``[batch, n_chunks, n_heads, L, P]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
    """
    # Row-major strides for the flat backing buffers.
    var c_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, state_dim
    )
    var ent_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, head_dim, state_dim
    )
    var a_strides = _row_major_strides_4d(batch, n_chunks, n_heads, chunk_len)
    var y_strides = _row_major_strides_5d(
        batch, n_chunks, n_heads, chunk_len, head_dim
    )

    for bi in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_base = (
                    bi * a_strides[0] + c * a_strides[1] + h * a_strides[2]
                )
                var c_base = (
                    bi * c_strides[0] + c * c_strides[1] + h * c_strides[2]
                )
                var ent_base = (
                    bi * ent_strides[0]
                    + c * ent_strides[1]
                    + h * ent_strides[2]
                )
                var y_base = (
                    bi * y_strides[0] + c * y_strides[1] + h * y_strides[2]
                )

                # Inclusive prefix sum of A over the chunk -> decay per token.
                var a_cumsum = Float32(0.0)
                for l in range(chunk_len):
                    var a_off = a_base + l * a_strides[3]
                    a_cumsum += A.ptr[a_off].cast[DType.float32]()
                    var decay = exp(a_cumsum)

                    for p in range(head_dim):
                        # Y_off[l, p] = decay * sum_n C[l, n] * entering[p, n].
                        var acc = Float32(0.0)
                        for n in range(state_dim):
                            var c_off = (
                                c_base + l * c_strides[3] + n * c_strides[4]
                            )
                            var ent_off = (
                                ent_base
                                + p * ent_strides[3]
                                + n * ent_strides[4]
                            )
                            var c_val = C.ptr[c_off].cast[DType.float32]()
                            var ent_val = entering_state.ptr[ent_off].cast[
                                DType.float32
                            ]()
                            acc += c_val * ent_val

                        var y_off = decay * acc

                        # Y[l, p] = Y_diag[l, p] + Y_off[l, p].
                        var yd_off = (
                            y_base + l * y_strides[3] + p * y_strides[4]
                        )
                        var y_diag_val = Y_diag.ptr[yd_off].cast[
                            DType.float32
                        ]()
                        Y.ptr[yd_off] = (y_diag_val + y_off).cast[
                            kernel_dtype
                        ]()


# ===----------------------------------------------------------------------=== #
# GPU Implementation
# ===----------------------------------------------------------------------=== #


def ssd_output_recombination_fwd_gpu[
    kernel_dtype: DType,
    c_LT: TensorLayout,
    entering_LT: TensorLayout,
    a_LT: TensorLayout,
    y_diag_LT: TensorLayout,
    y_LT: TensorLayout,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
    C: TileTensor[kernel_dtype, c_LT, MutExternalOrigin],
    entering_state: TileTensor[kernel_dtype, entering_LT, MutExternalOrigin],
    A: TileTensor[kernel_dtype, a_LT, MutExternalOrigin],
    Y_diag: TileTensor[kernel_dtype, y_diag_LT, MutExternalOrigin],
    Y: TileTensor[kernel_dtype, y_LT, MutExternalOrigin],
):
    """Compute the SSD output recombination (stage 4) on GPU.

    The computation is element-wise parallel: each GPU thread owns a single
    output scalar ``Y[bi, c, h, l, p]``. The only sequential dependency is the
    per-``(bi, c, h, l)`` inclusive prefix sum ``A_cumsum[l]``, which each thread
    recomputes up to its own ``l`` (cheap, and avoids cross-thread sync).

    The same math as the CPU kernel, for the scalar ``Y[l, p]``:
        - ``A_cumsum[l] = sum_{k <= l} A[k]``
        - ``decay = exp(A_cumsum[l])``
        - ``Y_off[l, p] = decay * sum_n C[l, n] * entering_state[p, n]``
        - ``Y[l, p] = Y_diag[l, p] + Y_off[l, p]``

    Tensor shapes (row-major):
        - ``C``: ``[batch, n_chunks, n_heads, chunk_len, state_dim]``
        - ``entering_state``: ``[batch, n_chunks, n_heads, head_dim, state_dim]``
        - ``A``: ``[batch, n_chunks, n_heads, chunk_len]``
        - ``Y_diag``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]``
        - ``Y``: ``[batch, n_chunks, n_heads, chunk_len, head_dim]`` (output)

    Parameters:
        kernel_dtype: The element type of every tensor (e.g. ``float32``).
        c_LT: Layout of the ``C`` tensor.
        entering_LT: Layout of the ``entering_state`` tensor.
        a_LT: Layout of the ``A`` tensor.
        y_diag_LT: Layout of the ``Y_diag`` tensor.
        y_LT: Layout of the ``Y`` output tensor.

    Args:
        batch: Number of batch elements.
        n_chunks: Number of chunks per sequence.
        n_heads: Number of heads.
        chunk_len: Tokens per chunk (``L``).
        head_dim: Head dimension (``P``).
        state_dim: State dimension (``N``).
        C: Per-token C projection, shape ``[batch, n_chunks, n_heads, L, N]``.
        entering_state: State entering this chunk,
            shape ``[batch, n_chunks, n_heads, P, N]``.
        A: Per-token in-chunk log-decay,
            shape ``[batch, n_chunks, n_heads, L]``.
        Y_diag: Intra-chunk diagonal output,
            shape ``[batch, n_chunks, n_heads, L, P]``.
        Y: Output, shape ``[batch, n_chunks, n_heads, L, P]``.
    """
    var flat_idx = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var total_threads = batch * n_chunks * n_heads * chunk_len * head_dim
    if flat_idx >= total_threads:
        return

    # Decompose the flat thread index into (batch, chunk, head, l, p).
    var p = flat_idx % head_dim
    var l = (flat_idx // head_dim) % chunk_len
    var h = (flat_idx // (head_dim * chunk_len)) % n_heads
    var c = (flat_idx // (head_dim * chunk_len * n_heads)) % n_chunks
    var bi = flat_idx // (head_dim * chunk_len * n_heads * n_chunks)

    # Row-major strides for the flat backing buffers.
    # C: [batch, n_chunks, n_heads, chunk_len, state_dim].
    var c_n_stride = 1
    var c_l_stride = state_dim
    var c_h_stride = chunk_len * c_l_stride
    var c_c_stride = n_heads * c_h_stride
    var c_b_stride = n_chunks * c_c_stride

    # entering_state: [batch, n_chunks, n_heads, head_dim, state_dim].
    var ent_n_stride = 1
    var ent_p_stride = state_dim
    var ent_h_stride = head_dim * ent_p_stride
    var ent_c_stride = n_heads * ent_h_stride
    var ent_b_stride = n_chunks * ent_c_stride

    # A: [batch, n_chunks, n_heads, chunk_len].
    var a_l_stride = 1
    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    # Y / Y_diag: [batch, n_chunks, n_heads, chunk_len, head_dim].
    var y_p_stride = 1
    var y_l_stride = head_dim
    var y_h_stride = chunk_len * y_l_stride
    var y_c_stride = n_heads * y_h_stride
    var y_b_stride = n_chunks * y_c_stride

    # Inclusive prefix sum of A over [0, l] for this (bi, c, h).
    var a_base = bi * a_b_stride + c * a_c_stride + h * a_h_stride
    var a_cumsum = Float32(0.0)
    for k in range(l + 1):
        a_cumsum += A.ptr[a_base + k * a_l_stride].cast[DType.float32]()
    var decay = exp(a_cumsum)

    # Y_off[l, p] = decay * sum_n C[l, n] * entering_state[p, n].
    var c_base = (
        bi * c_b_stride + c * c_c_stride + h * c_h_stride + l * c_l_stride
    )
    var ent_base = (
        bi * ent_b_stride
        + c * ent_c_stride
        + h * ent_h_stride
        + p * ent_p_stride
    )
    var acc = Float32(0.0)
    for n in range(state_dim):
        var c_val = C.ptr[c_base + n * c_n_stride].cast[DType.float32]()
        var ent_val = entering_state.ptr[ent_base + n * ent_n_stride].cast[
            DType.float32
        ]()
        acc += c_val * ent_val
    var y_off = decay * acc

    # Y[l, p] = Y_diag[l, p] + Y_off[l, p].
    var y_off_flat = (
        bi * y_b_stride
        + c * y_c_stride
        + h * y_h_stride
        + l * y_l_stride
        + p * y_p_stride
    )
    var y_diag_val = Y_diag.ptr[y_off_flat].cast[DType.float32]()
    Y.ptr[y_off_flat] = (y_diag_val + y_off).cast[kernel_dtype]()
