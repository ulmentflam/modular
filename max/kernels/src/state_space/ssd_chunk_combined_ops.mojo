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
"""Fused SSD chunk-scan operation registration for Mamba2.

This module registers ``ssd_chunk_scan_combined``, which stitches the four
verified SSD stage kernels — intra-chunk diagonal, chunk end-state, inter-chunk
recurrence, and output recombination — into a single MAX graph op matching the
torch ``mamba_chunk_scan_combined`` contract:

    op(x, dt, A, B, C, chunk_size)
        == ssd_minimal_discrete(x * dt[..., None], A_h * dt, B, C, chunk_size)

with all stage 1-4 work happening behind the registration boundary.

Inputs (row-major):
    - ``x``: ``[batch, seqlen, n_heads, head_dim]`` — value-like input ``X``.
    - ``dt``: ``[batch, seqlen, n_heads]`` — per-token time delta.
    - ``A``: ``[n_heads]`` — scalar log-decay per head.
    - ``B``: ``[batch, seqlen, n_heads, state_dim]``.
    - ``C``: ``[batch, seqlen, n_heads, state_dim]``.

Outputs:
    - ``Y: [batch, seqlen, n_heads, head_dim]`` (same dtype as inputs).
    - ``final_state: [batch, n_heads, head_dim, state_dim]`` — the SSM state
      after the last chunk, used to seed the decode (``selective_scan_update``)
      cache so step-mode continuation picks up where prefill left off.

The op currently assumes ``ngroups == n_heads`` (no group broadcasting yet) and
that ``seqlen`` divides evenly by ``chunk_size``; both restrictions match the
upstream Mamba2 reference at the dimensions we care about for prefill.
"""

import extensibility as compiler
from extensibility import InputTensor, OutputTensor
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.utils.index import Index, IndexList

from layout import Layout, LayoutTensor, RuntimeLayout

from state_space.ssd_chunk import ssd_intra_chunk_fwd_cpu
from state_space.ssd_chunk_state import ssd_chunk_state_fwd_cpu
from state_space.ssd_chunk_scan import ssd_chunk_scan_fwd_cpu
from state_space.ssd_chunk_combine import ssd_output_recombination_fwd_cpu


# ===----------------------------------------------------------------------=== #
# Pre-stage helpers (CPU): discretization + reshape
# ===----------------------------------------------------------------------=== #


def _ssd_combined_precompute_cpu[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    n_chunks: Int,
    chunk_size: Int,
    x_ptr: UnsafePointer[Scalar[dtype], ...],
    dt_ptr: UnsafePointer[Scalar[dtype], ...],
    A_ptr: UnsafePointer[Scalar[dtype], ...],
    B_ptr: UnsafePointer[Scalar[dtype], ...],
    C_ptr: UnsafePointer[Scalar[dtype], ...],
    X_disc_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    A_disc_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    B_chunk_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    C_chunk_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    chunk_decays_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
):
    """Reshape inputs into chunked form and discretize on CPU.

    Inputs:
        x:  ``[batch, seqlen, n_heads, head_dim]``
        dt: ``[batch, seqlen, n_heads]``
        A:  ``[n_heads]`` (scalar decay per head)
        B:  ``[batch, seqlen, n_heads, state_dim]``
        C:  ``[batch, seqlen, n_heads, state_dim]``

    Produces:
        X_disc:       ``[batch, n_chunks, n_heads, chunk_size, head_dim]``
                      with ``X_disc[b, c, h, l, p] = x[b, c*L+l, h, p] *
                      dt[b, c*L+l, h]``.
        A_disc:       ``[batch, n_chunks, n_heads, chunk_size]`` with
                      ``A_disc[b, c, h, l] = A[h] * dt[b, c*L+l, h]``.
        B_chunk:      reshape of B into chunked form (no value change).
        C_chunk:      reshape of C into chunked form (no value change).
        chunk_decays: ``[batch, n_chunks, n_heads]``,
                      ``= sum over l of A_disc[b, c, h, l]``.
    """
    # Note: all input/output buffers are row-major contiguous.
    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var a_h = A_ptr[h].cast[DType.float32]()
                var chunk_sum = Float32(0.0)
                for l in range(chunk_size):
                    var t = c * chunk_size + l
                    var dt_off = (b * seqlen + t) * n_heads + h
                    var dt_val = dt_ptr[dt_off].cast[DType.float32]()
                    var a_disc_val = a_h * dt_val
                    chunk_sum += a_disc_val

                    var a_disc_off = (
                        (b * n_chunks + c) * n_heads + h
                    ) * chunk_size + l
                    A_disc_ptr[a_disc_off] = a_disc_val.cast[dtype]()

                    # X_disc[b, c, h, l, p] = x[b, t, h, p] * dt[b, t, h]
                    var x_in_base = ((b * seqlen + t) * n_heads + h) * head_dim
                    var x_out_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * head_dim
                    for p in range(head_dim):
                        var x_val = x_ptr[x_in_base + p].cast[DType.float32]()
                        X_disc_ptr[x_out_base + p] = (x_val * dt_val).cast[
                            dtype
                        ]()

                    # B_chunk / C_chunk: pure reshape (same value).
                    var bc_in_base = (
                        (b * seqlen + t) * n_heads + h
                    ) * state_dim
                    var bc_out_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * state_dim
                    for n in range(state_dim):
                        B_chunk_ptr[bc_out_base + n] = B_ptr[bc_in_base + n]
                        C_chunk_ptr[bc_out_base + n] = C_ptr[bc_in_base + n]

                var cd_off = (b * n_chunks + c) * n_heads + h
                chunk_decays_ptr[cd_off] = chunk_sum.cast[dtype]()


def _ssd_combined_postprocess_cpu[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    n_chunks: Int,
    chunk_size: Int,
    Y_chunked_ptr: UnsafePointer[Scalar[dtype], ...],
    Y_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
):
    """Flatten ``Y_chunked: [b, nc, h, L, P]`` back into ``Y: [b, S, h, P]``."""
    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                for l in range(chunk_size):
                    var t = c * chunk_size + l
                    var y_in_base = (
                        ((b * n_chunks + c) * n_heads + h) * chunk_size + l
                    ) * head_dim
                    var y_out_base = (b * seqlen + t) * n_heads * head_dim + (
                        h * head_dim
                    )
                    for p in range(head_dim):
                        Y_ptr[y_out_base + p] = Y_chunked_ptr[y_in_base + p]


# ===----------------------------------------------------------------------=== #
# CPU dispatch — runs all four stages
# ===----------------------------------------------------------------------=== #


def _ssd_chunk_scan_combined_cpu[
    dtype: DType,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    chunk_size: Int,
    x_ptr: UnsafePointer[Scalar[dtype], ...],
    dt_ptr: UnsafePointer[Scalar[dtype], ...],
    A_ptr: UnsafePointer[Scalar[dtype], ...],
    B_ptr: UnsafePointer[Scalar[dtype], ...],
    C_ptr: UnsafePointer[Scalar[dtype], ...],
    Y_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
    final_state_ptr: UnsafePointer[mut=True, Scalar[dtype], ...],
) raises:
    """Run the full SSD chunk-scan-combined on CPU.

    ``final_state_ptr`` is filled with the post-last-chunk SSM state, shape
    ``[batch, n_heads, head_dim, state_dim]`` row-major. Callers use it to seed
    the step-mode SSM cache; pass a dummy buffer of the right size if the
    state is not needed.
    """
    if seqlen % chunk_size != 0:
        raise Error(
            "ssd_chunk_scan_combined: seqlen must be divisible by chunk_size"
        )
    var n_chunks = seqlen // chunk_size

    # Heap allocations for intermediates. Sized once and zeroed where needed.
    var x_disc_size = batch * n_chunks * n_heads * chunk_size * head_dim
    var a_disc_size = batch * n_chunks * n_heads * chunk_size
    var bc_chunk_size = batch * n_chunks * n_heads * chunk_size * state_dim
    var y_diag_size = x_disc_size  # same shape as X_disc
    var chunk_states_size = batch * n_chunks * n_heads * head_dim * state_dim
    var entering_size = chunk_states_size
    var final_size = batch * n_heads * head_dim * state_dim
    var cd_size = batch * n_chunks * n_heads
    var y_chunked_size = x_disc_size

    var X_disc = List[Scalar[dtype]](length=x_disc_size, fill=Scalar[dtype](0))
    var A_disc = List[Scalar[dtype]](length=a_disc_size, fill=Scalar[dtype](0))
    var B_chunk = List[Scalar[dtype]](
        length=bc_chunk_size, fill=Scalar[dtype](0)
    )
    var C_chunk = List[Scalar[dtype]](
        length=bc_chunk_size, fill=Scalar[dtype](0)
    )
    var Y_diag = List[Scalar[dtype]](length=y_diag_size, fill=Scalar[dtype](0))
    var chunk_states = List[Scalar[dtype]](
        length=chunk_states_size, fill=Scalar[dtype](0)
    )
    var entering = List[Scalar[dtype]](
        length=entering_size, fill=Scalar[dtype](0)
    )
    var final = List[Scalar[dtype]](length=final_size, fill=Scalar[dtype](0))
    var chunk_decays = List[Scalar[dtype]](
        length=cd_size, fill=Scalar[dtype](0)
    )
    var Y_chunked = List[Scalar[dtype]](
        length=y_chunked_size, fill=Scalar[dtype](0)
    )

    # Stage 0: discretize + chunked reshape.
    _ssd_combined_precompute_cpu[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        n_chunks,
        chunk_size,
        x_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        X_disc.unsafe_ptr(),
        A_disc.unsafe_ptr(),
        B_chunk.unsafe_ptr(),
        C_chunk.unsafe_ptr(),
        chunk_decays.unsafe_ptr(),
    )

    # Build LayoutTensors over the intermediate buffers (row-major).
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var X_disc_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_disc.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, head_dim)
        ),
    )
    var A_disc_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_disc.unsafe_ptr(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size)
        ),
    )
    var B_chunk_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_chunk.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, state_dim)
        ),
    )
    var C_chunk_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        C_chunk.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, state_dim)
        ),
    )
    var Y_diag_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_diag.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, head_dim)
        ),
    )
    var chunk_states_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        chunk_states.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    var entering_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        entering.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, head_dim, state_dim)
        ),
    )
    var final_lt = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        final.unsafe_ptr(),
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_heads, head_dim, state_dim)
        ),
    )
    var chunk_decays_lt = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        chunk_decays.unsafe_ptr(),
        RuntimeLayout[layout_3d].row_major(Index(batch, n_chunks, n_heads)),
    )
    var Y_chunked_lt = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_chunked.unsafe_ptr(),
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_size, head_dim)
        ),
    )

    # Stage 1: intra-chunk diagonal -> Y_diag.
    # The kernel accumulates into Y, so Y_diag is pre-zeroed above.
    ssd_intra_chunk_fwd_cpu[
        dtype,
        C_chunk_lt.layout,
        B_chunk_lt.layout,
        X_disc_lt.layout,
        A_disc_lt.layout,
        Y_diag_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        state_dim,
        head_dim,
        C_chunk_lt,
        B_chunk_lt,
        X_disc_lt,
        A_disc_lt,
        Y_diag_lt,
    )

    # Stage 2: per-chunk end-states.
    ssd_chunk_state_fwd_cpu[
        dtype,
        B_chunk_lt.layout,
        X_disc_lt.layout,
        A_disc_lt.layout,
        chunk_states_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        state_dim,
        head_dim,
        B_chunk_lt,
        X_disc_lt,
        A_disc_lt,
        chunk_states_lt,
    )

    # Stage 3: inter-chunk recurrence.
    ssd_chunk_scan_fwd_cpu[
        dtype,
        chunk_states_lt.layout,
        chunk_decays_lt.layout,
        entering_lt.layout,
        final_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        head_dim,
        state_dim,
        chunk_states_lt,
        chunk_decays_lt,
        entering_lt,
        final_lt,
    )

    # Stage 4: output recombination (writes into Y_chunked, copying Y_diag in).
    ssd_output_recombination_fwd_cpu[
        dtype,
        C_chunk_lt.layout,
        entering_lt.layout,
        A_disc_lt.layout,
        Y_diag_lt.layout,
        Y_chunked_lt.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_size,
        head_dim,
        state_dim,
        C_chunk_lt,
        entering_lt,
        A_disc_lt,
        Y_diag_lt,
        Y_chunked_lt,
    )

    # Flatten the chunked output back to (batch, seqlen, n_heads, head_dim).
    _ssd_combined_postprocess_cpu[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        n_chunks,
        chunk_size,
        Y_chunked.unsafe_ptr(),
        Y_ptr,
    )

    # Copy the post-last-chunk SSM state into the caller's final_state buffer.
    # Shape is [batch, n_heads, head_dim, state_dim], identical to `final_lt`.
    var final_total = batch * n_heads * head_dim * state_dim
    var final_src = final.unsafe_ptr()
    for i in range(final_total):
        final_state_ptr[i] = final_src[i]


# ===----------------------------------------------------------------------=== #
# @compiler.register entry point
# ===----------------------------------------------------------------------=== #


@compiler.register("ssd_chunk_scan_combined")
struct SSDChunkScanCombined[chunk_size: Int = 256]:
    """Fused Mamba2 SSD chunk-scan combined op.

    Stitches the four verified SSD stage kernels into a single op matching the
    torch ``mamba_chunk_scan_combined`` fused contract.

    Parameters:
        chunk_size: Tokens per chunk. Must divide ``seqlen`` evenly. The
            default (256) matches the upstream Mamba2 prefill setting.

    Tensor Shapes:
        - output Y:           ``(batch, seqlen, n_heads, head_dim)``
        - output final_state: ``(batch, n_heads, head_dim, state_dim)``
        - x:                  ``(batch, seqlen, n_heads, head_dim)``
        - dt:                 ``(batch, seqlen, n_heads)``
        - A:                  ``(n_heads,)``
        - B:                  ``(batch, seqlen, n_heads, state_dim)``
        - C:                  ``(batch, seqlen, n_heads, state_dim)``
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        Y: OutputTensor[dtype=dtype, rank=4, ...],
        final_state: OutputTensor[dtype=dtype, rank=4, ...],
        x: InputTensor[dtype=dtype, rank=4, ...],
        dt: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if Y.shape() != x.shape():
            raise Error("ssd_chunk_scan_combined: Y shape must match x shape")

        var batch = x.dim_size(0)
        var seqlen = x.dim_size(1)
        var n_heads = x.dim_size(2)
        var head_dim = x.dim_size(3)
        var state_dim = B.dim_size(3)

        if (
            final_state.dim_size(0) != batch
            or final_state.dim_size(1) != n_heads
            or final_state.dim_size(2) != head_dim
            or final_state.dim_size(3) != state_dim
        ):
            raise Error(
                "ssd_chunk_scan_combined: final_state shape must be"
                " [batch, n_heads, head_dim, state_dim]"
            )

        comptime if is_cpu[target]():
            _ssd_chunk_scan_combined_cpu[dtype](
                batch,
                seqlen,
                n_heads,
                head_dim,
                state_dim,
                Self.chunk_size,
                x.unsafe_ptr(),
                dt.unsafe_ptr(),
                A.unsafe_ptr(),
                B.unsafe_ptr(),
                C.unsafe_ptr(),
                Y.unsafe_ptr(),
                final_state.unsafe_ptr(),
            )
        elif is_gpu[target]():
            # GPU path: stage through pinned host memory. Copy device inputs
            # to the host, run the CPU dispatch (which already passes parity
            # vs the torch reference), then copy outputs back to device. The
            # four device-resident GPU stage kernels are tested individually
            # in `test/gpu/state_space/`; a fully on-device stitched dispatch
            # is tracked as a separate follow-up. This staging path keeps
            # end-to-end correctness while we close the perf gap.
            var x_count = batch * seqlen * n_heads * head_dim
            var dt_count = batch * seqlen * n_heads
            var A_count = n_heads
            var bc_count = batch * seqlen * n_heads * state_dim
            var y_count = x_count
            var final_count = batch * n_heads * head_dim * state_dim

            comptime layout_1d = Layout.row_major[1]()

            var x_buf = ctx.enqueue_create_host_buffer[dtype](x_count)
            var x_lt = LayoutTensor[dtype, layout_1d, _](
                x_buf, RuntimeLayout[layout_1d].row_major(Index(x_count))
            )
            var dt_buf = ctx.enqueue_create_host_buffer[dtype](dt_count)
            var dt_lt = LayoutTensor[dtype, layout_1d, _](
                dt_buf, RuntimeLayout[layout_1d].row_major(Index(dt_count))
            )
            var A_buf = ctx.enqueue_create_host_buffer[dtype](A_count)
            var A_lt = LayoutTensor[dtype, layout_1d, _](
                A_buf, RuntimeLayout[layout_1d].row_major(Index(A_count))
            )
            var B_buf = ctx.enqueue_create_host_buffer[dtype](bc_count)
            var B_lt = LayoutTensor[dtype, layout_1d, _](
                B_buf, RuntimeLayout[layout_1d].row_major(Index(bc_count))
            )
            var C_buf = ctx.enqueue_create_host_buffer[dtype](bc_count)
            var C_lt = LayoutTensor[dtype, layout_1d, _](
                C_buf, RuntimeLayout[layout_1d].row_major(Index(bc_count))
            )
            var Y_buf = ctx.enqueue_create_host_buffer[dtype](y_count)
            var Y_lt = LayoutTensor[dtype, layout_1d, _](
                Y_buf, RuntimeLayout[layout_1d].row_major(Index(y_count))
            )
            var final_buf = ctx.enqueue_create_host_buffer[dtype](final_count)
            var final_lt = LayoutTensor[dtype, layout_1d, _](
                final_buf,
                RuntimeLayout[layout_1d].row_major(Index(final_count)),
            )

            with ctx.push_context():
                ctx.enqueue_copy(x_lt.ptr, x.unsafe_ptr(), x_count)
                ctx.enqueue_copy(dt_lt.ptr, dt.unsafe_ptr(), dt_count)
                ctx.enqueue_copy(A_lt.ptr, A.unsafe_ptr(), A_count)
                ctx.enqueue_copy(B_lt.ptr, B.unsafe_ptr(), bc_count)
                ctx.enqueue_copy(C_lt.ptr, C.unsafe_ptr(), bc_count)
            ctx.synchronize()

            _ssd_chunk_scan_combined_cpu[dtype](
                batch,
                seqlen,
                n_heads,
                head_dim,
                state_dim,
                Self.chunk_size,
                x_lt.ptr,
                dt_lt.ptr,
                A_lt.ptr,
                B_lt.ptr,
                C_lt.ptr,
                Y_lt.ptr,
                final_lt.ptr,
            )

            with ctx.push_context():
                ctx.enqueue_copy(Y.unsafe_ptr(), Y_lt.ptr, y_count)
                ctx.enqueue_copy(
                    final_state.unsafe_ptr(), final_lt.ptr, final_count
                )
            ctx.synchronize()
        else:
            raise Error("ssd_chunk_scan_combined: unsupported target")

    @staticmethod
    def shape[
        dtype: DType,
    ](
        x: InputTensor[dtype=dtype, rank=4, ...],
        dt: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
    ) -> IndexList[4]:
        # NOTE: Op has two OutputTensors (Y + final_state). Return only the
        # primary Y shape — a multi-IndexList / `Tuple` return tripped the
        # MLIR loader's slot-name table. Matches the working pattern used by
        # `gemv_and_partial_norm` in `linalg.mojo` and by the tuplefix branch
        # for `SelectiveScanUpdate` / `VarlenSelectiveStateUpdate`. The
        # final_state shape is reconstructed at the call site (Python wrapper
        # supplies the explicit TensorType in `ssd_chunk_scan_combined`).
        return x.shape()
