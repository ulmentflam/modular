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

from std.math import ceildiv, exp

from std.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.random import rand
from state_space.ssd_chunk import ssd_intra_chunk_fwd_gpu
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index


def run_ssd_intra_chunk_gpu[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the SSD intra-chunk GPU kernel and check it against a CPU reference.

    One GPU thread per ``(batch, chunk, head)`` slice computes the intra-chunk
    diagonal block output; this validates the result against the same naive
    triple-loop reference used by the CPU test.
    """
    comptime SLICE_BLOCK_SIZE = 128
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    # ── Host tensors ─────────────────────────────────────────────────────────
    var C_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    var C_h = LayoutTensor[dtype, layout_5d, _](
        C_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](C_h.ptr, C_h.size())

    var B_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    var B_h = LayoutTensor[dtype, layout_5d, _](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](B_h.ptr, B_h.size())

    var X_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    var X_h = LayoutTensor[dtype, layout_5d, _](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](X_h.ptr, X_h.size())

    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var A_h = LayoutTensor[dtype, layout_4d, _](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](A_h.ptr, A_h.size())
    # Make A negative-ish (a real SSM decays) so exp(cumsum diffs) stays
    # bounded. `rand` fills in [0, 1); map into a small negative range.
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        # Map [0, 1) -> [-0.6, -0.1].
        var scaled = Float32(-0.6) + Float32(0.5) * v
        A_h.ptr[i] = Scalar[dtype](scaled)

    var Y_gpu_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    var Y_gpu_h = LayoutTensor[dtype, layout_5d, _](
        Y_gpu_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # ── Device buffers ───────────────────────────────────────────────────────
    var C_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var B_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var X_device = ctx.enqueue_create_buffer[dtype](xy_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var Y_device = ctx.enqueue_create_buffer[dtype](xy_count)

    with ctx.push_context():
        ctx.enqueue_copy(C_device, C_h.ptr)
        ctx.enqueue_copy(B_device, B_h.ptr)
        ctx.enqueue_copy(X_device, X_h.ptr)
        ctx.enqueue_copy(A_device, A_h.ptr)

    var C_tt = TileTensor(
        C_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var B_tt = TileTensor(
        B_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var X_tt = TileTensor(
        X_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var A_tt = TileTensor(
        A_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var Y_tt = TileTensor(
        Y_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )

    var total_slices = batch * n_chunks * n_heads

    var compiled_func = ctx.compile_function[
        ssd_intra_chunk_fwd_gpu[
            dtype,
            C_tt.LayoutType,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            Y_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            batch,
            n_chunks,
            n_heads,
            chunk_len,
            state_dim,
            head_dim,
            C_tt,
            B_tt,
            X_tt,
            A_tt,
            Y_tt,
            grid_dim=(ceildiv(total_slices, SLICE_BLOCK_SIZE),),
            block_dim=(SLICE_BLOCK_SIZE,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(Y_gpu_h.ptr, Y_device)
    ctx.synchronize()

    # ── CPU reference: naive triple-loop of the SSD intra-chunk formula ──────
    var ref_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)

    var cb_h_stride = chunk_len * state_dim
    var cb_c_stride = n_heads * cb_h_stride
    var cb_b_stride = n_chunks * cb_c_stride

    var xy_h_stride = chunk_len * head_dim
    var xy_c_stride = n_heads * xy_h_stride
    var xy_b_stride = n_chunks * xy_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var cb_base = (
                    b * cb_b_stride + c * cb_c_stride + h * cb_h_stride
                )
                var xy_base = (
                    b * xy_b_stride + c * xy_c_stride + h * xy_h_stride
                )
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride

                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc

                for l in range(chunk_len):
                    for p in range(head_dim):
                        var y_acc = Float32(0.0)
                        for s in range(l + 1):
                            var dot = Float32(0.0)
                            for n in range(state_dim):
                                var cv = C_h.ptr[
                                    cb_base + l * state_dim + n
                                ].cast[DType.float32]()
                                var bv = B_h.ptr[
                                    cb_base + s * state_dim + n
                                ].cast[DType.float32]()
                                dot += cv * bv
                            var decay = exp(cumsum[l] - cumsum[s])
                            var xv = X_h.ptr[xy_base + s * head_dim + p].cast[
                                DType.float32
                            ]()
                            y_acc += dot * decay * xv
                        ref_heap[xy_base + l * head_dim + p] = Scalar[dtype](
                            y_acc
                        )

    # ── Compare GPU vs CPU reference ─────────────────────────────────────────
    for i in range(xy_count):
        assert_almost_equal(Y_gpu_h.ptr[i], ref_heap[i], rtol=rtol)


def test_ssd_intra_chunk_gpu_golden() raises:
    """Golden-parity test on GPU against the canonical PyTorch SSD reference.

    Same FIXED tiny input as the CPU golden test (batch=1, n_chunks=1,
    n_heads=1, L=4, N=3, P=2). Pins the GPU kernel to ``intra_chunk_diag`` in
    ``.planning/parity/ssd_minimal_ref.py``.
    """
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 1
    comptime n_heads = 1
    comptime chunk_len = 4
    comptime state_dim = 3
    comptime head_dim = 2

    var C_vals = [
        Float32(-0.15),
        0.79,
        0.95,
        -1.11,
        1.69,
        -0.89,
        -0.36,
        1.23,
        0.14,
        -1.68,
        0.32,
        0.13,
    ]
    var B_vals = [
        Float32(0.14),
        0.24,
        1.40,
        1.35,
        2.44,
        0.20,
        2.45,
        2.03,
        1.78,
        -0.92,
        -0.46,
        -0.72,
    ]
    var X_vals = [
        Float32(1.28),
        -0.99,
        1.81,
        -0.60,
        1.61,
        1.93,
        -0.42,
        -0.08,
    ]
    var A_vals = [Float32(-0.43), -0.34, -0.49, -0.46]
    var Y_expected = [
        Float32(1.918208),
        -1.483614,
        3.522012,
        -0.766567,
        6.067267,
        2.472605,
        -4.850489,
        -3.713203,
    ]

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    # Host buffers from fixed literals.
    var C_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    for i in range(cb_count):
        C_heap[i] = Scalar[dtype](C_vals[i])
    var B_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    for i in range(cb_count):
        B_heap[i] = Scalar[dtype](B_vals[i])
    var X_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    for i in range(xy_count):
        X_heap[i] = Scalar[dtype](X_vals[i])
    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    for i in range(a_count):
        A_heap[i] = Scalar[dtype](A_vals[i])
    var Y_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)

    var C_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var B_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var X_device = ctx.enqueue_create_buffer[dtype](xy_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var Y_device = ctx.enqueue_create_buffer[dtype](xy_count)

    with ctx.push_context():
        ctx.enqueue_copy(C_device, C_heap)
        ctx.enqueue_copy(B_device, B_heap)
        ctx.enqueue_copy(X_device, X_heap)
        ctx.enqueue_copy(A_device, A_heap)

    var C_tt = TileTensor(
        C_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var B_tt = TileTensor(
        B_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var X_tt = TileTensor(
        X_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var A_tt = TileTensor(
        A_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var Y_tt = TileTensor(
        Y_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )

    var total_slices = batch * n_chunks * n_heads

    var compiled_func = ctx.compile_function[
        ssd_intra_chunk_fwd_gpu[
            dtype,
            C_tt.LayoutType,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            Y_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            batch,
            n_chunks,
            n_heads,
            chunk_len,
            state_dim,
            head_dim,
            C_tt,
            B_tt,
            X_tt,
            A_tt,
            Y_tt,
            grid_dim=(1,),
            block_dim=(total_slices,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(Y_heap, Y_device)
    ctx.synchronize()

    for i in range(xy_count):
        assert_almost_equal(
            Y_heap[i],
            Scalar[dtype](Y_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_intra_chunk_gpu_basic() raises:
    """Basic single-batch, single-chunk case against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=1,
        n_chunks=1,
        n_heads=2,
        chunk_len=8,
        state_dim=16,
        head_dim=8,
        ctx=ctx,
    )


def test_ssd_intra_chunk_gpu_multi_chunk() raises:
    """Multiple chunks and heads against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=1,
        n_chunks=3,
        n_heads=2,
        chunk_len=16,
        state_dim=16,
        head_dim=8,
        ctx=ctx,
    )


def test_ssd_intra_chunk_gpu_multi_batch() raises:
    """Multiple batch elements against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=2,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        state_dim=8,
        head_dim=4,
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
