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
"""GPU parity tests for the fully on-device ssd_chunk_scan_combined dispatch.

Compares ``_ssd_chunk_scan_combined_gpu`` (the on-device stitched pipeline:
precompute -> intra -> chunk_state -> inter-scan -> combine -> postprocess)
against the golden-pinned ``_ssd_chunk_scan_combined_cpu`` on the same inputs.
The Mamba2-130m profile (state_dim=128, head_dim=64, chunk_size=256) exercises
the fused tensor-core chunk_state path; the small case exercises the scalar
path.
"""

from std.gpu.host import DeviceContext
from std.random import rand
from std.testing import TestSuite, assert_almost_equal

from state_space.ssd_chunk_combined_ops import (
    _ssd_chunk_scan_combined_cpu,
    _ssd_chunk_scan_combined_gpu,
)


def run_gpu_parity[
    dtype: DType,
    chunk_size: Int,
](
    batch: Int,
    seqlen: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.02,
) raises:
    """Run CPU + on-device GPU dispatch on identical inputs and compare."""
    var x_size = batch * seqlen * n_heads * head_dim
    var dt_size = batch * seqlen * n_heads
    var A_size = n_heads
    var bc_size = batch * seqlen * n_heads * state_dim
    var y_size = x_size
    var final_size = batch * n_heads * head_dim * state_dim

    # ── Host inputs (random; A negative so the chunk decays stay bounded) ────
    var x_h = ctx.enqueue_create_host_buffer[dtype](x_size)
    var dt_h = ctx.enqueue_create_host_buffer[dtype](dt_size)
    var A_h = ctx.enqueue_create_host_buffer[dtype](A_size)
    var B_h = ctx.enqueue_create_host_buffer[dtype](bc_size)
    var C_h = ctx.enqueue_create_host_buffer[dtype](bc_size)
    ctx.synchronize()

    rand[dtype](x_h.unsafe_ptr(), x_size)
    rand[dtype](B_h.unsafe_ptr(), bc_size)
    rand[dtype](C_h.unsafe_ptr(), bc_size)
    # dt in (0, 0.1]; A in [-1.0, -0.2].
    rand[dtype](dt_h.unsafe_ptr(), dt_size)
    for i in range(dt_size):
        dt_h[i] = Scalar[dtype](0.01 + 0.09 * dt_h[i].cast[DType.float32]())
    rand[dtype](A_h.unsafe_ptr(), A_size)
    for i in range(A_size):
        A_h[i] = Scalar[dtype](-1.0 + 0.8 * A_h[i].cast[DType.float32]())

    # ── CPU reference ────────────────────────────────────────────────────────
    var Y_ref = ctx.enqueue_create_host_buffer[dtype](y_size)
    var final_ref = ctx.enqueue_create_host_buffer[dtype](final_size)
    ctx.synchronize()
    _ssd_chunk_scan_combined_cpu[dtype](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        chunk_size,
        x_h.unsafe_ptr(),
        dt_h.unsafe_ptr(),
        A_h.unsafe_ptr(),
        B_h.unsafe_ptr(),
        C_h.unsafe_ptr(),
        Y_ref.unsafe_ptr(),
        final_ref.unsafe_ptr(),
    )

    # ── GPU on-device dispatch ───────────────────────────────────────────────
    var x_d = ctx.enqueue_create_buffer[dtype](x_size)
    var dt_d = ctx.enqueue_create_buffer[dtype](dt_size)
    var A_d = ctx.enqueue_create_buffer[dtype](A_size)
    var B_d = ctx.enqueue_create_buffer[dtype](bc_size)
    var C_d = ctx.enqueue_create_buffer[dtype](bc_size)
    var Y_d = ctx.enqueue_create_buffer[dtype](y_size)
    var final_d = ctx.enqueue_create_buffer[dtype](final_size)

    with ctx.push_context():
        ctx.enqueue_copy(x_d, x_h.unsafe_ptr())
        ctx.enqueue_copy(dt_d, dt_h.unsafe_ptr())
        ctx.enqueue_copy(A_d, A_h.unsafe_ptr())
        ctx.enqueue_copy(B_d, B_h.unsafe_ptr())
        ctx.enqueue_copy(C_d, C_h.unsafe_ptr())
    ctx.synchronize()

    _ssd_chunk_scan_combined_gpu[dtype, chunk_size](
        batch,
        seqlen,
        n_heads,
        head_dim,
        state_dim,
        x_d.unsafe_ptr(),
        dt_d.unsafe_ptr(),
        A_d.unsafe_ptr(),
        B_d.unsafe_ptr(),
        C_d.unsafe_ptr(),
        Y_d.unsafe_ptr(),
        final_d.unsafe_ptr(),
        ctx,
    )

    var Y_gpu = ctx.enqueue_create_host_buffer[dtype](y_size)
    var final_gpu = ctx.enqueue_create_host_buffer[dtype](final_size)
    with ctx.push_context():
        ctx.enqueue_copy(Y_gpu.unsafe_ptr(), Y_d)
        ctx.enqueue_copy(final_gpu.unsafe_ptr(), final_d)
    ctx.synchronize()

    for i in range(y_size):
        assert_almost_equal(Y_gpu[i], Y_ref[i], rtol=rtol, atol=1e-3)
    for i in range(final_size):
        assert_almost_equal(final_gpu[i], final_ref[i], rtol=rtol, atol=1e-3)


def test_ssd_chunk_scan_combined_gpu_small() raises:
    """Small case (scalar chunk_state path): b=1, S=8, h=2, P=4, N=8, L=4."""
    var ctx = DeviceContext()
    run_gpu_parity[DType.float32, chunk_size=4](
        batch=1,
        seqlen=8,
        n_heads=2,
        head_dim=4,
        state_dim=8,
        ctx=ctx,
    )


def test_ssd_chunk_scan_combined_gpu_mamba2_profile() raises:
    """Mamba2-130m profile (fused tensor-core chunk_state path).

    b=1, S=512 (2 chunks of 256), h=4, head_dim=64, state_dim=128.
    """
    var ctx = DeviceContext()
    run_gpu_parity[DType.float32, chunk_size=256](
        batch=1,
        seqlen=512,
        n_heads=4,
        head_dim=64,
        state_dim=128,
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
