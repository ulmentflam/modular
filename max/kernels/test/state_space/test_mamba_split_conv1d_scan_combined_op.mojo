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
"""Op-registration smoke test for `mamba_split_conv1d_scan_combined`.

Numerical parity is covered by `test_mamba_split_conv1d_scan_combined.mojo`;
this file exists to ensure the @compiler.register wrapper in
`selective_scan_ops.mojo` is built into the state_space library (a malformed
@compiler.register block fails the library build), and that the underlying
kernel is reachable via the same import path the wrapper uses.
"""

from std.math import ceildiv

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from state_space.selective_scan import mamba_split_conv1d_scan_combined_cpu

# Importing the ops module ensures the @compiler.register-decorated structs
# are part of the build graph for the consumer of the smoke test.
import state_space.selective_scan_ops

from std.testing import TestSuite, assert_true
from std.utils.index import Index


def test_mamba_split_conv1d_scan_combined_op_registered() raises:
    """Smoke test: mamba_split_conv1d_scan_combined op + underlying CPU kernel
    are callable.

    Constructs tiny LayoutTensor inputs, invokes the CPU implementation that
    the registered op dispatches to, and verifies the output buffer is finite
    (no NaN/Inf) and shape-correct.
    """
    comptime dtype = DType.float32
    comptime DSTATE = 8

    var batch = 1
    var nheads = 1
    var headdim = 2
    var dim = nheads * headdim
    var seqlen = 4
    var ngroups = 1
    var width = 3
    var chunk_size = 2048
    var n_chunks = ceildiv(seqlen, chunk_size)

    comptime layout_3d = Layout.row_major[3]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var zxbcdt_channels = 2 * dim + 2 * ngroups * DSTATE + nheads
    var zxbcdt_heap = List(
        length=batch * seqlen * zxbcdt_channels, fill=Scalar[dtype](0)
    )
    var zxbcdt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        zxbcdt_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(batch, seqlen, zxbcdt_channels)
        ),
    )

    var conv_weight_channels = dim + 2 * ngroups * DSTATE
    var conv_weight_heap = List(
        length=conv_weight_channels * width, fill=Scalar[dtype](0)
    )
    var conv_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        conv_weight_heap,
        RuntimeLayout[layout_2d].row_major(Index(conv_weight_channels, width)),
    )

    var conv_bias_heap = List(
        length=conv_weight_channels, fill=Scalar[dtype](0)
    )
    var conv_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        conv_bias_heap,
        RuntimeLayout[layout_1d].row_major(Index(conv_weight_channels)),
    )

    var dt_bias_heap = List(length=nheads, fill=Scalar[dtype](0))
    var dt_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        dt_bias_heap, RuntimeLayout[layout_1d].row_major(Index(nheads))
    )

    var A_heap = List(length=nheads, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        A_heap, RuntimeLayout[layout_1d].row_major(Index(nheads))
    )

    var D_heap = List(length=nheads * headdim, fill=Scalar[dtype](0))
    var D_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        D_heap,
        RuntimeLayout[layout_2d].row_major(Index(nheads, headdim)),
    )

    var x_heap = List(
        length=batch * dim * n_chunks * 2 * DSTATE, fill=Scalar[dtype](0)
    )
    var x_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        x_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, dim, n_chunks, 2 * DSTATE)
        ),
    )

    var out_z_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var out_z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        out_z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    var dt_heap = List(length=batch * nheads * seqlen, fill=Scalar[dtype](0))
    var dt_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        dt_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, nheads, seqlen)),
    )

    var B_heap = List(
        length=batch * ngroups * DSTATE * seqlen, fill=Scalar[dtype](0)
    )
    var B_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, ngroups, DSTATE, seqlen)
        ),
    )

    var C_heap = List(
        length=batch * ngroups * DSTATE * seqlen, fill=Scalar[dtype](0)
    )
    var C_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, ngroups, DSTATE, seqlen)
        ),
    )

    var z_heap = List(length=batch * dim * seqlen, fill=Scalar[dtype](0))
    var z_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        z_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, dim, seqlen)),
    )

    var rmsnorm_weight_heap = List(length=dim, fill=Scalar[dtype](0))
    var rmsnorm_weight_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        rmsnorm_weight_heap, RuntimeLayout[layout_1d].row_major(Index(dim))
    )

    var outproj_weight_heap = List(length=1, fill=Scalar[dtype](0))
    var outproj_weight_h = LayoutTensor[dtype, layout_2d, MutAnyOrigin](
        outproj_weight_heap,
        RuntimeLayout[layout_2d].row_major(Index(0, 0)),
    )

    var outproj_bias_heap = List(length=1, fill=Scalar[dtype](0))
    var outproj_bias_h = LayoutTensor[dtype, layout_1d, MutAnyOrigin](
        outproj_bias_heap, RuntimeLayout[layout_1d].row_major(Index(0))
    )

    var output_heap = List(length=batch * seqlen * dim, fill=Scalar[dtype](0))
    var output_h = LayoutTensor[dtype, layout_3d, MutAnyOrigin](
        output_heap,
        RuntimeLayout[layout_3d].row_major(Index(batch, seqlen, dim)),
    )

    random(zxbcdt_h)
    random(conv_weight_h)
    random(conv_bias_h)
    random(dt_bias_h)
    random(A_h)
    random(D_h)
    random(rmsnorm_weight_h)
    # Keep RMSNorm weight positive.
    for i in range(dim):
        rmsnorm_weight_h.ptr[i] = abs(rmsnorm_weight_h.ptr[i]) + Scalar[dtype](
            0.1
        )

    var epsilon = Scalar[dtype](0.001)

    mamba_split_conv1d_scan_combined_cpu[
        dtype,
        DSTATE,
        zxbcdt_h.layout,
        conv_weight_h.layout,
        conv_bias_h.layout,
        output_h.layout,
        x_h.layout,
        out_z_h.layout,
        dt_h.layout,
        A_h.layout,
        B_h.layout,
        C_h.layout,
        D_h.layout,
        z_h.layout,
        dt_bias_h.layout,
        rmsnorm_weight_h.layout,
        outproj_weight_h.layout,
        outproj_bias_h.layout,
    ](
        batch,
        seqlen,
        dim,
        nheads,
        headdim,
        ngroups,
        width,
        chunk_size,
        Int8(0),  # delta_softplus
        Int8(0),  # norm_before_gate
        Int8(1),  # has_rmsnorm
        Int8(0),  # has_outproj
        zxbcdt_h,
        conv_weight_h,
        conv_bias_h,
        dt_bias_h,
        A_h,
        D_h,
        x_h,
        out_z_h,
        dt_h,
        B_h,
        C_h,
        z_h,
        rmsnorm_weight_h,
        outproj_weight_h,
        outproj_bias_h,
        output_h,
        epsilon,
    )

    # Shape check + finite check.
    assert_true(output_h.dim(0) == batch)
    assert_true(output_h.dim(1) == seqlen)
    assert_true(output_h.dim(2) == dim)

    var out_size = batch * seqlen * dim
    for i in range(out_size):
        var v = Float32(output_h.ptr[i])
        assert_true(v == v, "output contains NaN")
        assert_true((v - v) == 0.0, "output contains Inf")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
