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
"""Selective scan operation registrations for Mamba SSM.

This module registers the following ops:
- selective_scan_fwd: Full selective scan forward pass
- selective_scan_fwd_minimal: Minimal variant without optional tensors
- selective_scan_update: Single-step update for autoregressive inference
"""

from std.math import ceildiv

import extensibility as compiler
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu

from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList

from state_space.selective_scan import (
    selective_scan_fwd_cpu,
    selective_scan_fwd_gpu,
    selective_scan_fwd_cpu_minimal,
    selective_scan_fwd_gpu_minimal,
    selective_scan_update_cpu,
    selective_scan_update_gpu,
    ssd_combined_cpu,
    ssd_combined_gpu,
    mamba_split_conv1d_scan_combined_cpu,
    mamba_split_conv1d_scan_combined_gpu,
)


@compiler.register("selective_scan_fwd")
struct SelectiveScanFwd[delta_softplus: Bool = False]:
    """Selective scan forward pass operation for Mamba SSM.

    Performs the selective scan computation used in Mamba state space models.
    This is the core operation that processes sequences through the SSM.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - out_z: (batch, dim, seqlen) - Gated output (if z is provided)
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        out_z: OutputTensor[dtype=dtype, rank=3, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor[DType.int32]()
        var x_tt = x.to_tile_tensor[DType.int32]()
        var out_z_tt = out_z.to_tile_tensor[DType.int32]()
        var u_tt = u.to_tile_tensor[DType.int32]()
        var delta_tt = delta.to_tile_tensor[DType.int32]()
        var A_tt = A.to_tile_tensor[DType.int32]()
        var B_tt = B.to_tile_tensor[DType.int32]()
        var C_tt = C.to_tile_tensor[DType.int32]()
        var D_tt = D.to_tile_tensor[DType.int32]()
        var z_tt = z.to_tile_tensor[DType.int32]()
        var delta_bias_tt = delta_bias.to_tile_tensor[DType.int32]()

        var output_strides = output.strides()
        var x_strides = x.strides()
        var out_z_strides = out_z.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var delta_bias_strides = delta_bias.strides()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        # Dispatch runtime dstate to compile-time DSTATE for @parameter for
        # loop unrolling and guaranteed register allocation on GPU.
        comptime if is_cpu[target]():
            if dstate == 16:
                selective_scan_fwd_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    output_strides,
                    x_strides,
                    out_z_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    Optional[DeviceContext](ctx),
                )
            else:
                selective_scan_fwd_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    output_strides,
                    x_strides,
                    out_z_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        out_z_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    output_strides,
                    x_strides,
                    out_z_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        out_z_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        delta_bias_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    out_z_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    delta_bias_tt,
                    output_strides,
                    x_strides,
                    out_z_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    delta_bias_strides,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target: " + target)

    @staticmethod
    def shape[
        dtype: DType,
    ](
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
    ) -> IndexList[3]:
        return u.shape()


@compiler.register("selective_scan_fwd_minimal")
struct SelectiveScanFwdMinimal[delta_softplus: Bool = False]:
    """Minimal selective scan forward pass - no optional D, z, or delta_bias.

    This variant avoids passing empty tensors that could have null pointers.
    Use when D, z, and delta_bias are not provided.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, num_chunks, 2*dstate) - Checkpoint tensor for chunking
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_tt = output.to_tile_tensor[DType.int32]()
        var x_tt = x.to_tile_tensor[DType.int32]()
        var u_tt = u.to_tile_tensor[DType.int32]()
        var delta_tt = delta.to_tile_tensor[DType.int32]()
        var A_tt = A.to_tile_tensor[DType.int32]()
        var B_tt = B.to_tile_tensor[DType.int32]()
        var C_tt = C.to_tile_tensor[DType.int32]()

        var output_strides = output.strides()
        var x_strides = x.strides()
        var u_strides = u.strides()
        var delta_strides = delta.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                selective_scan_fwd_cpu_minimal[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    Optional[DeviceContext](ctx),
                )
            else:
                selective_scan_fwd_cpu_minimal[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu_minimal[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    grid_dim=(num_blocks),
                    block_dim=(BLOCK_SIZE),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_fwd_gpu_minimal[
                        dtype,
                        DSTATE_VAL,
                        output_tt.LayoutType,
                        x_tt.LayoutType,
                        u_tt.LayoutType,
                        delta_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_tt,
                    x_tt,
                    u_tt,
                    delta_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    output_strides,
                    x_strides,
                    u_strides,
                    delta_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    grid_dim=(num_blocks),
                    block_dim=(BLOCK_SIZE),
                )
        else:
            raise Error("Unsupported target device")

    @staticmethod
    def shape[
        dtype: DType,
    ](
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
    ) -> IndexList[3]:
        return u.shape()


@compiler.register("selective_scan_update")
struct SelectiveScanUpdate[delta_softplus: Bool = False]:
    """Selective scan update operation for autoregressive inference.

    Performs a single step of the SSM recurrence for incremental token generation.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - state_out: (batch, dim, dstate) - Updated state output
        - output: (batch, dim) - Output tensor
        - state_in: (batch, dim, dstate) - Input state
        - x: (batch, dim) - Input tensor
        - dt: (batch, dim) - Time delta tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate) - Input matrix
        - C: (batch, n_groups, dstate) - Output matrix
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim) - Gating tensor (optional, can be empty)
        - dt_bias: (dim,) - Time delta bias (optional, can be empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        state_out: OutputTensor[dtype=dtype, rank=3, ...],
        output: OutputTensor[dtype=dtype, rank=2, ...],
        state_in: InputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        dt: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=2, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var batch = state_out.dim_size(0)
        var dim = state_out.dim_size(1)
        var dstate = state_out.dim_size(2)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var state_out_tt = state_out.to_tile_tensor[DType.int32]()
        var output_tt = output.to_tile_tensor[DType.int32]()
        var state_in_tt = state_in.to_tile_tensor[DType.int32]()
        var x_tt = x.to_tile_tensor[DType.int32]()
        var dt_tt = dt.to_tile_tensor[DType.int32]()
        var A_tt = A.to_tile_tensor[DType.int32]()
        var B_tt = B.to_tile_tensor[DType.int32]()
        var C_tt = C.to_tile_tensor[DType.int32]()
        var D_tt = D.to_tile_tensor[DType.int32]()
        var z_tt = z.to_tile_tensor[DType.int32]()
        var dt_bias_tt = dt_bias.to_tile_tensor[DType.int32]()

        var state_out_strides = state_out.strides()
        var output_strides = output.strides()
        var state_in_strides = state_in.strides()
        var x_strides = x.strides()
        var dt_strides = dt.strides()
        var A_strides = A.strides()
        var B_strides = B.strides()
        var C_strides = C.strides()
        var D_strides = D.strides()
        var z_strides = z.strides()
        var dt_bias_strides = dt_bias.strides()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 128 and dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: "
                + String(dstate)
                + ". Expected 8, 16, or 128 (Mamba2)."
            )

        comptime if is_cpu[target]():
            if dstate == 128:
                selective_scan_update_cpu[
                    dtype,
                    128,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    state_out_strides,
                    output_strides,
                    state_in_strides,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    dt_bias_strides,
                    Optional[DeviceContext](ctx),
                )
            elif dstate == 16:
                selective_scan_update_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    state_out_strides,
                    output_strides,
                    state_in_strides,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    dt_bias_strides,
                    Optional[DeviceContext](ctx),
                )
            else:
                selective_scan_update_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    state_out_strides,
                    output_strides,
                    state_in_strides,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    dt_bias_strides,
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 128:
                comptime DSTATE_VAL = 128
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_out_tt.LayoutType,
                        output_tt.LayoutType,
                        state_in_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    state_out_strides,
                    output_strides,
                    state_in_strides,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    dt_bias_strides,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            elif dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_out_tt.LayoutType,
                        output_tt.LayoutType,
                        state_in_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    state_out_strides,
                    output_strides,
                    state_in_strides,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    dt_bias_strides,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    selective_scan_update_gpu[
                        dtype,
                        DSTATE_VAL,
                        state_out_tt.LayoutType,
                        output_tt.LayoutType,
                        state_in_tt.LayoutType,
                        x_tt.LayoutType,
                        dt_tt.LayoutType,
                        A_tt.LayoutType,
                        B_tt.LayoutType,
                        C_tt.LayoutType,
                        D_tt.LayoutType,
                        z_tt.LayoutType,
                        dt_bias_tt.LayoutType,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    group_size,
                    delta_softplus_int8,
                    state_out_tt,
                    output_tt,
                    state_in_tt,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    z_tt,
                    dt_bias_tt,
                    state_out_strides,
                    output_strides,
                    state_in_strides,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    z_strides,
                    dt_bias_strides,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target: " + target)

    @staticmethod
    def shape[
        dtype: DType,
    ](
        state_in: InputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        dt: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=2, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
    ) -> IndexList[3]:
        # NOTE: Op has two OutputTensors (state_out + output). Return only the
        # primary state shape — `Tuple` return tripped the MLIR loader's
        # slot-name table. Matches the working pattern in `linalg.mojo`'s
        # `gemv_and_partial_norm`: op semantics handle the shape split.
        return state_in.shape()


# ===----------------------------------------------------------------------=== #
# SSD Combined (Mamba1 sequential scan fused with norm + residual)
# ===----------------------------------------------------------------------=== #


@compiler.register("ssd_combined")
struct SsdCombined[delta_softplus: Bool = False]:
    """Fused selective scan + RMS-norm + residual operation for Mamba blocks.

    Performs: norm(residual + selective_scan(input)), optionally gated by SiLU(z).
    Despite the name (`ssd_combined`), this is the Mamba1-shape *sequential* scan
    fused with the post-norm/residual path — not the chunk-SSD algorithm.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Final fused output (normalized, optionally gated)
        - x: (batch, dim, num_chunks, 2*dstate) - Scan checkpoint state (output)
        - out_z: (batch, dim, seqlen) - Pre-gating normalized output (output, can be empty)
        - residual: (batch, dim, seqlen) - Residual tensor added before norm
        - u: (batch, dim, seqlen) - Input tensor
        - delta: (batch, dim, seqlen) - Time step tensor
        - A: (dim, dstate) - State transition matrix
        - B: (batch, n_groups, dstate, seqlen) - Input projection
        - C: (batch, n_groups, dstate, seqlen) - Output projection
        - D: (dim,) - Skip connection (optional, can be empty)
        - z: (batch, dim, seqlen) - Gating tensor (optional, can be empty)
        - delta_bias: (dim,) - Delta bias (optional, can be empty)
        - gamma: (dim,) - RMSNorm weight
        - epsilon: Scalar - Numerical stability epsilon for RMSNorm
        - weight_offset: Scalar - Offset added to gamma before scaling
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        x: OutputTensor[dtype=dtype, rank=4, ...],
        out_z: OutputTensor[dtype=dtype, rank=3, ...],
        residual: InputTensor[dtype=dtype, rank=3, ...],
        u: InputTensor[dtype=dtype, rank=3, ...],
        delta: InputTensor[dtype=dtype, rank=3, ...],
        A: InputTensor[dtype=dtype, rank=2, ...],
        B: InputTensor[dtype=dtype, rank=4, ...],
        C: InputTensor[dtype=dtype, rank=4, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        z: InputTensor[dtype=dtype, rank=3, ...],
        delta_bias: InputTensor[dtype=dtype, rank=1, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != u.shape():
            raise Error("Output shape must match input u shape")

        var batch = output.dim_size(0)
        var dim = output.dim_size(1)
        var seqlen = output.dim_size(2)
        var dstate = A.dim_size(1)
        var n_groups = B.dim_size(1)
        var group_size = dim // n_groups

        var output_lt = output.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var out_z_lt = out_z.to_layout_tensor()
        var residual_lt = residual.to_layout_tensor()
        var u_lt = u.to_layout_tensor()
        var delta_lt = delta.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var z_lt = z.to_layout_tensor()
        var delta_bias_lt = delta_bias.to_layout_tensor()
        var gamma_lt = gamma.to_layout_tensor()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                ssd_combined_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_lt,
                    x_lt,
                    out_z_lt,
                    residual_lt,
                    u_lt,
                    delta_lt,
                    A_lt,
                    B_lt,
                    C_lt,
                    D_lt,
                    z_lt,
                    delta_bias_lt,
                    gamma_lt,
                    epsilon,
                    weight_offset,
                    Optional[DeviceContext](ctx),
                )
            else:
                ssd_combined_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_lt,
                    x_lt,
                    out_z_lt,
                    residual_lt,
                    u_lt,
                    delta_lt,
                    A_lt,
                    B_lt,
                    C_lt,
                    D_lt,
                    z_lt,
                    delta_bias_lt,
                    gamma_lt,
                    epsilon,
                    weight_offset,
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    ssd_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        residual_lt.layout,
                        u_lt.layout,
                        delta_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        delta_bias_lt.layout,
                        gamma_lt.layout,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_lt,
                    x_lt,
                    out_z_lt,
                    residual_lt,
                    u_lt,
                    delta_lt,
                    A_lt,
                    B_lt,
                    C_lt,
                    D_lt,
                    z_lt,
                    delta_bias_lt,
                    gamma_lt,
                    epsilon,
                    weight_offset,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    ssd_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        residual_lt.layout,
                        u_lt.layout,
                        delta_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        delta_bias_lt.layout,
                        gamma_lt.layout,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    dim,
                    seqlen,
                    group_size,
                    delta_softplus_int8,
                    output_lt,
                    x_lt,
                    out_z_lt,
                    residual_lt,
                    u_lt,
                    delta_lt,
                    A_lt,
                    B_lt,
                    C_lt,
                    D_lt,
                    z_lt,
                    delta_bias_lt,
                    gamma_lt,
                    epsilon,
                    weight_offset,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target: " + target)


# ===----------------------------------------------------------------------=== #
# Mamba Split Conv1d Scan Combined (Mamba2-shape fused conv + sequential scan)
# ===----------------------------------------------------------------------=== #


@compiler.register("mamba_split_conv1d_scan_combined")
struct MambaSplitConv1dScanCombined[delta_softplus: Bool = False]:
    """Fused split-conv1d + selective scan + optional RMSNorm + optional outproj.

    Mamba2-shape kernel: head-sharded A (one scalar per head), sequential scan
    inside, but fused with the upstream conv1d (after splitting xBC out of the
    packed `zxbcdt` projection input) and optional downstream norm/outproj.

    Parameters:
        delta_softplus: If True, applies softplus activation to delta values.

    Tensor Shapes:
        - zxbcdt: (batch, seqlen, 2*dim + 2*ngroups*dstate + nheads) - Packed projection input
        - conv_weight: (dim + 2*ngroups*dstate, width) - Conv1d weights for x/B/C channels
        - conv_bias: (dim + 2*ngroups*dstate,) - Conv1d bias
        - dt_bias: (nheads,) - dt bias (per head)
        - A: (nheads,) - Scalar state-transition value per head
        - D: (nheads, headdim) or (nheads,) - Skip connection (optional, can be empty)
        - x: (batch, dim, num_chunks, 2*dstate) - Scan checkpoint state (output)
        - out_z: (batch, dim, seqlen) - Pre-gating output (output)
        - dt: (batch, nheads, seqlen) - dt tensor (output)
        - B: (batch, ngroups, dstate, seqlen) - Input projection (output of conv split)
        - C: (batch, ngroups, dstate, seqlen) - Output projection (output of conv split)
        - z: (batch, dim, seqlen) - Gating tensor (output, split from zxbcdt)
        - rmsnorm_weight: (dim,) - RMSNorm weight (optional, can be empty)
        - outproj_weight: (out_dim, dim) - Output projection weight (optional, can be empty)
        - outproj_bias: (out_dim,) - Output projection bias (optional, can be empty)
        - output: (batch, seqlen, dim) or (batch, seqlen, out_dim) - Final output
        - epsilon: Scalar - RMSNorm epsilon

    Runtime scalar args:
        seqlen, dim, nheads, headdim, ngroups, width, chunk_size - Shape config
        norm_before_gate, has_rmsnorm, has_outproj - Int8 flags (0/1)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        x: OutputTensor[dtype=dtype, rank=4, ...],
        out_z: OutputTensor[dtype=dtype, rank=3, ...],
        dt: OutputTensor[dtype=dtype, rank=3, ...],
        B: OutputTensor[dtype=dtype, rank=4, ...],
        C: OutputTensor[dtype=dtype, rank=4, ...],
        z: OutputTensor[dtype=dtype, rank=3, ...],
        output: OutputTensor[dtype=dtype, rank=3, ...],
        zxbcdt: InputTensor[dtype=dtype, rank=3, ...],
        conv_weight: InputTensor[dtype=dtype, rank=2, ...],
        conv_bias: InputTensor[dtype=dtype, rank=1, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        D: InputTensor[dtype=dtype, rank=2, ...],
        rmsnorm_weight: InputTensor[dtype=dtype, rank=1, ...],
        outproj_weight: InputTensor[dtype=dtype, rank=2, ...],
        outproj_bias: InputTensor[dtype=dtype, rank=1, ...],
        nheads: Int,
        headdim: Int,
        ngroups: Int,
        width: Int,
        chunk_size: Int,
        norm_before_gate: Int8,
        has_rmsnorm: Int8,
        has_outproj: Int8,
        epsilon: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        var batch = zxbcdt.dim_size(0)
        var seqlen = zxbcdt.dim_size(1)
        var dim = nheads * headdim
        # dstate inferred from B's third dim (B is (batch, ngroups, dstate, seqlen))
        var dstate = B.dim_size(2)

        var zxbcdt_lt = zxbcdt.to_layout_tensor()
        var conv_weight_lt = conv_weight.to_layout_tensor()
        var conv_bias_lt = conv_bias.to_layout_tensor()
        var dt_bias_lt = dt_bias.to_layout_tensor()
        var A_lt = A.to_layout_tensor()
        var D_lt = D.to_layout_tensor()
        var x_lt = x.to_layout_tensor()
        var out_z_lt = out_z.to_layout_tensor()
        var dt_lt = dt.to_layout_tensor()
        var B_lt = B.to_layout_tensor()
        var C_lt = C.to_layout_tensor()
        var z_lt = z.to_layout_tensor()
        var rmsnorm_weight_lt = rmsnorm_weight.to_layout_tensor()
        var outproj_weight_lt = outproj_weight.to_layout_tensor()
        var outproj_bias_lt = outproj_bias.to_layout_tensor()
        var output_lt = output.to_layout_tensor()

        comptime delta_softplus_int8: Int8 = Int8(
            1
        ) if Self.delta_softplus else Int8(0)

        if dstate != 16 and dstate != 8:
            raise Error(
                "Unsupported dstate: " + String(dstate) + ". Expected 8 or 16."
            )

        comptime if is_cpu[target]():
            if dstate == 16:
                mamba_split_conv1d_scan_combined_cpu[
                    dtype,
                    16,
                ](
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    chunk_size,
                    delta_softplus_int8,
                    norm_before_gate,
                    has_rmsnorm,
                    has_outproj,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                    Optional[DeviceContext](ctx),
                )
            else:
                mamba_split_conv1d_scan_combined_cpu[
                    dtype,
                    8,
                ](
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    chunk_size,
                    delta_softplus_int8,
                    norm_before_gate,
                    has_rmsnorm,
                    has_outproj,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                    Optional[DeviceContext](ctx),
                )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            var total_batch_dim = batch * dim
            comptime BLOCK_SIZE = 128
            var num_blocks = ceildiv(total_batch_dim, BLOCK_SIZE)

            if dstate == 16:
                comptime DSTATE_VAL = 16
                var compiled_kernel = gpu_ctx.compile_function[
                    mamba_split_conv1d_scan_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        zxbcdt_lt.layout,
                        conv_weight_lt.layout,
                        conv_bias_lt.layout,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        dt_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        dt_bias_lt.layout,
                        rmsnorm_weight_lt.layout,
                        outproj_weight_lt.layout,
                        outproj_bias_lt.layout,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    chunk_size,
                    delta_softplus_int8,
                    norm_before_gate,
                    has_rmsnorm,
                    has_outproj,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
            else:
                comptime DSTATE_VAL = 8
                var compiled_kernel = gpu_ctx.compile_function[
                    mamba_split_conv1d_scan_combined_gpu[
                        dtype,
                        DSTATE_VAL,
                        zxbcdt_lt.layout,
                        conv_weight_lt.layout,
                        conv_bias_lt.layout,
                        output_lt.layout,
                        x_lt.layout,
                        out_z_lt.layout,
                        dt_lt.layout,
                        A_lt.layout,
                        B_lt.layout,
                        C_lt.layout,
                        D_lt.layout,
                        z_lt.layout,
                        dt_bias_lt.layout,
                        rmsnorm_weight_lt.layout,
                        outproj_weight_lt.layout,
                        outproj_bias_lt.layout,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_kernel,
                    total_batch_dim,
                    batch,
                    seqlen,
                    dim,
                    nheads,
                    headdim,
                    ngroups,
                    width,
                    chunk_size,
                    delta_softplus_int8,
                    norm_before_gate,
                    has_rmsnorm,
                    has_outproj,
                    zxbcdt_lt,
                    conv_weight_lt,
                    conv_bias_lt,
                    dt_bias_lt,
                    A_lt,
                    D_lt,
                    x_lt,
                    out_z_lt,
                    dt_lt,
                    B_lt,
                    C_lt,
                    z_lt,
                    rmsnorm_weight_lt,
                    outproj_weight_lt,
                    outproj_bias_lt,
                    output_lt,
                    epsilon,
                    grid_dim=(num_blocks,),
                    block_dim=(BLOCK_SIZE,),
                )
        else:
            raise Error("Unsupported target: " + target)
