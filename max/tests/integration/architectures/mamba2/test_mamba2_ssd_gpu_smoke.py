# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""GPU smoke for ssd_chunk_scan_combined.

Confirms the GPU dispatch path (currently host-staged) executes end-to-end
on the host accelerator and that Y agrees with the CPU dispatch to within
the parity tolerance. Skipped if no Accelerator is available.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.mamba2.functional_ops import (
    ssd_chunk_scan_combined,
)


def _run_on(
    device_ref: DeviceRef, device: md.Device
) -> tuple[np.ndarray, np.ndarray]:
    batch = 1
    seqlen = 4
    n_heads = 1
    head_dim = 2
    state_dim = 3
    chunk_size = 2

    with Graph(
        "mamba2_ssd_chunk_scan_combined_gpu_smoke",
        input_types=[
            TensorType(
                DType.float32,
                [batch, seqlen, n_heads, head_dim],
                device=device_ref,
            ),
            TensorType(
                DType.float32, [batch, seqlen, n_heads], device=device_ref
            ),
            TensorType(DType.float32, [n_heads], device=device_ref),
            TensorType(
                DType.float32,
                [batch, seqlen, n_heads, state_dim],
                device=device_ref,
            ),
            TensorType(
                DType.float32,
                [batch, seqlen, n_heads, state_dim],
                device=device_ref,
            ),
        ],
    ) as graph:
        x_v, dt_v, A_v, B_v, C_v = (g.tensor for g in graph.inputs)
        y_v, final_state_v = ssd_chunk_scan_combined(
            x_v, dt_v, A_v, B_v, C_v, chunk_size=chunk_size
        )
        graph.output(y_v, final_state_v)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    in_device = model.input_devices[0]

    x = np.array(
        [0.77, -0.17, -0.66, -0.39, 0.31, -0.62, -1.19, -1.28], dtype=np.float32
    ).reshape(batch, seqlen, n_heads, head_dim)
    dt = np.array([0.33, 0.26, 0.12, 0.24], dtype=np.float32).reshape(
        batch, seqlen, n_heads
    )
    A = np.array([-1.2], dtype=np.float32)
    B = np.array(
        [
            -0.91,
            0.56,
            -1.13,
            -0.41,
            0.59,
            0.28,
            1.46,
            0.67,
            0.19,
            -0.05,
            0.00,
            -0.72,
        ],
        dtype=np.float32,
    ).reshape(batch, seqlen, n_heads, state_dim)
    C = np.array(
        [
            0.64,
            -0.61,
            -1.26,
            -1.05,
            -1.03,
            -0.35,
            1.92,
            0.30,
            1.47,
            -0.55,
            1.06,
            -0.74,
        ],
        dtype=np.float32,
    ).reshape(batch, seqlen, n_heads, state_dim)

    outs = model.execute(
        md.Buffer.from_numpy(x).to(in_device),
        md.Buffer.from_numpy(dt).to(in_device),
        md.Buffer.from_numpy(A).to(in_device),
        md.Buffer.from_numpy(B).to(in_device),
        md.Buffer.from_numpy(C).to(in_device),
    )
    return outs[0].to_numpy(), outs[1].to_numpy()


def test_ssd_chunk_scan_combined_gpu_matches_cpu() -> None:
    """GPU staged dispatch matches CPU dispatch (and the reference golden)."""
    if accelerator_count() == 0:
        pytest.skip("no accelerator available")

    expected_y = np.array(
        [
            0.126999,
            -0.028039,
            0.191223,
            -0.003887,
            -0.370202,
            -0.111640,
            -0.005157,
            -0.252940,
        ],
        dtype=np.float32,
    ).reshape(1, 4, 1, 2)
    expected_final = np.array(
        [-0.009206, 0.020579, 0.043290, -0.014832, -0.091143, 0.222278],
        dtype=np.float32,
    ).reshape(1, 1, 2, 3)

    y_cpu, final_cpu = _run_on(DeviceRef.CPU(), CPU())
    y_gpu, final_gpu = _run_on(DeviceRef.GPU(), Accelerator())

    np.testing.assert_allclose(y_cpu, expected_y, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(final_cpu, expected_final, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(final_gpu, final_cpu, rtol=1e-4, atol=1e-4)
