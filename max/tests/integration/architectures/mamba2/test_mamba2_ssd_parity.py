# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""MAX vs ssd_minimal_discrete parity for ssd_chunk_scan_combined (GPU).

Covers the cross-framework parity gap. The bazel env has MAX + numpy but a
torch that can't load CUDA here; the parity venv has torch but no MAX. So the
oracle here is a numpy port of ``ssd_minimal_discrete`` that is validated to
match the vendored torch reference (``modular-parity/ssd_minimal_ref.py``) to
~1e-6 on identical seeded inputs (see ``modular-parity/validate_np_ref.py``).
``np.random.default_rng`` is stable across numpy versions, so the inputs here
are bit-identical to the ones used in that torch validation. Therefore:

    MAX op == numpy oracle (this test) and numpy oracle == torch (validated)
    => MAX ssd_chunk_scan_combined == torch ssd_minimal at real size.

The discretization contract (from the torch reference):
    op(x, dt, A, B, C, chunk_size)
        == ssd_minimal_discrete(x * dt[..., None], A[h] * dt, B, C, chunk_size)
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import pytest
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.mamba2.functional_ops import (
    ssd_chunk_scan_combined,
)


# ── numpy oracle (validated against torch ssd_minimal_discrete) ──────────────
def _segsum_np(x: np.ndarray) -> np.ndarray:
    T = x.shape[-1]
    xr = np.broadcast_to(x[..., :, None], x.shape + (T,)).copy()
    d = np.arange(T)[:, None]
    e = np.arange(T)[None, :]
    xr = np.where(e < d, xr, 0.0)
    s = np.cumsum(xr, axis=-2)
    return np.where(e <= d, s, -np.inf)


def _ssd_minimal_np(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    block_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    b, l, h, p = X.shape
    c = l // block_len

    def ch(t: np.ndarray) -> np.ndarray:
        return t.reshape((b, c, block_len) + t.shape[2:])

    Xc, Ac, Bc, Cc = ch(X), ch(A), ch(B), ch(C)
    Ah = np.transpose(Ac, (0, 3, 1, 2))
    A_cumsum = np.cumsum(Ah, axis=-1)
    L = np.exp(_segsum_np(Ah))
    Y_diag = np.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, L, Xc)
    decay_states = np.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = np.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, Xc)
    states_cat = np.concatenate([np.zeros_like(states[:, :1]), states], axis=1)
    pad = np.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))
    decay_chunk = np.exp(_segsum_np(pad))
    new_states = np.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    states_off, final_state = new_states[:, :-1], new_states[:, -1]
    Y_off = np.einsum(
        "bclhn,bchpn,bhcl->bclhp", Cc, states_off, np.exp(A_cumsum)
    )
    return (Y_diag + Y_off).reshape(b, l, h, p), final_state


def _gen_inputs(
    seed: int, b: int, s: int, h: int, p: int, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((b, s, h, p)).astype(np.float32)
    B = rng.standard_normal((b, s, h, n)).astype(np.float32)
    C = rng.standard_normal((b, s, h, n)).astype(np.float32)
    dt = (0.01 + 0.09 * rng.random((b, s, h))).astype(np.float32)
    A = (-1.0 + 0.8 * rng.random((h,))).astype(np.float32)
    return x, dt, A, B, C


def _run_max(
    x: np.ndarray,
    dt: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    chunk_size: int,
    device: md.Device,
    device_ref: DeviceRef,
) -> tuple[np.ndarray, np.ndarray]:
    b, s, h, p = x.shape
    n = B.shape[-1]
    with Graph(
        "mamba2_ssd_parity",
        input_types=[
            TensorType(DType.float32, [b, s, h, p], device=device_ref),
            TensorType(DType.float32, [b, s, h], device=device_ref),
            TensorType(DType.float32, [h], device=device_ref),
            TensorType(DType.float32, [b, s, h, n], device=device_ref),
            TensorType(DType.float32, [b, s, h, n], device=device_ref),
        ],
    ) as graph:
        x_v, dt_v, A_v, B_v, C_v = (g.tensor for g in graph.inputs)
        y_v, f_v = ssd_chunk_scan_combined(
            x_v, dt_v, A_v, B_v, C_v, chunk_size=chunk_size
        )
        graph.output(y_v, f_v)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    ind = model.input_devices[0]
    outs = model.execute(
        md.Buffer.from_numpy(x).to(ind),
        md.Buffer.from_numpy(dt).to(ind),
        md.Buffer.from_numpy(A).to(ind),
        md.Buffer.from_numpy(B).to(ind),
        md.Buffer.from_numpy(C).to(ind),
    )
    return outs[0].to_numpy(), outs[1].to_numpy()


@pytest.mark.parametrize(
    "b,s,h,p,n,cs",
    [
        (1, 8, 2, 4, 8, 4),  # small (scalar chunk_state path)
        (1, 512, 4, 64, 128, 256),  # Mamba2-130m profile (fused path)
    ],
)
def test_ssd_chunk_scan_combined_matches_ssd_minimal(
    b: int, s: int, h: int, p: int, n: int, cs: int
) -> None:
    if accelerator_count() == 0:
        pytest.skip("no accelerator available")

    x, dt, A, B, C = _gen_inputs(0, b, s, h, p, n)

    # Oracle: ssd_minimal_discrete on the discretized inputs.
    X = x * dt[..., None]
    A_disc = A[None, None, :] * dt
    y_ref, f_ref = _ssd_minimal_np(X, A_disc, B, C, cs)

    device = Accelerator()
    y_max, f_max = _run_max(x, dt, A, B, C, cs, device, DeviceRef.GPU())

    # The Mamba2 profile takes the fused tensor-core chunk_state path, which
    # runs the K=chunk_len reduction in TF32 (~10-bit mantissa) while the numpy
    # oracle is FP32, so allow TF32-level error (atol 1e-2; the scalar small
    # case agrees far tighter). Already cross-checked GPU==CPU at rtol=2e-2 in
    # test_ssd_chunk_combined_ops.
    np.testing.assert_allclose(y_max, y_ref, rtol=2e-2, atol=1e-2)
    np.testing.assert_allclose(f_max, f_ref, rtol=2e-2, atol=1e-2)
