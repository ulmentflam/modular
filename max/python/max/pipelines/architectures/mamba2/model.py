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
"""Pipeline-level Mamba2Model wiring (prefill + step + SSM cache).

Mirrors :mod:`max.pipelines.architectures.mamba.model` for the SSD
architecture. Key differences from Phase-1:

* The SSD prefill kernel now exposes ``final_state`` alongside ``Y``;
  :class:`Mamba2Prefill` forwards each layer's final SSM state and the
  tail of its pre-conv xBC input as the rolling conv state, and
  :class:`Mamba2Model.execute` seeds both halves of the per-slot cache
  before step-mode decode runs. The conv-state slice is purely Python-side
  (no kernel change needed) — the last ``d_conv-1`` tokens of the conv
  input are exactly the decode kernel's expected rolling-history layout.
* Per-slot cache shapes match the Mamba2 mixer layout
  (``conv_state: [1, conv_dim, d_conv-1]``,
  ``ssm_state: [1, nheads, head_dim, d_state]``).

End-to-end smoke is currently blocked by the upstream Tuple-shape bug
in the state_space kernel package — see
``proposed/approvals/causal_conv1d_ops-tuple-shape-bug.md``. This module
is intentionally written so that, once the loader unblocks, only the
``execute`` path needs to be exercised; nothing in the
``Mamba2Model.__init__`` path requires runtime kernel execution.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.modeling.types import LogProbabilities, RequestID
from max.profiler import traced
from transformers import AutoConfig

from .functional_ops import _get_state_space_paths
from .model_config import Mamba2Config
from .ssm_cache import Mamba2SSMStateCache
from .weight_adapters import convert_mamba2_state_dict

logger = logging.getLogger("max.pipelines.mamba2")


class Mamba2ModelInputs(ModelInputs):
    """Inputs for the Mamba2 pipeline model.

    Mirrors :class:`max.pipelines.architectures.mamba.model.MambaModelInputs`
    exactly — same flag semantics, same layer_states convention
    (``[conv_0, ssm_0, conv_1, ssm_1, ...]`` per Mamba1 ordering).
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    is_prefill: bool
    layer_states: list[Buffer]
    request_ids: list[RequestID]

    def __init__(
        self,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        is_prefill: bool = True,
        layer_states: list[Buffer] | None = None,
        request_ids: list[RequestID] | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.is_prefill = is_prefill
        self.layer_states = layer_states or []
        self.request_ids = request_ids or []


class Mamba2Model(PipelineModelWithKVCache[TextContext]):
    """Mamba2 pipeline model with incremental SSM state caching.

    The model compiles two separate MAX-graph modules:

    * :class:`Mamba2Prefill` — full-prompt forward, returns
      ``(logits, conv_0, ssm_0, ..., conv_{N-1}, ssm_{N-1})``. After
      execute, the per-layer ``(conv, ssm)`` pairs are written into the
      cache via :meth:`Mamba2SSMStateCache.update_states` so step-mode
      decode resumes from prefill's exact final state.
    * :class:`Mamba2Step` — single-token forward consuming the per-slot
      ``(conv_state, ssm_state)`` tuples and producing updated states
      that are stored back into the cache.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
        )
        self._prefill_model, self._step_model = self._load_models(session)
        self._ssm_cache = self._create_ssm_cache()
        self.logprobs_device = devices[0]
        self.logprobs_model = self._load_logprobs_model(session)

        # SSM and conv state are both seeded across the prefill -> step
        # boundary. The SSD op exposes ``final_state`` (SSM half); the
        # conv half is sliced Python-side from the last ``d_conv - 1``
        # pre-conv xBC tokens.

    # -- cache setup ----------------------------------------------------

    def _create_ssm_cache(self) -> Mamba2SSMStateCache:
        """Pre-allocate per-slot conv/ssm state buffers from the model config."""
        cfg = self._model_config
        max_slots = self.pipeline_config.runtime.max_batch_size or 1
        return Mamba2SSMStateCache(
            num_layers=cfg.n_layer,
            conv_dim=cfg.conv_dim,
            d_conv=cfg.d_conv,
            nheads=cfg.nheads,
            head_dim=cfg.headdim,
            d_state=cfg.d_state,
            dtype=self.dtype,
            max_slots=max_slots,
            device=self.devices[0],
        )

    # -- interface contract ---------------------------------------------

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        n_layer = getattr(
            huggingface_config, "num_hidden_layers", None
        ) or getattr(huggingface_config, "n_layer", None)
        if n_layer is None:
            raise ValueError(
                "HF config is missing both `num_hidden_layers` and `n_layer`"
            )
        return int(n_layer)

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Return the configured maximum sequence length.

        Mamba2 has no positional embeddings, so any reasonable upper
        bound works. We mirror Mamba1's policy: respect a configured
        ``max_position_embeddings`` if the checkpoint ships one, else
        default to 2048 (matches the reference's default chunk_size
        sweep ceiling and is generous enough that step-mode generation
        is the only limiter in practice).
        """
        from max.pipelines.lib import upper_bounded_default

        model_config = pipeline_config.model
        try:
            return upper_bounded_default(
                upper_bound=getattr(
                    huggingface_config, "max_position_embeddings", 2048
                ),
                default=model_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Mamba2; "
                f"max_length ({model_config.max_length}) exceeds "
                f"max_position_embeddings "
                f"({getattr(huggingface_config, 'max_position_embeddings', 2048)})."
            ) from e

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Return minimal dummy KV cache params (SSM cache lives separately).

        Mirrors Phase-1: Mamba2 uses :class:`Mamba2SSMStateCache` for the
        real per-request state. The dummy KV params satisfy the
        :class:`PipelineModelWithKVCache` interface with negligible
        memory overhead so the standard scheduler keeps working.
        """
        return KVCacheParams(
            dtype=cache_dtype or DType.float32,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            devices=devices,
            page_size=128,
        )

    @classmethod
    def estimate_activation_memory(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserve GPU memory for the per-request SSM state pool.

        Mirrors qwen3_5's :class:`GatedDeltaNetStateCache` pattern. The
        SSM cache holds ``num_layers`` x ``(conv_state + ssm_state)``
        per slot:

        * ``conv_state``: ``conv_dim * (d_conv - 1)`` elements
        * ``ssm_state``:  ``nheads * head_dim * d_state`` elements

        The pipeline's default ``estimate_activation_memory`` returns 0,
        which leaves the SSM pool unaccounted-for and breaks the cache
        allocator's budget. By surfacing the pool size as activation
        memory, the rest of the pipeline plumbing (max_length sizing,
        OOM protection, the dummy KV cache) all see consistent numbers.
        """
        cfg = Mamba2Config.from_hf_config(huggingface_config)
        max_batch = pipeline_config.runtime.max_batch_size or 1

        encoding = pipeline_config.model.quantization_encoding
        if encoding is not None:
            from max.pipelines.lib import supported_encoding_dtype

            state_dtype = supported_encoding_dtype(encoding)
        else:
            state_dtype = DType.float32
        dtype_bytes = state_dtype.size_in_bytes

        conv_state_elems = cfg.conv_dim * (cfg.d_conv - 1)
        ssm_state_elems = cfg.nheads * cfg.headdim * cfg.d_state
        per_layer_bytes = (conv_state_elems + ssm_state_elems) * dtype_bytes
        return max_batch * cfg.n_layer * per_layer_bytes

    # -- compilation ----------------------------------------------------

    def _load_logprobs_model(self, session: InferenceSession) -> Model:
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    def _build_model_config(self) -> Mamba2Config:
        """Bridge the HF config to a :class:`Mamba2Config`.

        :class:`Mamba2Config` is a self-contained shape dataclass (it
        doesn't subclass :class:`ArchConfigWithKVCache`), so we use its
        :meth:`from_hf_config` loader rather than a Phase-1-style
        ``initialize`` / ``finalize`` pair. The pipeline config's
        ``max_length`` plumbing is honored by
        :meth:`calculate_max_seq_len` on the model side, so the dataclass
        intentionally does not carry a ``max_seq_len`` field.
        """
        return Mamba2Config.from_hf_config(self.huggingface_config)

    @traced
    def _load_models(self, session: InferenceSession) -> tuple[Any, Any]:
        from max.experimental import functional as F
        from max.experimental.tensor import default_dtype
        from max.graph import TensorType
        from max.pipelines.lib import CompilationTimer

        from .mamba2_module import Mamba2Prefill, Mamba2Step

        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Build the Mamba2 config from the HF AutoConfig. The weight
        # adapter consumes the same dataclass so prefill and step
        # compile against an identical view of the shapes.
        model_config = self._build_model_config()
        self._model_config = model_config

        # Parse the safetensor state dict via the standard utility, then
        # run it through the Mamba2 adapter so the MAX-side names match
        # what `Mamba2Prefill` / `Mamba2Step` expect.
        raw_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )
        # The standard utility may have already invoked an adapter; if
        # the caller didn't pass one, fall through to our adapter.
        if self.adapter is None:
            # raw_state_dict here is just the safetensor view — feed it
            # through the Mamba2 adapter.
            state_dict = convert_mamba2_state_dict(
                cast(Any, self.weights), model_config
            )
        else:
            state_dict = raw_state_dict

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        # Stash shape values for step-mode TensorType construction.
        num_layers = model_config.n_layer
        conv_dim = model_config.conv_dim
        d_conv = model_config.d_conv
        nheads = model_config.nheads
        head_dim = model_config.headdim
        d_state = model_config.d_state

        kernel_paths = list(_get_state_space_paths())

        # --- Compile prefill model ---
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        row_offsets_type = TensorType(
            DType.uint32, shape=["row_offsets_len"], device=device_ref
        )

        with CompilationTimer("prefill") as timer:
            with F.lazy(), default_dtype(self.dtype):
                prefill_module = Mamba2Prefill(model_config)
                prefill_module.to(device0)

            timer.mark_build_complete()
            prefill_model = prefill_module.compile(
                tokens_type,
                row_offsets_type,
                weights=state_dict,
                custom_extensions=kernel_paths,
            )

        # --- Compile step model ---
        step_tokens_type = TensorType(
            DType.int64, shape=["batch"], device=device_ref
        )
        layer_state_types: list[TensorType] = []
        for _ in range(num_layers):
            # conv_state: [batch, conv_dim, d_conv - 1]
            layer_state_types.append(
                TensorType(
                    self.dtype,
                    shape=["batch", conv_dim, d_conv - 1],
                    device=device_ref,
                )
            )
            # ssm_state: [batch, nheads, head_dim, d_state]
            layer_state_types.append(
                TensorType(
                    self.dtype,
                    shape=["batch", nheads, head_dim, d_state],
                    device=device_ref,
                )
            )

        with CompilationTimer("step") as timer:
            with F.lazy(), default_dtype(self.dtype):
                step_module = Mamba2Step(model_config)
                step_module.to(device0)

            timer.mark_build_complete()
            step_model = step_module.compile(
                step_tokens_type,
                *layer_state_types,
                weights=state_dict,
                custom_extensions=kernel_paths,
            )

        return prefill_model, step_model

    # -- execution ------------------------------------------------------

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Mamba2ModelInputs)

        if model_inputs.is_prefill:
            outputs = self._prefill_model(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
            )
        else:
            outputs = self._step_model(
                model_inputs.tokens,
                *model_inputs.layer_states,
            )

        # First output is logits. The remaining outputs are interleaved
        # ``[conv_0, ssm_0, conv_1, ssm_1, ...]`` for both prefill and
        # step paths, so a single ``update_states`` call seeds the cache
        # in either mode.
        logits = cast(Buffer, outputs[0].driver_tensor)
        new_states = [s.driver_tensor for s in outputs[1:]]
        if model_inputs.request_ids and new_states:
            self._ssm_cache.update_states(model_inputs.request_ids, new_states)

        # DEBUG(parity): dump prefill logits to disk for HF comparison.
        import os as _os

        dump_path = _os.environ.get("MAMBA2_DUMP_PREFILL_LOGITS")
        if dump_path and model_inputs.is_prefill:
            import numpy as _np

            _np.save(dump_path, logits.to_numpy())
            logger.info(
                f"MAMBA2_DUMP_PREFILL_LOGITS: wrote logits shape "
                f"{logits.to_numpy().shape} to {dump_path}"
            )

        return ModelOutputs(
            logits=logits,
            next_token_logits=logits,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Mamba2ModelInputs:
        if len(replica_batches) != 1:
            raise ValueError("Mamba2 does not support DP>1")

        context_batch = replica_batches[0]
        request_ids = [ctx.request_id for ctx in context_batch]

        for rid in request_ids:
            self._ssm_cache.claim(rid)

        input_row_offsets = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        tokens_buf = Buffer.from_numpy(tokens).to(self.devices[0])
        offsets_buf = Buffer.from_numpy(input_row_offsets).to(self.devices[0])
        n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        # If any request already has computed states (continuation), use
        # step mode. Mirrors Phase-1: SSM state is not reconstructable
        # from tokens alone — it must be carried forward.
        has_existing_states = any(
            self._ssm_cache.contains(rid)
            and self._ssm_cache.has_valid_state(rid)
            for rid in request_ids
        )

        if has_existing_states:
            layer_states = self._ssm_cache.get_states(request_ids)
            inputs = Mamba2ModelInputs(
                tokens_buf,
                offsets_buf,
                n_logits_buf,
                is_prefill=False,
                layer_states=layer_states,
                request_ids=request_ids,
            )
            inputs.kv_cache_inputs = kv_cache_inputs
            return inputs

        inputs = Mamba2ModelInputs(
            tokens_buf,
            offsets_buf,
            n_logits_buf,
            is_prefill=True,
            request_ids=request_ids,
        )
        inputs.kv_cache_inputs = kv_cache_inputs
        return inputs

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Mamba2ModelInputs:
        prev = cast(Mamba2ModelInputs, prev_model_inputs)
        layer_states = self._ssm_cache.get_states(prev.request_ids)

        inputs = Mamba2ModelInputs(
            tokens=next_tokens,
            input_row_offsets=prev.input_row_offsets,
            return_n_logits=prev.return_n_logits,
            is_prefill=False,
            layer_states=layer_states,
            request_ids=prev.request_ids,
        )
        inputs.kv_cache_inputs = prev.kv_cache_inputs
        return inputs

    def release(self, request_id: RequestID) -> None:
        """Release SSM cache slot when a request completes."""
        self._ssm_cache.release(request_id)

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        logits = model_outputs.logits
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, Mamba2ModelInputs)

        sampled_tokens = next_tokens.to_numpy()
        tokens = model_inputs.tokens.to_numpy()
        input_row_offsets = model_inputs.input_row_offsets.to_numpy()

        return compute_log_probabilities_ragged(
            self.logprobs_device,
            self.logprobs_model,
            input_row_offsets=input_row_offsets,
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=tokens,
            sampled_tokens=sampled_tokens,
            batch_top_n=batch_top_n,
            batch_echo=batch_echo,
        )
