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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from max.graph.weights import WeightsFormat
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    upper_bounded_default,
)
from max.pipelines.modeling.types import PipelineTask
from typing_extensions import Self

from . import weight_adapters
from .model import Mamba2Model
from .model_config import Mamba2Config
from .tokenizer import Mamba2Tokenizer

if TYPE_CHECKING:
    from max.nn.kv_cache.cache_params import KVCacheParamInterface
    from max.pipelines.lib import PipelineConfig
    from max.pipelines.lib.config.model_config import MAXModelConfig


@dataclass
class _Mamba2ArchConfig(Mamba2Config):
    """Registry-facing :class:`Mamba2Config` that satisfies :class:`ArchConfig`.

    :class:`Mamba2Config` is intentionally a pure shape dataclass and does
    not implement the :class:`~max.pipelines.lib.interfaces.ArchConfig`
    protocol (full pipeline-config integration lands in a later RFC 0003
    item). The registry, however, calls ``arch.config.initialize()`` and
    ``arch_config.get_max_seq_len()`` to size the tokenizer. This thin
    subclass bridges that gap by delegating to
    :meth:`Mamba2Config.from_hf_config` for initialization and to the
    HF-config's ``max_position_embeddings`` (capped by the user's
    ``--max-length``) for the sequence-length estimate, matching the
    policy implemented by :meth:`Mamba2Model.calculate_max_seq_len`.
    """

    # Cached max sequence length plumbed through from `initialize` so
    # `get_max_seq_len` can return it without re-reading the pipeline
    # config. Not part of the shape-config surface — kept as a plain
    # field with a sentinel default for mypy.
    _max_seq_len: int = 0

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        """Build the arch config from a :class:`PipelineConfig`.

        Reads the HF config off ``model_config.huggingface_config`` (or
        ``pipeline_config.model.huggingface_config`` when no explicit
        model config is given) and maps it through
        :meth:`Mamba2Config.from_hf_config`. The configured max length is
        capped against ``max_position_embeddings`` (default 2048) — the
        same policy :meth:`Mamba2Model.calculate_max_seq_len` uses on the
        model side.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but config could not be "
                "loaded."
            )

        shape_cfg = Mamba2Config.from_hf_config(huggingface_config)

        try:
            max_seq_len = upper_bounded_default(
                upper_bound=getattr(
                    huggingface_config, "max_position_embeddings", 2048
                ),
                default=model_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Mamba2; "
                f"max_length ({model_config.max_length}) exceeds "
                "max_position_embeddings "
                f"({getattr(huggingface_config, 'max_position_embeddings', 2048)})."
            ) from e

        # Construct via the subclass with all of `Mamba2Config`'s
        # required fields. The derived fields (`d_inner`, `nheads`, ...)
        # are recomputed by `__post_init__` so they match `shape_cfg`.
        return cls(
            d_model=shape_cfg.d_model,
            n_layer=shape_cfg.n_layer,
            vocab_size=shape_cfg.vocab_size,
            pad_vocab_size_multiple=shape_cfg.pad_vocab_size_multiple,
            d_state=shape_cfg.d_state,
            d_conv=shape_cfg.d_conv,
            expand=shape_cfg.expand,
            headdim=shape_cfg.headdim,
            ngroups=shape_cfg.ngroups,
            chunk_size=shape_cfg.chunk_size,
            rms_norm_eps=shape_cfg.rms_norm_eps,
            residual_in_fp32=shape_cfg.residual_in_fp32,
            fused_add_norm=shape_cfg.fused_add_norm,
            tie_embeddings=shape_cfg.tie_embeddings,
            use_bias=shape_cfg.use_bias,
            use_conv_bias=shape_cfg.use_conv_bias,
            A_init_range=shape_cfg.A_init_range,
            dt_min=shape_cfg.dt_min,
            dt_max=shape_cfg.dt_max,
            dt_init_floor=shape_cfg.dt_init_floor,
            _max_seq_len=int(max_seq_len),
        )

    def get_max_seq_len(self) -> int:
        """Maximum sequence length plumbed from the pipeline config."""
        return self._max_seq_len

    def get_kv_params(self) -> KVCacheParamInterface:
        """Dummy KV-cache params so the pipeline allocator has a budget.

        Mamba2's real per-request state lives in :class:`Mamba2SSMStateCache`,
        not in a paged KV cache. But the pipeline's memory estimator runs
        ``_calculate_kv_cache_size`` only when the arch config satisfies
        :class:`ArchConfigWithKVCache`; otherwise it returns 0 and
        ``load_kv_manager`` then refuses to allocate a single page.
        Mirroring Mamba1's :class:`MambaConfig`, we expose a minimal stub
        here (n_kv_heads=1, head_dim=1, num_layers=1) so the allocator
        reserves a few KiB and the rest of the pipeline plumbing works.
        The actual SSM state pool is sized via
        :meth:`Mamba2Model.estimate_activation_memory` (the
        GatedDeltaNetStateCache pattern from qwen3_5).
        """
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn.kv_cache import KVCacheParams

        return KVCacheParams(
            dtype=DType.float32,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            devices=[DeviceRef.CPU()],
            page_size=128,
        )


mamba2_arch = SupportedArchitecture(
    name="Mamba2ForCausalLM",
    example_repo_ids=[
        "state-spaces/mamba2-130m",
        "state-spaces/mamba2-370m",
        "state-spaces/mamba2-780m",
        "state-spaces/mamba2-1.3b",
        "state-spaces/mamba2-2.7b",
    ],
    default_encoding="float32",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=Mamba2Model,
    tokenizer=Mamba2Tokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_mamba2_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=_Mamba2ArchConfig,
)
