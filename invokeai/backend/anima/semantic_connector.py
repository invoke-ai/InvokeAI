"""Anima-3.8B's semantic connector: Qwen3.5 conditioning on top of Anima's native LLM adapter.

Anima-3.8B v1.1 bundles this connector into its checkpoint under `anima_v2_connector.`. It replaces
the plain `LLMAdapter` call: the native adapter's six blocks still run, unchanged and in order, and
after each one two residual cross-attentions add Qwen3.5 information --

- the *quality anchor* attends to a per-block mix of four Qwen3.5 hidden layers (7, 15, 23, 31);
- the *v2 injection* attends to a 64-token bank that a timestep-aware perceiver resampler distills
  from the same four layers.

The resampler is conditioned on the diffusion timestep, so unlike the native adapter's output the
connector's output changes from step to step and has to be recomputed inside the denoising loop.

Ported from the reference ComfyUI extension (https://github.com/GumGum10/comfyui-anima-3-8B, MIT),
`semantic_connector_v2.py` and `progressive_cross_adapter.py`. Module and parameter names follow
the checkpoint so the bundled weights load without a key map. Training-only machinery
(initialization, trainability, gradient checkpointing) is left out.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from invokeai.backend.anima.anima_transformer import LLMAdapter, LLMAdapterAttention, masked_sdpa

#: The Qwen3.5 4B hidden layers the connector was trained on.
QWEN35_LAYER_INDICES: tuple[int, ...] = (7, 15, 23, 31)
#: Qwen3.5 4B hidden size.
QWEN35_HIDDEN_SIZE = 2560


@dataclass(frozen=True)
class AnimaSemanticConnectorConfig:
    """The connector's hyperparameters. The bundle records them in its header (see `from_metadata`)."""

    num_queries: int = 64
    resampler_blocks: int = 6
    resampler_dim: int = 2048
    resampler_heads: int = 16
    mlp_hidden_dim: int = 5632
    semantic_source_dim: int = QWEN35_HIDDEN_SIZE
    layer_indices: tuple[int, ...] = QWEN35_LAYER_INDICES

    #: The architecture name the bundle declares for the connector it carries.
    ARCHITECTURE = "anima_qwen35_quality_anchored_semantic_connector_v2"

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "AnimaSemanticConnectorConfig":
        """Read the hyperparameters from a v1.1 bundle's safetensors header.

        Raises:
            ValueError: if the header declares a connector architecture other than the one ported here.
        """
        architecture = metadata.get("anima_v2_adapter_architecture")
        if architecture is not None and architecture != cls.ARCHITECTURE:
            raise ValueError(f"Unsupported Anima semantic connector architecture {architecture!r}.")

        def number(name: str, default: int) -> int:
            return int(metadata.get(f"anima_v2_adapter_{name}", default))

        layer_indices = QWEN35_LAYER_INDICES
        if (raw := metadata.get("anima_v2_adapter_layer_indices")) is not None:
            layer_indices = tuple(int(v) for v in raw.strip("[]").split(",") if v.strip())
        return cls(
            num_queries=number("semantic_query_tokens", cls.num_queries),
            resampler_blocks=number("semantic_resampler_blocks", cls.resampler_blocks),
            resampler_dim=number("semantic_resampler_dim", cls.resampler_dim),
            resampler_heads=number("semantic_resampler_heads", cls.resampler_heads),
            mlp_hidden_dim=number("semantic_resampler_mlp_hidden_dim", cls.mlp_hidden_dim),
            layer_indices=layer_indices,
        )


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Embed Anima's continuous [0, 1] flow timestep. Scaled by 1000 like the reference."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    angles = timesteps.float().reshape(-1, 1) * 1_000.0 * frequencies.reshape(1, -1)
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResamplerAttention(nn.Module):
    """Plain multi-head attention with independently sized query and context streams. No norms, no RoPE."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, query_dim, bias=False)

    def forward(
        self, query: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, query_tokens, query_dim = query.shape
        context_tokens = context.shape[1]

        def heads(value: torch.Tensor, tokens: int) -> torch.Tensor:
            return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = heads(self.q_proj(query), query_tokens)
        k = heads(self.k_proj(context), context_tokens)
        v = heads(self.v_proj(context), context_tokens)
        mask = None
        if context_mask is not None:
            mask = context_mask.to(torch.bool).reshape(batch, 1, 1, context_tokens)
        attended = masked_sdpa(q, k, v, mask)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, query_tokens, query_dim))


class TimestepModulatedNorm(nn.Module):
    """Parameter-free LayerNorm whose scale and shift come from the timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, value: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return self.norm(value) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SemanticResamplerBlock(nn.Module):
    """One timestep-aware block: cross-attention to Qwen3.5, self-attention among queries, SwiGLU."""

    def __init__(self, dim: int, qwen_dim: int, num_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.cross_norm = TimestepModulatedNorm(dim)
        self.self_norm = TimestepModulatedNorm(dim)
        self.mlp_norm = TimestepModulatedNorm(dim)
        self.source_norm = nn.LayerNorm(qwen_dim)
        self.cross_attention = ResamplerAttention(dim, qwen_dim, num_heads)
        self.self_attention = ResamplerAttention(dim, dim, num_heads)
        self.mlp_in = nn.Linear(dim, 2 * mlp_hidden_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, dim, bias=False)
        # Scale and shift for each of the three sub-blocks.
        self.time_modulation = nn.Linear(dim, 6 * dim, bias=True)

    def forward(
        self,
        queries: torch.Tensor,
        qwen_features: torch.Tensor,
        timestep_embedding: torch.Tensor,
        qwen_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        modulation = self.time_modulation(F.silu(timestep_embedding))
        cross_scale, cross_shift, self_scale, self_shift, mlp_scale, mlp_shift = modulation.chunk(6, dim=-1)
        queries = queries + self.cross_attention(
            self.cross_norm(queries, cross_scale, cross_shift), self.source_norm(qwen_features), qwen_mask
        )
        normalized = self.self_norm(queries, self_scale, self_shift)
        queries = queries + self.self_attention(normalized, normalized)
        normalized = self.mlp_norm(queries, mlp_scale, mlp_shift)
        gate, value = self.mlp_in(normalized).chunk(2, dim=-1)
        return queries + self.mlp_out(F.silu(gate) * value)


class TimestepAwareSemanticResampler(nn.Module):
    """Compresses four Qwen3.5 layer streams into a bank of semantic query tokens, per timestep."""

    def __init__(
        self,
        qwen_dim: int,
        output_dim: int,
        num_layers: int,
        num_queries: int,
        num_blocks: int,
        model_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, model_dim))
        self.layer_embeddings = nn.Parameter(torch.empty(num_layers, 1, qwen_dim))
        self.time_mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.blocks = nn.ModuleList(
            [SemanticResamplerBlock(model_dim, qwen_dim, num_heads, mlp_hidden_dim) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim, bias=False)

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        timesteps: torch.Tensor,
        source_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} Qwen3.5 layers, got {len(hidden_states)}")
        batch = hidden_states[0].shape[0]
        qwen_features = torch.cat(
            [hidden + self.layer_embeddings[index].unsqueeze(0) for index, hidden in enumerate(hidden_states)], dim=1
        )
        qwen_mask = torch.cat([source_mask] * self.num_layers, dim=1) if source_mask is not None else None

        time = sinusoidal_timestep_embedding(timesteps, self.model_dim).to(dtype=qwen_features.dtype)
        time = self.time_mlp(time)
        queries = self.query_tokens.expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(queries, qwen_features, time, qwen_mask)
        return self.output_projection(self.output_norm(queries))


class QualityAnchor(nn.Module):
    """The frozen v1 cross-attentions to Qwen3.5 that the v2 connector was trained on top of.

    One residual cross-attention per native adapter block, each reading its own learned softmax mix
    of the four Qwen3.5 layers (`layer_mix_logits`).
    """

    def __init__(self, model_dim: int, num_heads: int, num_blocks: int, semantic_source_dim: int, num_layers: int):
        super().__init__()
        head_dim = model_dim // num_heads
        self.query_norms = nn.ModuleList([nn.RMSNorm(model_dim, eps=1e-6) for _ in range(num_blocks)])
        self.source_norms = nn.ModuleList([nn.RMSNorm(semantic_source_dim, eps=1e-6) for _ in range(num_blocks)])
        self.semantic_attentions = nn.ModuleList(
            [LLMAdapterAttention(model_dim, semantic_source_dim, num_heads, head_dim) for _ in range(num_blocks)]
        )
        self.layer_mix_logits = nn.Parameter(torch.zeros(num_blocks, num_layers))

    @staticmethod
    def mixed_source(hidden_states: Sequence[torch.Tensor], mix: torch.Tensor, block_index: int) -> torch.Tensor:
        return sum(hidden * mix[block_index, layer_index] for layer_index, hidden in enumerate(hidden_states))  # type: ignore[return-value]


class AnimaSemanticConnector(nn.Module):
    """Quality-anchored semantic connector v2. Wraps a native `LLMAdapter` it does not own.

    The adapter is passed to `forward` instead of being held: it belongs to the transformer, which
    patches it with LoRAs and moves it between devices, and a second registration would serialize and
    move it twice.
    """

    def __init__(
        self,
        config: AnimaSemanticConnectorConfig,
        model_dim: int = 1024,
        num_heads: int = 16,
        num_adapter_blocks: int = 6,
    ):
        super().__init__()
        self.config = config
        num_layers = len(config.layer_indices)
        head_dim = model_dim // num_heads
        self.quality_anchor = QualityAnchor(
            model_dim, num_heads, num_adapter_blocks, config.semantic_source_dim, num_layers
        )
        self.semantic_resampler = TimestepAwareSemanticResampler(
            qwen_dim=config.semantic_source_dim,
            output_dim=model_dim,
            num_layers=num_layers,
            num_queries=config.num_queries,
            num_blocks=config.resampler_blocks,
            model_dim=config.resampler_dim,
            num_heads=config.resampler_heads,
            mlp_hidden_dim=config.mlp_hidden_dim,
        )
        self.v2_query_norms = nn.ModuleList([nn.RMSNorm(model_dim, eps=1e-6) for _ in range(num_adapter_blocks)])
        self.v2_semantic_norms = nn.ModuleList([nn.RMSNorm(model_dim, eps=1e-6) for _ in range(num_adapter_blocks)])
        self.v2_attentions = nn.ModuleList(
            [LLMAdapterAttention(model_dim, model_dim, num_heads, head_dim) for _ in range(num_adapter_blocks)]
        )

    @staticmethod
    def _attention_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = mask.to(torch.bool)
        return mask[:, None, None, :] if mask.ndim == 2 else mask

    def forward(
        self,
        native_adapter: LLMAdapter,
        native_source: torch.Tensor,
        target_input_ids: torch.Tensor,
        semantic_hidden_states: Sequence[torch.Tensor],
        semantic_mask: Optional[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Produce the DiT's cross-attention context for one denoising step.

        Args:
            native_adapter: The transformer's own `LLMAdapter`.
            native_source: Qwen3 0.6B hidden states. Shape: (B, L_qwen3, 1024).
            target_input_ids: T5-XXL token IDs. Shape: (B, L_t5).
            semantic_hidden_states: Qwen3.5 hidden states, one per entry of `config.layer_indices`.
                Each of shape (B, L_qwen35, 2560).
            semantic_mask: True for valid Qwen3.5 tokens. Shape: (B, L_qwen35). A fully masked row
                (the reference's encoding of an empty prompt) contributes nothing.
            timesteps: The flow timestep (sigma). Shape: (B,).

        Returns:
            Context of shape (B, L_t5, 1024), before Anima's padding to 512 tokens.
        """
        if len(semantic_hidden_states) != len(self.config.layer_indices):
            raise ValueError(
                f"Expected {len(self.config.layer_indices)} Qwen3.5 layers, got {len(semantic_hidden_states)}"
            )
        semantic_attention_mask = self._attention_mask(semantic_mask)
        semantic_bank = self.semantic_resampler(semantic_hidden_states, timesteps, semantic_mask)

        x = native_adapter.embed(target_input_ids).to(dtype=native_source.dtype)
        rotary = native_adapter.rotary_emb

        def positions(length: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return rotary(x, torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(0))

        query_rope = positions(x.shape[1])
        native_rope = positions(native_source.shape[1])
        anchor_rope = positions(semantic_hidden_states[0].shape[1])
        bank_rope = positions(semantic_bank.shape[1])
        anchor_mix = self.quality_anchor.layer_mix_logits.float().softmax(dim=-1).to(x.dtype)

        anchor = self.quality_anchor
        for index, native_block in enumerate(native_adapter.blocks):
            x = native_block(x, context=native_source, pos_target=query_rope, pos_source=native_rope)
            anchor_source = anchor.source_norms[index](anchor.mixed_source(semantic_hidden_states, anchor_mix, index))
            x = x + anchor.semantic_attentions[index](
                anchor.query_norms[index](x),
                context=anchor_source,
                attn_mask=semantic_attention_mask,
                pos_q=query_rope,
                pos_k=anchor_rope,
            )
            x = x + self.v2_attentions[index](
                self.v2_query_norms[index](x),
                context=self.v2_semantic_norms[index](semantic_bank),
                pos_q=query_rope,
                pos_k=bank_rope,
            )

        return native_adapter.norm(native_adapter.out_proj(x))
