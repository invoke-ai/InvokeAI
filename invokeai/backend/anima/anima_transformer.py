"""Anima transformer model: Cosmos Predict2 MiniTrainDIT + LLM Adapter.

The Anima architecture combines:
1. MiniTrainDIT: A Cosmos Predict2 DiT backbone with 28 blocks, 2048-dim hidden state,
   and 3D RoPE positional embeddings.
2. LLMAdapter: A 6-layer cross-attention transformer that fuses Qwen3 0.6B hidden states
   with learned T5-XXL token embeddings to produce conditioning for the DiT.

Original source code:
- MiniTrainDIT backbone and positional embeddings: https://github.com/nvidia-cosmos/cosmos-predict2
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
- LLMAdapter and Anima wrapper: Clean-room implementation based on
  https://github.com/hdae/diffusers-anima (Apache-2.0)
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

logger = logging.getLogger(__name__)


# ============================================================================
# Positional Embeddings
# Original source: https://github.com/nvidia-cosmos/cosmos-predict2
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. Apache-2.0
# ============================================================================


class VideoRopePosition3DEmb(nn.Module):
    """3D Rotary Position Embedding for video/image transformers.

    Generates rotary embeddings with separate frequency components for
    height, width, and temporal dimensions.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(
        self,
        x_B_T_H_W_C: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps, device=device)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        B, T, H, W, _ = B_T_H_W_C
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)

        half_emb_h = torch.outer(seq[:H].to(device=device), h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W].to(device=device), w_spatial_freqs)

        if fps is None or self.enable_fps_modulation is False:
            half_emb_t = torch.outer(seq[:T].to(device=device), temporal_freqs)
        else:
            half_emb_t = torch.outer(seq[:T].to(device=device) / fps * self.base_fps, temporal_freqs)

        half_emb_h = torch.stack(
            [torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1
        )
        half_emb_w = torch.stack(
            [torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1
        )
        half_emb_t = torch.stack(
            [torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1
        )

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


def _normalize(x: torch.Tensor, dim: Optional[list[int]] = None, eps: float = 0) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class LearnablePosEmbAxis(nn.Module):
    """Learnable per-axis positional embeddings."""

    def __init__(
        self,
        *,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.pos_emb_h = nn.Parameter(torch.empty(len_h, model_channels, device=device, dtype=dtype))
        self.pos_emb_w = nn.Parameter(torch.empty(len_w, model_channels, device=device, dtype=dtype))
        self.pos_emb_t = nn.Parameter(torch.empty(len_t, model_channels, device=device, dtype=dtype))

    def forward(
        self,
        x_B_T_H_W_C: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return self.generate_embeddings(x_B_T_H_W_C.shape, device=device, dtype=dtype)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        B, T, H, W, _ = B_T_H_W_C
        emb_h_H = self.pos_emb_h[:H].to(device=device, dtype=dtype)
        emb_w_W = self.pos_emb_w[:W].to(device=device, dtype=dtype)
        emb_t_T = self.pos_emb_t[:T].to(device=device, dtype=dtype)
        emb = (
            repeat(emb_t_T, "t d -> b t h w d", b=B, h=H, w=W)
            + repeat(emb_h_H, "h d -> b t h w d", b=B, t=T, w=W)
            + repeat(emb_w_W, "w d -> b t h w d", b=B, t=T, h=H)
        )
        return _normalize(emb, dim=-1, eps=1e-6)


# ============================================================================
# Cosmos Predict2 MiniTrainDIT
# Original source: https://github.com/nvidia-cosmos/cosmos-predict2
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. Apache-2.0
# ============================================================================


def apply_rotary_pos_emb_cosmos(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings in Cosmos format (2x2 rotation matrices)."""
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class CosmosAttention(nn.Module):
    """Multi-head attention for the Cosmos DiT backbone.

    Supports both self-attention and cross-attention with QK normalization
    and rotary position embeddings.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = (rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim) for t in (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_selfattn and rope_emb is not None:
            q = apply_rotary_pos_emb_cosmos(q, rope_emb)
            k = apply_rotary_pos_emb_cosmos(k, rope_emb)

        # Reshape for scaled_dot_product_attention: (B, heads, seq, dim)
        in_q_shape = q.shape
        in_k_shape = k.shape
        q = rearrange(q, "b ... h d -> b h ... d").reshape(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
        k = rearrange(k, "b ... h d -> b h ... d").reshape(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        v = rearrange(v, "b ... h d -> b h ... d").reshape(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])

        result = F.scaled_dot_product_attention(q, k, v)
        result = rearrange(result, "b h s d -> b s (h d)")
        return self.output_dropout(self.output_proj(result))


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    """Projects sinusoidal timestep embeddings to model dimension."""

    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

    def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    """Patchify input tensor via rearrange + linear projection."""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        return self.proj(x)


class FinalLayer(nn.Module):
    """Final AdaLN-modulated output projection."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.use_adaln_lora = use_adaln_lora

        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 2 * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift, scale = (self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]).chunk(
                2, dim=-1
            )
        else:
            shift, scale = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")

        x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        return self.linear(x_B_T_H_W_D)


class DiTBlock(nn.Module):
    """Cosmos DiT transformer block with self-attention, cross-attention, and MLP.

    Each component uses AdaLN (Adaptive Layer Normalization) modulation from
    the timestep embedding.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.use_adaln_lora = use_adaln_lora

        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = CosmosAttention(x_dim, None, num_heads, x_dim // num_heads)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CosmosAttention(x_dim, context_dim, num_heads, x_dim // num_heads)

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        # AdaLN modulation layers (shift, scale, gate for each of 3 components)
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype

        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        # Compute AdaLN modulations
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        shift_sa, scale_sa, gate_sa = (rearrange(t, "b t d -> b t 1 1 d") for t in (shift_sa, scale_sa, gate_sa))
        shift_ca, scale_ca, gate_ca = (rearrange(t, "b t d -> b t 1 1 d") for t in (shift_ca, scale_ca, gate_ca))
        shift_mlp, scale_mlp, gate_mlp = (rearrange(t, "b t d -> b t 1 1 d") for t in (shift_mlp, scale_mlp, gate_mlp))

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln(x: torch.Tensor, norm: nn.Module, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return norm(x) * (1 + scale) + shift

        # Self-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        result = rearrange(
            self.self_attn(
                rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"), None, rope_emb=rope_emb_L_1_1_D
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa.to(residual_dtype) * result.to(residual_dtype)

        # Cross-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        result = rearrange(
            self.cross_attn(
                rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result.to(residual_dtype) * gate_ca.to(residual_dtype) + x_B_T_H_W_D

        # MLP
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        result = self.mlp(normed.to(compute_dtype))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp.to(residual_dtype) * result.to(residual_dtype)

        return x_B_T_H_W_D


class MiniTrainDIT(nn.Module):
    """Cosmos Predict2 DiT backbone for video/image generation.

    This is the core transformer architecture that Anima extends. It processes
    3D latent tensors (B, C, T, H, W) with patch embedding, positional encoding,
    and adaptive layer normalization.

    Args:
        max_img_h: Maximum image height in pixels.
        max_img_w: Maximum image width in pixels.
        max_frames: Maximum number of video frames.
        in_channels: Number of input latent channels.
        out_channels: Number of output channels.
        patch_spatial: Spatial patch size.
        patch_temporal: Temporal patch size.
        concat_padding_mask: Whether to concatenate a padding mask channel.
        model_channels: Hidden dimension of the transformer.
        num_blocks: Number of DiT blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        crossattn_emb_channels: Cross-attention context dimension.
        use_adaln_lora: Whether to use AdaLN-LoRA.
        adaln_lora_dim: AdaLN-LoRA bottleneck dimension.
        extra_per_block_abs_pos_emb: Whether to use extra learnable positional embeddings.
    """

    def __init__(
        self,
        max_img_h: int = 240,
        max_img_w: int = 240,
        max_frames: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        image_model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb

        # Positional embeddings
        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_channels // num_heads,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            max_fps=max_fps,
            min_fps=min_fps,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
            enable_fps_modulation=rope_enable_fps_modulation,
        )

        if extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = LearnablePosEmbAxis(
                model_channels=model_channels,
                len_h=max_img_h // patch_spatial,
                len_w=max_img_w // patch_spatial,
                len_t=max_frames // patch_temporal,
            )

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        self.t_embedding_norm = nn.RMSNorm(model_channels, eps=1e-6)

        # Patch embedding
        embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=embed_in_channels,
            out_channels=model_channels,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final output layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )

    def _pad_to_patch_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input tensor so dimensions are divisible by patch sizes."""
        _, _, T, H, W = x.shape
        pad_t = (self.patch_temporal - T % self.patch_temporal) % self.patch_temporal
        pad_h = (self.patch_spatial - H % self.patch_spatial) % self.patch_spatial
        pad_w = (self.patch_spatial - W % self.patch_spatial) % self.patch_spatial
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        return x

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0],
                    1,
                    x_B_C_T_H_W.shape[3],
                    x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype,
                    device=x_B_C_T_H_W.device,
                )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        extra_pos_emb = None
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(
                x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype
            )

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device), extra_pos_emb

        return x_B_T_H_W_D, None, extra_pos_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        orig_shape = list(x.shape)
        x = self._pad_to_patch_size(x)

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x, fps=fps, padding_mask=padding_mask
        )

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype))
        t_emb = self.t_embedding_norm(t_emb)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0) if rope_emb_L_1_1_D is not None else None,
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
        }

        # Keep residual stream in fp32 for numerical stability with fp16 compute
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_emb, context, **block_kwargs)

        x_out = self.final_layer(x_B_T_H_W_D.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln_lora)
        x_out = self.unpatchify(x_out)[:, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]]
        return x_out


# ============================================================================
# LLM Adapter
# Reference implementation: https://github.com/hdae/diffusers-anima
# SPDX-License-Identifier: Apache-2.0
# ============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension in half and negate-swap: [-x2, x1]."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to tensor x given precomputed cos/sin."""
    return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))


class LLMAdapterRotaryEmbedding(nn.Module):
    """Rotary position embedding for the LLM Adapter's attention layers."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        half_dim = head_dim // 2
        index = torch.arange(half_dim, dtype=torch.float32)
        exponent = (2.0 / float(head_dim)) * index
        inv_freq = torch.reciprocal(torch.pow(torch.tensor(theta, dtype=torch.float32), exponent))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids.to(device=x.device, dtype=torch.float32)
        inv = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bl,d->bld", pos, inv)
        emb = freqs.repeat(1, 1, 2)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    """Attention for the LLM Adapter with QK normalization and rotary position embeddings."""

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, head_dim: int):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        pos_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_k: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        context = x if context is None else context

        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(context.shape[0], context.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(context.shape[0], context.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pos_q is not None and pos_k is not None:
            q = _apply_rope(q, *pos_q)
            k = _apply_rope(k, *pos_k)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1).contiguous()
        return self.o_proj(y)


class LLMAdapterTransformerBlock(nn.Module):
    """Single transformer block in the LLM Adapter.

    Each block contains self-attention, cross-attention, and MLP with
    RMSNorm pre-normalization.
    """

    def __init__(
        self,
        source_dim: int,
        model_dim: int,
        num_heads: int = 16,
    ):
        super().__init__()
        head_dim = model_dim // num_heads

        self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.self_attn = LLMAdapterAttention(model_dim, model_dim, num_heads, head_dim)

        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = LLMAdapterAttention(model_dim, source_dim, num_heads, head_dim)

        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        pos_target: Tuple[torch.Tensor, torch.Tensor],
        pos_source: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm_self_attn(x),
            attn_mask=target_mask,
            pos_q=pos_target,
            pos_k=pos_target,
        )
        x = x + self.cross_attn(
            self.norm_cross_attn(x),
            context=context,
            attn_mask=source_mask,
            pos_q=pos_target,
            pos_k=pos_source,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    """LLM Adapter: bridges Qwen3 hidden states and T5-XXL token embeddings.

    Takes Qwen3 hidden states and T5-XXL token IDs, produces conditioning
    embeddings for the Cosmos DiT via cross-attention through 6 transformer layers.

    Args:
        vocab_size: Size of the T5 token vocabulary.
        dim: Model dimension (used for embeddings, projections, and all layers).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        vocab_size: int = 32128,
        dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [LLMAdapterTransformerBlock(source_dim=dim, model_dim=dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.rotary_emb = LLMAdapterRotaryEmbedding(dim // num_heads)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Expand attention masks for multi-head attention
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask[:, None, None, :]

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask[:, None, None, :]

        context = source_hidden_states
        x = self.embed(target_input_ids).to(dtype=context.dtype)

        # Build position IDs and compute rotary embeddings
        target_pos_ids = torch.arange(x.shape[1], device=x.device, dtype=torch.long).unsqueeze(0)
        source_pos_ids = torch.arange(context.shape[1], device=x.device, dtype=torch.long).unsqueeze(0)
        pos_target = self.rotary_emb(x, target_pos_ids)
        pos_source = self.rotary_emb(x, source_pos_ids)

        for block in self.blocks:
            x = block(
                x,
                context=context,
                target_mask=target_attention_mask,
                source_mask=source_attention_mask,
                pos_target=pos_target,
                pos_source=pos_source,
            )
        return self.norm(self.out_proj(x))


# ============================================================================
# Anima: MiniTrainDIT + LLMAdapter
# Reference implementation: https://github.com/hdae/diffusers-anima
# SPDX-License-Identifier: Apache-2.0
# ============================================================================


class AnimaTransformer(MiniTrainDIT):
    """Anima transformer: Cosmos Predict2 DiT with integrated LLM Adapter.

    Extends MiniTrainDIT by adding the LLMAdapter component that preprocesses
    text embeddings before they are fed to the DiT cross-attention layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_adapter = LLMAdapter()

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: Optional[torch.Tensor],
        t5xxl_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the LLM Adapter to produce conditioning for the DiT.

        Args:
            text_embeds: Qwen3 hidden states. Shape: (batch, seq_len, 1024).
            text_ids: T5-XXL token IDs. Shape: (batch, seq_len). If None, returns text_embeds directly.
            t5xxl_weights: Optional per-token weights. Shape: (batch, seq_len, 1).

        Returns:
            Conditioning tensor. Shape: (batch, 512, 1024), zero-padded if needed.
        """
        if text_ids is None:
            return text_embeds
        out = self.llm_adapter(text_embeds, text_ids)
        if t5xxl_weights is not None:
            out = out * t5xxl_weights
        if out.shape[1] < 512:
            out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
        return out

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        t5xxl_ids: Optional[torch.Tensor] = None,
        t5xxl_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with LLM Adapter preprocessing.

        Args:
            x: Input latent tensor. Shape: (B, C, T, H, W).
            timesteps: Timestep values. Shape: (B,) or (B, T).
            context: Qwen3 hidden states. Shape: (B, seq_len, 1024).
            t5xxl_ids: T5-XXL token IDs. Shape: (B, seq_len).
            t5xxl_weights: Per-token weights. Shape: (B, seq_len, 1).

        Returns:
            Denoised output. Shape: (B, C, T, H, W).
        """
        if t5xxl_ids is not None:
            context = self.preprocess_text_embeds(context, t5xxl_ids, t5xxl_weights=t5xxl_weights)
        return super().forward(x, timesteps, context, **kwargs)
