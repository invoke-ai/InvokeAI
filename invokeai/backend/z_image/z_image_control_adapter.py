# Adapted from https://github.com/aigc-apps/VideoX-Fun/blob/main/videox_fun/models/z_image_transformer2d_control.py
# Copyright (c) Alibaba, Inc. and its affiliates.
# Apache License 2.0

"""
Z-Image Control Adapter for InvokeAI.

This module provides a standalone control adapter that can be combined with
a base ZImageTransformer2DModel at runtime. The adapter contains only the
control-specific layers (control_layers, control_all_x_embedder, control_noise_refiner).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_z_image import (
    ADALN_EMBED_DIM,
    SEQ_MULTI_OF,
    ZImageTransformerBlock,
)
from diffusers.utils import is_torch_version
from torch.nn.utils.rnn import pad_sequence


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    """Control-specific transformer block with skip connections for hint generation."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        block_id: int = 0,
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c: list[torch.Tensor] = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, attn_mask=attn_mask, freqs_cis=freqs_cis, adaln_input=adaln_input)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class ZImageControlAdapter(ModelMixin, ConfigMixin):
    """Standalone Z-Image Control Adapter.

    This adapter contains only the control-specific layers and can be combined
    with a base ZImageTransformer2DModel at runtime. It computes control hints
    that are added to the transformer's hidden states.

    The adapter supports 5 control modes: Canny, HED, Depth, Pose, MLSD.
    Recommended control_context_scale: 0.65-0.80.
    """

    @register_to_config
    def __init__(
        self,
        num_control_blocks: int = 6,  # Number of control layer blocks
        control_in_dim: int = 16,
        all_patch_size: tuple[int, ...] = (2,),
        all_f_patch_size: tuple[int, ...] = (1,),
        dim: int = 3840,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.control_in_dim = control_in_dim
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size

        # Control patch embeddings
        all_x_embedder = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(
                f_patch_size * patch_size * patch_size * control_in_dim,
                dim,
                bias=True,
            )
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

        self.control_all_x_embedder = nn.ModuleDict(all_x_embedder)

        # Control noise refiner
        self.control_noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        # Control transformer blocks
        self.control_layers = nn.ModuleList(
            [
                ZImageControlTransformerBlock(
                    i,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    block_id=i,
                )
                for i in range(num_control_blocks)
            ]
        )

        # Padding token for control context
        self.x_pad_token = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.x_pad_token, std=0.02)

    def forward(
        self,
        control_context: List[torch.Tensor],
        unified_hidden_states: torch.Tensor,
        cap_feats: torch.Tensor,
        timestep_emb: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        rope_embedder,
        patchify_fn,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> tuple[torch.Tensor, ...]:
        """Compute control hints from control context.

        Args:
            control_context: List of control image latents [C, 1, H, W]
            unified_hidden_states: Combined image+caption embeddings from main path
            cap_feats: Caption feature embeddings
            timestep_emb: Timestep embeddings
            attn_mask: Attention mask
            freqs_cis: RoPE frequencies
            rope_embedder: RoPE embedder from base model
            patchify_fn: Patchify function from base model
            patch_size: Spatial patch size
            f_patch_size: Frame patch size

        Returns:
            Tuple of hint tensors to be added at each control layer position
        """
        bsz = len(control_context)
        device = control_context[0].device

        # Patchify control context using base model's patchify
        (
            control_context_patches,
            x_size,
            x_pos_ids,
            x_inner_pad_mask,
        ) = patchify_fn(control_context, patch_size, f_patch_size, cap_feats.size(1))

        # Embed control context
        x_item_seqlens = [len(_) for _ in control_context_patches]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        control_context_cat = torch.cat(control_context_patches, dim=0)
        control_context_cat = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context_cat)

        # Match timestep dtype
        adaln_input = timestep_emb.type_as(control_context_cat)
        control_context_cat[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        control_context_list = list(control_context_cat.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        control_context_padded = pad_sequence(control_context_list, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Refine control context
        for layer in self.control_noise_refiner:
            control_context_padded = layer(control_context_padded, x_attn_mask, x_freqs_cis, adaln_input)

        # Unify with caption features
        cap_item_seqlens = [cap_feats.size(1)] * bsz
        control_context_unified = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            control_context_unified.append(
                torch.cat([control_context_padded[i][:x_len], cap_feats[i][:cap_len]])
            )
        control_context_unified = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)
        c = control_context_unified

        # Process through control layers
        for layer in self.control_layers:
            c = layer(
                c,
                x=unified_hidden_states,
                attn_mask=attn_mask,
                freqs_cis=freqs_cis,
                adaln_input=adaln_input,
            )

        hints = torch.unbind(c)[:-1]
        return hints
