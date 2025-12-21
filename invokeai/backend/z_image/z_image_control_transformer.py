# Adapted from https://github.com/aigc-apps/VideoX-Fun/blob/main/videox_fun/models/z_image_transformer2d_control.py
# Copyright (c) Alibaba, Inc. and its affiliates.
# Apache License 2.0

"""
Z-Image Control Transformer for InvokeAI.

This module provides the ZImageControlTransformer2DModel which extends the base
ZImageTransformer2DModel with control conditioning capabilities (Canny, HED, Depth, Pose, MLSD).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.transformers.transformer_z_image import (
    SEQ_MULTI_OF,
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
)
from diffusers.utils import is_torch_version
from torch.nn.utils.rnn import pad_sequence


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    """Control-specific transformer block with skip connections for hint generation.

    This block extends ZImageTransformerBlock with before_proj and after_proj layers
    that create skip connections for the control signal. The hints are accumulated
    across blocks and used to condition the main transformer.
    """

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


class BaseZImageTransformerBlock(ZImageTransformerBlock):
    """Modified transformer block that accepts control hints.

    This block extends ZImageTransformerBlock to add control hints to the
    hidden states at specific positions in the network.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        block_id: Optional[int] = 0,
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        hints: Optional[tuple[torch.Tensor, ...]] = None,
        context_scale: float = 1.0,
    ) -> torch.Tensor:
        hidden_states = super().forward(
            hidden_states,
            attn_mask=attn_mask,
            freqs_cis=freqs_cis,
            adaln_input=adaln_input,
        )
        if self.block_id is not None and hints is not None:
            hidden_states = hidden_states + hints[self.block_id] * context_scale
        return hidden_states


class ZImageControlTransformer2DModel(ZImageTransformer2DModel):
    """Z-Image Control Transformer for spatial conditioning.

    This model extends ZImageTransformer2DModel with control layers that process
    a control image (e.g., Canny edges, depth map) and inject control signals
    into the main transformer at every other layer.

    The control model supports 5 modes: Canny, HED, Depth, Pose, MLSD.
    Recommended control_context_scale: 0.65-0.80.

    Args:
        control_layers_places: List of layer indices where control is applied.
            Defaults to every other layer [0, 2, 4, ...].
        control_in_dim: Input dimension for control context. Defaults to in_channels.
        All other args are passed to ZImageTransformer2DModel.
    """

    @register_to_config
    def __init__(
        self,
        control_layers_places: Optional[List[int]] = None,
        control_in_dim: Optional[int] = None,
        all_patch_size: tuple[int, ...] = (2,),
        all_f_patch_size: tuple[int, ...] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (1024, 512, 512),
    ):
        super().__init__(
            all_patch_size=all_patch_size,
            all_f_patch_size=all_f_patch_size,
            in_channels=in_channels,
            dim=dim,
            n_layers=n_layers,
            n_refiner_layers=n_refiner_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=norm_eps,
            qk_norm=qk_norm,
            cap_feat_dim=cap_feat_dim,
            rope_theta=rope_theta,
            t_scale=t_scale,
            axes_dims=axes_dims,
            axes_lens=axes_lens,
        )

        # Control layer configuration
        self.control_layers_places = (
            list(range(0, n_layers, 2)) if control_layers_places is None else control_layers_places
        )
        self.control_in_dim = in_channels if control_in_dim is None else control_in_dim

        assert 0 in self.control_layers_places
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers_places)}

        # Replace standard layers with control-aware layers
        del self.layers
        self.layers = nn.ModuleList(
            [
                BaseZImageTransformerBlock(
                    i,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    block_id=self.control_layers_mapping[i] if i in self.control_layers_places else None,
                )
                for i in range(n_layers)
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
                for i in range(len(self.control_layers_places))
            ]
        )

        # Control patch embeddings
        all_x_embedder = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size, strict=True):
            x_embedder = nn.Linear(
                f_patch_size * patch_size * patch_size * self.control_in_dim,
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

    def patchify(
        self,
        all_image: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cap_seq_len: int,
    ) -> tuple[List[torch.Tensor], List[tuple], List[torch.Tensor], List[torch.Tensor]]:
        """Patchify images without embedding.

        This method extracts patches from images for control context processing.
        Unlike patchify_and_embed, this only processes images without caption features.

        Args:
            all_image: List of image tensors [C, F, H, W]
            patch_size: Spatial patch size (height and width)
            f_patch_size: Frame patch size
            cap_seq_len: Caption sequence length (for position ID offset)

        Returns:
            Tuple of:
            - all_image_out: List of patchified image tensors
            - all_image_size: List of (F, H, W) tuples
            - all_image_pos_ids: List of position ID tensors
            - all_image_pad_mask: List of padding mask tensors
        """
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []

        # Calculate padded caption length for position offset
        cap_padding_len = (-cap_seq_len) % SEQ_MULTI_OF
        cap_padded_len = cap_seq_len + cap_padding_len

        for image in all_image:
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            # Patchify: [C, F, H, W] -> [(F*H*W)/(patch), patch_elements * C]
            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            # Create position IDs
            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_padded_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)

            # Padding mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )

            # Padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return all_image_out, all_image_size, all_image_pos_ids, all_image_pad_mask

    def forward_control(
        self,
        x: torch.Tensor,
        cap_feats: torch.Tensor,
        control_context: List[torch.Tensor],
        kwargs: Dict[str, Any],
        t: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> tuple[torch.Tensor, ...]:
        """Process control context and generate hints for the main transformer.

        Args:
            x: Unified image+caption embeddings from main path
            cap_feats: Caption feature embeddings
            control_context: List of control images (VAE-encoded latents)
            kwargs: Additional kwargs including attn_mask, freqs_cis
            t: Timestep embeddings
            patch_size: Spatial patch size
            f_patch_size: Frame patch size

        Returns:
            Tuple of hint tensors to be added at each control layer position
        """
        bsz = len(control_context)
        device = control_context[0].device

        # Patchify control context
        (
            control_context_patches,
            x_size,
            x_pos_ids,
            x_inner_pad_mask,
        ) = self.patchify(control_context, patch_size, f_patch_size, cap_feats.size(1))

        # Embed control context
        x_item_seqlens = [len(_) for _ in control_context_patches]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        control_context_cat = torch.cat(control_context_patches, dim=0)
        control_context_cat = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context_cat)

        # Match t_embedder output dtype
        adaln_input = t.type_as(control_context_cat)
        control_context_cat[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        control_context_list = list(control_context_cat.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        control_context_padded = pad_sequence(control_context_list, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Refine control context
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.control_noise_refiner:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                control_context_padded = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    control_context_padded,
                    x_attn_mask,
                    x_freqs_cis,
                    adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.control_noise_refiner:
                control_context_padded = layer(control_context_padded, x_attn_mask, x_freqs_cis, adaln_input)

        # Unify with caption features
        cap_item_seqlens = [cap_feats.size(1)] * bsz  # Assume same length for batch
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
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(
                        layer,
                        x=x,
                        attn_mask=kwargs["attn_mask"],
                        freqs_cis=kwargs["freqs_cis"],
                        adaln_input=kwargs["adaln_input"],
                    ),
                    c,
                    **ckpt_kwargs,
                )
            else:
                c = layer(
                    c,
                    x=x,
                    attn_mask=kwargs["attn_mask"],
                    freqs_cis=kwargs["freqs_cis"],
                    adaln_input=kwargs["adaln_input"],
                )

        hints = torch.unbind(c)[:-1]
        return hints

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cap_feats: List[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        control_context: Optional[List[torch.Tensor]] = None,
        control_context_scale: float = 1.0,
    ) -> tuple[List[torch.Tensor], dict]:
        """Forward pass with control conditioning.

        Args:
            x: List of image tensors [B, C, 1, H, W]
            t: Timestep tensor
            cap_feats: List of caption feature tensors
            patch_size: Spatial patch size (default 2)
            f_patch_size: Frame patch size (default 1)
            control_context: List of control image latents (VAE-encoded)
            control_context_scale: Strength of control signal (0.65-0.80 recommended)

        Returns:
            Tuple of (output tensors, empty dict)
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        if control_context is None:
            # Fall back to base model behavior without control
            return super().forward(x, t, cap_feats, patch_size, f_patch_size)

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # Image embedding and refinement
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Noise refiner
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    x_attn_mask,
                    x_freqs_cis,
                    adaln_input,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # Caption embedding and refinement
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                cap_feats = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    cap_feats,
                    cap_attn_mask,
                    cap_freqs_cis,
                    **ckpt_kwargs,
                )
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # Unified processing
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens, strict=True)]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        # Generate control hints
        kwargs = {
            "attn_mask": unified_attn_mask,
            "freqs_cis": unified_freqs_cis,
            "adaln_input": adaln_input,
        }
        hints = self.forward_control(
            unified,
            cap_feats,
            control_context,
            kwargs,
            t=t,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )

        # Main transformer with control hints
        for layer in self.layers:
            layer_kwargs = {
                "attn_mask": unified_attn_mask,
                "freqs_cis": unified_freqs_cis,
                "adaln_input": adaln_input,
                "hints": hints,
                "context_scale": control_context_scale,
            }
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, **static_kwargs):
                    def custom_forward(*inputs):
                        return module(*inputs, **static_kwargs)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                unified = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, **layer_kwargs),
                    unified,
                    **ckpt_kwargs,
                )
            else:
                unified = layer(unified, **layer_kwargs)

        # Final layer and unpatchify
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        x = torch.stack(x)
        return x, {}
