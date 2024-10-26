# This file was initially copied from:
# https://github.com/huggingface/diffusers/blob/99f608218caa069a2f16dcf9efab46959b15aec0/src/diffusers/models/controlnet_flux.py


from dataclasses import dataclass

import torch
import torch.nn as nn

from invokeai.backend.flux.controlnet.zero_module import zero_module
from invokeai.backend.flux.model import FluxParams
from invokeai.backend.flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class InstantXControlNetFluxOutput:
    controlnet_block_samples: list[torch.Tensor] | None
    controlnet_single_block_samples: list[torch.Tensor] | None


# NOTE(ryand): Mapping between diffusers FLUX transformer params and BFL FLUX transformer params:
# - Diffusers: BFL
# - in_channels: in_channels
# - num_layers: depth
# - num_single_layers: depth_single_blocks
# - attention_head_dim: hidden_size // num_heads
# - num_attention_heads: num_heads
# - joint_attention_dim: context_in_dim
# - pooled_projection_dim: vec_in_dim
# - guidance_embeds: guidance_embed
# - axes_dims_rope: axes_dim


class InstantXControlNetFlux(torch.nn.Module):
    def __init__(self, params: FluxParams, num_control_modes: int | None = None):
        """
        Args:
            params (FluxParams): The parameters for the FLUX model.
            num_control_modes (int | None, optional): The number of controlnet modes. If non-None, then the model is a
                'union controlnet' model and expects a mode conditioning input at runtime.
        """
        super().__init__()

        # The following modules mirror the base FLUX transformer model.
        # -------------------------------------------------------------
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # The following modules are specific to the ControlNet model.
        # -----------------------------------------------------------
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.double_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_blocks)):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.hidden_size, self.hidden_size)))

        self.is_union = False
        if num_control_modes is not None:
            self.is_union = True
            self.controlnet_mode_embedder = nn.Embedding(num_control_modes, self.hidden_size)

        self.controlnet_x_embedder = zero_module(torch.nn.Linear(self.in_channels, self.hidden_size))

    def forward(
        self,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor | None,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> InstantXControlNetFluxOutput:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)

        # Add controlnet_cond embedding.
        img = img + self.controlnet_x_embedder(controlnet_cond)

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        # If this is a union ControlNet, then concat the control mode embedding to the T5 text embedding.
        if self.is_union:
            if controlnet_mode is None:
                # We allow users to enter 'None' as the controlnet_mode if they don't want to worry about this input.
                # We've chosen to use a zero-embedding in this case.
                zero_index = torch.zeros([1, 1], dtype=torch.long, device=txt.device)
                controlnet_mode_emb = torch.zeros_like(self.controlnet_mode_embedder(zero_index))
            else:
                controlnet_mode_emb = self.controlnet_mode_embedder(controlnet_mode)
            txt = torch.cat([controlnet_mode_emb, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:, :1, :], txt_ids], dim=1)
        else:
            assert controlnet_mode is None

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        double_block_samples: list[torch.Tensor] = []
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            double_block_samples.append(img)

        img = torch.cat((txt, img), 1)

        single_block_samples: list[torch.Tensor] = []
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
            single_block_samples.append(img[:, txt.shape[1] :])

        # ControlNet Block
        controlnet_double_block_samples: list[torch.Tensor] = []
        for double_block_sample, controlnet_block in zip(double_block_samples, self.controlnet_blocks, strict=True):
            double_block_sample = controlnet_block(double_block_sample)
            controlnet_double_block_samples.append(double_block_sample)

        controlnet_single_block_samples: list[torch.Tensor] = []
        for single_block_sample, controlnet_block in zip(
            single_block_samples, self.controlnet_single_blocks, strict=True
        ):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples.append(single_block_sample)

        return InstantXControlNetFluxOutput(
            controlnet_block_samples=controlnet_double_block_samples or None,
            controlnet_single_block_samples=controlnet_single_block_samples or None,
        )
