# Initially pulled from https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from invokeai.backend.flux.custom_block_processor import CustomDoubleStreamBlockProcessor
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

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

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None,
        timestep_index: int,
        total_num_timesteps: int,
        controlnet_double_block_residuals: list[Tensor] | None,
        controlnet_single_block_residuals: list[Tensor] | None,
        ip_adapter_extensions: list[XLabsIPAdapterExtension],
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Validate double_block_residuals shape.
        if controlnet_double_block_residuals is not None:
            assert len(controlnet_double_block_residuals) == len(self.double_blocks)
        for block_index, block in enumerate(self.double_blocks):
            assert isinstance(block, DoubleStreamBlock)

            img, txt = CustomDoubleStreamBlockProcessor.custom_double_block_forward(
                timestep_index=timestep_index,
                total_num_timesteps=total_num_timesteps,
                block_index=block_index,
                block=block,
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                ip_adapter_extensions=ip_adapter_extensions,
            )

            if controlnet_double_block_residuals is not None:
                img += controlnet_double_block_residuals[block_index]

        img = torch.cat((txt, img), 1)

        # Validate single_block_residuals shape.
        if controlnet_single_block_residuals is not None:
            assert len(controlnet_single_block_residuals) == len(self.single_blocks)

        for block_index, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)

            if controlnet_single_block_residuals is not None:
                img[:, txt.shape[1] :, ...] += controlnet_single_block_residuals[block_index]

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
