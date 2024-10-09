# This file was initially based on:
# https://github.com/XLabs-AI/x-flux/blob/47495425dbed499be1e8e5a6e52628b07349cba2/src/flux/controlnet.py


from dataclasses import dataclass

import torch
from einops import rearrange

from invokeai.backend.flux.controlnet.zero_module import zero_module
from invokeai.backend.flux.model import FluxParams
from invokeai.backend.flux.modules.layers import DoubleStreamBlock, EmbedND, MLPEmbedder, timestep_embedding


@dataclass
class XLabsControlNetFluxOutput:
    controlnet_double_block_residuals: list[torch.Tensor] | None


class XLabsControlNetFlux(torch.nn.Module):
    """A ControlNet model for FLUX.

    The architecture is very similar to the base FLUX model, with the following differences:
    - A `controlnet_depth` parameter is passed to control the number of double_blocks that the ControlNet is applied to.
      In order to keep the ControlNet small, this is typically much less than the depth of the base FLUX model.
    - There is a set of `controlnet_blocks` that are applied to the output of each double_block.
    """

    def __init__(self, params: FluxParams, controlnet_depth: int = 2):
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
        self.img_in = torch.nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else torch.nn.Identity()
        )
        self.txt_in = torch.nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = torch.nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(controlnet_depth)
            ]
        )

        # Add ControlNet blocks.
        self.controlnet_blocks = torch.nn.ModuleList([])
        for _ in range(controlnet_depth):
            controlnet_block = torch.nn.Linear(self.hidden_size, self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
        self.pos_embed_input = torch.nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.input_hint_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1, stride=2),
            torch.nn.SiLU(),
            zero_module(torch.nn.Conv2d(16, 16, 3, padding=1)),
        )

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        controlnet_cond: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> XLabsControlNetFluxOutput:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        controlnet_cond = self.input_hint_block(controlnet_cond)
        controlnet_cond = rearrange(controlnet_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_res_samples: list[torch.Tensor] = []

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            block_res_samples.append(img)

        controlnet_block_res_samples: list[torch.Tensor] = []
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks, strict=True):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples.append(block_res_sample)

        return XLabsControlNetFluxOutput(controlnet_double_block_residuals=controlnet_block_res_samples)
