from dataclasses import dataclass
from typing import Literal, TypedDict

import torch

from invokeai.backend.sd3.mmditx import MMDiTX
from invokeai.backend.sd3.sd3_impls import ModelSamplingDiscreteFlow


class ContextEmbedderConfig(TypedDict):
    target: Literal["torch.nn.Linear"]
    params: dict[str, int]


@dataclass
class Sd3MMDiTXParams:
    patch_size: int
    depth: int
    num_patches: int
    pos_embed_max_size: int
    adm_in_channels: int
    context_shape: tuple[int, int]
    qk_norm: Literal["rms", None]
    x_block_self_attn_layers: list[int]
    context_embedder_config: ContextEmbedderConfig


class Sd3MMDiTX(torch.nn.Module):
    """This class is based closely on
    https://github.com/Stability-AI/sd3.5/blob/19bf11c4e1e37324c5aa5a61f010d4127848a09c/sd3_impls.py#L53
    but has more standard model loading semantics.
    """

    def __init__(
        self,
        params: Sd3MMDiTXParams,
        shift: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.diffusion_model = MMDiTX(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=params.pos_embed_max_size,
            patch_size=params.patch_size,
            in_channels=16,
            depth=params.depth,
            num_patches=params.num_patches,
            adm_in_channels=params.adm_in_channels,
            context_embedder_config=params.context_embedder_config,
            qk_norm=params.qk_norm,
            x_block_self_attn_layers=params.x_block_self_attn_layers,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)

    def apply_model(self, x: torch.Tensor, sigma: torch.Tensor, c_crossattn: torch.Tensor, y: torch.Tensor):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        model_output = self.diffusion_model(x.to(dtype), timestep, context=c_crossattn.to(dtype), y=y.to(dtype)).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, x: torch.Tensor, sigma: float, c_crossattn: torch.Tensor, y: torch.Tensor):
        return self.apply_model(x=x, sigma=sigma, c_crossattn=c_crossattn, y=y)

    def get_dtype(self):
        return self.diffusion_model.dtype
