# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) & the InvokeAI Team

import math
from typing import Literal

from pydantic import Field, validator
import torch
from invokeai.app.invocations.latent import LatentsField

from invokeai.app.util.misc import SEED_MAX, get_random_seed
from ...backend.util.devices import choose_torch_device, torch_dtype
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationConfig,
    InvocationContext,
)

"""
Utilities
"""


def get_noise(
    width: int,
    height: int,
    device: torch.device,
    seed: int = 0,
    latent_channels: int = 4,
    downsampling_factor: int = 8,
    use_cpu: bool = True,
    perlin: float = 0.0,
):
    """Generate noise for a given image size."""
    noise_device_type = "cpu" if use_cpu else device.type

    # limit noise to only the diffusion image channels, not the mask channels
    input_channels = min(latent_channels, 4)
    generator = torch.Generator(device=noise_device_type).manual_seed(seed)

    noise_tensor = torch.randn(
        [
            1,
            input_channels,
            height // downsampling_factor,
            width // downsampling_factor,
        ],
        dtype=torch_dtype(device),
        device=noise_device_type,
        generator=generator,
    ).to("cpu")

    return noise_tensor


"""
Nodes
"""


class NoiseOutput(BaseInvocationOutput):
    """Invocation noise output"""

    # fmt: off
    type:  Literal["noise_output"] = "noise_output"

    # Inputs
    noise: LatentsField            = Field(default=None, description="The output noise")
    width:                     int = Field(description="The width of the noise in pixels")
    height:                    int = Field(description="The height of the noise in pixels")
    # fmt: on


def build_noise_output(latents_name: str, latents: torch.Tensor):
    return NoiseOutput(
        noise=LatentsField(latents_name=latents_name),
        width=latents.size()[3] * 8,
        height=latents.size()[2] * 8,
    )


class NoiseInvocation(BaseInvocation):
    """Generates latent noise."""

    type: Literal["noise"] = "noise"

    # Inputs
    seed: int = Field(
        ge=0,
        le=SEED_MAX,
        description="The seed to use",
        default_factory=get_random_seed,
    )
    width: int = Field(
        default=512,
        multiple_of=8,
        gt=0,
        description="The width of the resulting noise",
    )
    height: int = Field(
        default=512,
        multiple_of=8,
        gt=0,
        description="The height of the resulting noise",
    )
    use_cpu: bool = Field(
        default=True,
        description="Use CPU for noise generation (for reproducible results across platforms)",
    )

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Noise",
                "tags": ["latents", "noise"],
            },
        }

    @validator("seed", pre=True)
    def modulo_seed(cls, v):
        """Returns the seed modulo (SEED_MAX + 1) to ensure it is within the valid range."""
        return v % (SEED_MAX + 1)

    def invoke(self, context: InvocationContext) -> NoiseOutput:
        noise = get_noise(
            width=self.width,
            height=self.height,
            device=choose_torch_device(),
            seed=self.seed,
            use_cpu=self.use_cpu,
        )
        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, noise)
        return build_noise_output(latents_name=name, latents=noise)
