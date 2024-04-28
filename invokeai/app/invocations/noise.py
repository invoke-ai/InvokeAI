# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) & the InvokeAI Team


import torch
from pydantic import field_validator

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import FieldDescriptions, InputField, LatentsField, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX

from ...backend.util.devices import TorchDevice
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
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
        dtype=TorchDevice.choose_torch_dtype(device=device),
        device=noise_device_type,
        generator=generator,
    ).to("cpu")

    return noise_tensor


"""
Nodes
"""


@invocation_output("noise_output")
class NoiseOutput(BaseInvocationOutput):
    """Invocation noise output"""

    noise: LatentsField = OutputField(description=FieldDescriptions.noise)
    width: int = OutputField(description=FieldDescriptions.width)
    height: int = OutputField(description=FieldDescriptions.height)

    @classmethod
    def build(cls, latents_name: str, latents: torch.Tensor, seed: int) -> "NoiseOutput":
        return cls(
            noise=LatentsField(latents_name=latents_name, seed=seed),
            width=latents.size()[3] * LATENT_SCALE_FACTOR,
            height=latents.size()[2] * LATENT_SCALE_FACTOR,
        )


@invocation(
    "noise",
    title="Noise",
    tags=["latents", "noise"],
    category="latents",
    version="1.0.2",
)
class NoiseInvocation(BaseInvocation):
    """Generates latent noise."""

    seed: int = InputField(
        default=0,
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
    )
    width: int = InputField(
        default=512,
        multiple_of=LATENT_SCALE_FACTOR,
        gt=0,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        default=512,
        multiple_of=LATENT_SCALE_FACTOR,
        gt=0,
        description=FieldDescriptions.height,
    )
    use_cpu: bool = InputField(
        default=True,
        description="Use CPU for noise generation (for reproducible results across platforms)",
    )

    @field_validator("seed", mode="before")
    def modulo_seed(cls, v):
        """Return the seed modulo (SEED_MAX + 1) to ensure it is within the valid range."""
        return v % (SEED_MAX + 1)

    def invoke(self, context: InvocationContext) -> NoiseOutput:
        noise = get_noise(
            width=self.width,
            height=self.height,
            device=TorchDevice.choose_torch_device(),
            seed=self.seed,
            use_cpu=self.use_cpu,
        )
        name = context.tensors.save(tensor=noise)
        return NoiseOutput.build(latents_name=name, latents=noise, seed=self.seed)
