# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) & the InvokeAI Team


import numpy as np
import torch
from pydantic import validator

from invokeai.app.invocations.latent import LatentsField
from invokeai.app.util.misc import SEED_MAX, get_random_seed

from ...backend.util.devices import choose_torch_device, torch_dtype
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
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
        dtype=torch_dtype(device),
        device=noise_device_type,
        generator=generator,
    ).to("cpu")

    return noise_tensor


"""
Nodes
"""


@invocation_output("noise_output")
class NoiseOutput(BaseInvocationOutput):
    """Invocation noise output."""

    noise: LatentsField = OutputField(default=None, description=FieldDescriptions.noise)
    width: int = OutputField(description=FieldDescriptions.width)
    height: int = OutputField(description=FieldDescriptions.height)


def build_noise_output(latents_name: str, latents: torch.Tensor, seed: int):
    return NoiseOutput(
        noise=LatentsField(latents_name=latents_name, seed=seed),
        width=latents.size()[3] * 8,
        height=latents.size()[2] * 8,
    )


@invocation("noise", title="Noise", tags=["latents", "noise"], category="latents", version="1.0.0")
class NoiseInvocation(BaseInvocation):
    """Generates latent noise."""

    seed: int = InputField(
        ge=0,
        le=SEED_MAX,
        description=FieldDescriptions.seed,
        default_factory=get_random_seed,
    )
    width: int = InputField(
        default=512,
        multiple_of=8,
        gt=0,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        default=512,
        multiple_of=8,
        gt=0,
        description=FieldDescriptions.height,
    )
    use_cpu: bool = InputField(
        default=True,
        description="Use CPU for noise generation (for reproducible results across platforms)",
    )

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
        return build_noise_output(latents_name=name, latents=noise, seed=self.seed)


@invocation(
    "blend_noise", title="Blend Noise", tags=["latents", "noise", "variations"], category="latents", version="1.0.0"
)
class BlendNoiseInvocation(BaseInvocation):
    """Blend two noise tensors according to a proportion. Useful for generating variations."""

    noise_A: LatentsField = InputField(description=FieldDescriptions.noise, input=Input.Connection, ui_order=0)
    noise_B: LatentsField = InputField(description=FieldDescriptions.noise, input=Input.Connection, ui_order=1)
    blend_ratio: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.blend_alpha)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> NoiseOutput:
        """Combine two noise vectors, returning a blend that can be used to generate variations."""
        noise_a = context.services.latents.get(self.noise_A.latents_name)
        noise_b = context.services.latents.get(self.noise_B.latents_name)

        if noise_a is None or noise_b is None:
            raise Exception("Both noise_A and noise_B must be provided.")
        if noise_a.shape != noise_b.shape:
            raise Exception("Both noise_A and noise_B must be same dimensions.")

        seed = self.noise_A.seed
        alpha = self.blend_ratio
        merged_noise = self.slerp(alpha, noise_a, noise_b)

        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, merged_noise)
        return build_noise_output(latents_name=name, latents=merged_noise, seed=seed)

    def slerp(self, t: float, v0: torch.tensor, v1: torch.tensor, DOT_THRESHOLD: float = 0.9995):
        """
        Spherical linear interpolation.

        :param t: Mixing value, float between 0.0 and 1.0.
        :param v0: Source noise
        :param v1: Target noise
        :DOT_THRESHOLD: Threshold for considering two vectors colineal. Don't change.

        :Returns: Interpolation vector between v0 and v1
        """
        device = v0.device or choose_torch_device()
        v0 = v0.detach().cpu().numpy()
        v1 = v1.detach().cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        return torch.from_numpy(v2).to(device)
