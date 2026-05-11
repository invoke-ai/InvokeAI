from typing import Literal

import torch

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.util.devices import TorchDevice

LatentNoiseType = Literal["SD", "FLUX", "FLUX.2", "SD3", "CogView4", "Z-Image", "Anima"]


def validate_noise_dimensions(noise_type: LatentNoiseType, width: int, height: int) -> None:
    multiple_of = 8
    if noise_type in ("FLUX", "FLUX.2", "SD3", "Z-Image"):
        multiple_of = 16
    elif noise_type == "CogView4":
        multiple_of = 32

    if width % multiple_of != 0 or height % multiple_of != 0:
        raise ValueError(f"{noise_type} noise width and height must be a multiple of {multiple_of}")


def get_expected_noise_shape(
    noise_type: LatentNoiseType, width: int, height: int, num_channels: int | None = None
) -> tuple[int, ...]:
    validate_noise_dimensions(noise_type, width, height)

    if noise_type == "SD":
        return (1, 4, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "FLUX":
        return (1, 16, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "FLUX.2":
        return (1, 32, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "SD3":
        return (1, 16, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "CogView4":
        return (1, 16, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "Z-Image":
        return (1, 16, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "Anima":
        return (1, 16, 1, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    raise ValueError(f"Unsupported noise type: {noise_type}")


def validate_noise_tensor_shape(
    noise: torch.Tensor, noise_type: LatentNoiseType, width: int, height: int, num_channels: int | None = None
) -> None:
    expected_shape = get_expected_noise_shape(noise_type, width, height, num_channels)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(f"Expected noise with shape {expected_shape}, got {tuple(noise.shape)}")


def generate_noise_tensor(
    noise_type: LatentNoiseType,
    width: int,
    height: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    use_cpu: bool = True,
) -> torch.Tensor:
    validate_noise_dimensions(noise_type, width, height)
    rand_device = "cpu" if use_cpu else device.type
    rand_dtype = TorchDevice.choose_torch_dtype(device=device)

    if noise_type == "SD":
        return torch.randn(
            1,
            4,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            dtype=rand_dtype,
            device=rand_device,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    if noise_type == "FLUX":
        return torch.randn(
            1,
            16,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    if noise_type == "FLUX.2":
        return torch.randn(
            1,
            32,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    if noise_type == "SD3":
        return torch.randn(
            1,
            16,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    if noise_type == "CogView4":
        return torch.randn(
            1,
            16,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    if noise_type == "Z-Image":
        return torch.randn(
            1,
            16,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=torch.float32,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    if noise_type == "Anima":
        return torch.randn(
            1,
            16,
            1,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=torch.float32,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to("cpu")
    raise ValueError(f"Unsupported noise type: {noise_type}")
