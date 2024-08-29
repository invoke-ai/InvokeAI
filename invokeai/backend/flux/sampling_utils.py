# Initially pulled from https://github.com/black-forest-labs/flux

import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    # We always generate noise on the same device and dtype then cast to ensure consistency across devices/dtypes.
    rand_device = "cpu"
    rand_dtype = torch.float16
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=rand_device,
        dtype=rand_dtype,
        generator=torch.Generator(device=rand_device).manual_seed(seed),
    ).to(device=device, dtype=dtype)


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def pack(x: Tensor) -> Tensor:
    # Pixel unshuffle with a scale of 2, and flatten the height/width dimensions to get an array of patches.
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def generate_img_ids(h: int, w: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    img_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def prepare_latent_img_patches(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an input image in latent space to patches for diffusion.

    This implementation was extracted from:
    https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/sampling.py#L32

    Args:
        img (torch.Tensor): Input image in latent space.

    Returns:
        tuple[Tensor, Tensor]: (img, img_ids), as defined in the original flux repo.
    """
    bs, c, h, w = img.shape

    img = pack(img)

    # Generate patch position ids.
    img_ids = torch.zeros(h // 2, w // 2, 3, device=img.device, dtype=img.dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=img.device, dtype=img.dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=img.device, dtype=img.dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    return img, img_ids
