# Initially pulled from https://github.com/black-forest-labs/flux

import math
from typing import Callable

import torch
from einops import rearrange, repeat


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


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
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


def _find_last_index_ge_val(timesteps: list[float], val: float, eps: float = 1e-6) -> int:
    """Find the last index in timesteps that is >= val.

    We use epsilon-close equality to avoid potential floating point errors.
    """
    idx = len(list(filter(lambda t: t >= (val - eps), timesteps))) - 1
    assert idx >= 0
    return idx


def clip_timestep_schedule(timesteps: list[float], denoising_start: float, denoising_end: float) -> list[float]:
    """Clip the timestep schedule to the denoising range.

    Args:
        timesteps (list[float]): The original timestep schedule: [1.0, ..., 0.0].
        denoising_start (float): A value in [0, 1] specifying the start of the denoising process. E.g. a value of 0.2
            would mean that the denoising process start at the last timestep in the schedule >= 0.8.
        denoising_end (float): A value in [0, 1] specifying the end of the denoising process. E.g. a value of 0.8 would
            mean that the denoising process end at the last timestep in the schedule >= 0.2.

    Returns:
        list[float]: The clipped timestep schedule.
    """
    assert 0.0 <= denoising_start <= 1.0
    assert 0.0 <= denoising_end <= 1.0
    assert denoising_start <= denoising_end

    t_start_val = 1.0 - denoising_start
    t_end_val = 1.0 - denoising_end

    t_start_idx = _find_last_index_ge_val(timesteps, t_start_val)
    t_end_idx = _find_last_index_ge_val(timesteps, t_end_val)

    clipped_timesteps = timesteps[t_start_idx : t_end_idx + 1]

    return clipped_timesteps


def clip_timestep_schedule_fractional(
    timesteps: list[float], denoising_start: float, denoising_end: float
) -> list[float]:
    """Clip the timestep schedule to the denoising range. Insert new timesteps to exactly match the desired denoising
    range. (A fractional version of clip_timestep_schedule().)

    Args:
        timesteps (list[float]): The original timestep schedule: [1.0, ..., 0.0].
        denoising_start (float): A value in [0, 1] specifying the start of the denoising process. E.g. a value of 0.2
            would mean that the denoising process start at t=0.8.
        denoising_end (float): A value in [0, 1] specifying the end of the denoising process. E.g. a value of 0.8 would
            mean that the denoising process ends at t=0.2.

    Returns:
        list[float]: The clipped timestep schedule.
    """
    assert 0.0 <= denoising_start <= 1.0
    assert 0.0 <= denoising_end <= 1.0
    assert denoising_start <= denoising_end

    t_start_val = 1.0 - denoising_start
    t_end_val = 1.0 - denoising_end

    t_start_idx = _find_last_index_ge_val(timesteps, t_start_val)
    t_end_idx = _find_last_index_ge_val(timesteps, t_end_val)

    clipped_timesteps = timesteps[t_start_idx : t_end_idx + 1]

    # We know that clipped_timesteps[0] >= t_start_val. Replace clipped_timesteps[0] with t_start_val.
    clipped_timesteps[0] = t_start_val

    # We know that clipped_timesteps[-1] >= t_end_val. If clipped_timesteps[-1] > t_end_val, add another step to
    # t_end_val.
    eps = 1e-6
    if clipped_timesteps[-1] > t_end_val + eps:
        clipped_timesteps.append(t_end_val)

    return clipped_timesteps


def unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack flat array of patch embeddings to latent image."""
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def pack(x: torch.Tensor) -> torch.Tensor:
    """Pack latent image to flattented array of patch embeddings."""
    # Pixel unshuffle with a scale of 2, and flatten the height/width dimensions to get an array of patches.
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def generate_img_ids(h: int, w: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate tensor of image position ids.

    Args:
        h (int): Height of image in latent space.
        w (int): Width of image in latent space.
        batch_size (int): Batch size.
        device (torch.device): Device.
        dtype (torch.dtype): dtype.

    Returns:
        torch.Tensor: Image position ids.
    """

    if device.type == "mps":
        orig_dtype = dtype
        dtype = torch.float16

    img_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=dtype)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    if device.type == "mps":
        img_ids.to(orig_dtype)

    return img_ids


def prepare_multi_ip(img: torch.Tensor, ref_imgs: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate universal rotary position embedding(UnoPE) for reference images.

    Args:
        img (torch.Tensor): latent image representation for denoising
        ref_imgs (list[torch.Tensor]): list of reference images

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]: packed reference images and position embeddings
    """
    bs, c, h, w = img.shape

    ref_img_ids: list[torch.Tensor] = []
    ref_imgs_list: list[torch.Tensor] = []
    pe_shift_w, pe_shift_h = w // 2, h // 2
    for ref_img in ref_imgs:
        _, _, ref_h1, ref_w1 = ref_img.shape
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids1 = torch.zeros(ref_h1 // 2, ref_w1 // 2, 3)
        # img id offsets its maximum values ​​in width and height respectively
        h_offset = pe_shift_h
        w_offset = pe_shift_w
        ref_img_ids1[..., 1] = ref_img_ids1[..., 1] + torch.arange(ref_h1 // 2)[:, None] + h_offset
        ref_img_ids1[..., 2] = ref_img_ids1[..., 2] + torch.arange(ref_w1 // 2)[None, :] + w_offset
        ref_img_ids1 = repeat(ref_img_ids1, "h w c -> b (h w) c", b=bs)
        ref_img_ids.append(ref_img_ids1)
        ref_imgs_list.append(ref_img)

        # Update pe shift
        pe_shift_h += ref_h1 // 2
        pe_shift_w += ref_w1 // 2

    return (
        # "img": img,
        # "img_ids": img_ids.to(img.device),
        ref_imgs_list,
        [ref_img_id.to(img.device) for ref_img_id in ref_img_ids],
    )
