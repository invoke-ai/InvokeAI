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


# The two points the mu fit below is built on. 4096 image tokens is a 1024x1024 frame -- one
# megapixel, and the resolution the reference pipeline defaults to. The line says nothing beyond it.
MU_FIT_MIN_SEQ_LEN = 256
MU_FIT_MAX_SEQ_LEN = 4096


def get_lin_function(
    x1: float = MU_FIT_MIN_SEQ_LEN, y1: float = 0.5, x2: float = MU_FIT_MAX_SEQ_LEN, y2: float = 1.15
) -> Callable[[float], float]:
    """A line through two points. Callers must not evaluate it past `x2` -- see `get_schedule`."""
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
        # The fit is a straight line through two points, so it is not evaluated past the second one.
        # Extrapolated, mu reaches 3.23 at 2048px, 6.70 at 3072px and 11.55 at 4096px, and the
        # schedule flattens with it -- the smallest non-zero timestep rises to 0.47, 0.97 and
        # 0.9997. `clip_timestep_schedule_fractional` then has almost nothing left below
        # `1 - denoising_start`. Steps surviving a 30-step request at `denoising_start=0.5`:
        # 8 at 1 MP, 6 at 1.6 MP, 4 at 2.4 MP, 2 at 4.2 MP, and one from ~2112px square. From
        # ~3376px square any `denoising_start` of 0.01 or more leaves one; only a full denoise from 0
        # escapes the clip. Asking for more steps barely helps: at 2048px and 0.499, 100 keep 4 and
        # 1000 keep 39.
        #
        # The effect is graded, not a cliff: between 1 MP and ~1.6 MP the unclamped schedule is not
        # degenerate, and holding mu there is a judgement -- never evaluate the fit outside the
        # range it was built on -- rather than a defect fix. `invokeai/backend/ltx2/sampling.py`
        # holds its own mu at both anchors on the same reasoning; ComfyUI's default FLUX.1 sampling
        # uses a fixed shift of 1.15 at every resolution.
        #
        # Only the upper anchor is held. Below 256 tokens mu merely sags to 0.457, which is
        # harmless, and clamping there would cost small frames the interpolation they rely on.
        #
        # At or below 4096 tokens this is a no-op -- note that is an *area*, so a 1024x1280 canvas
        # is already above it.
        mu = get_lin_function(y1=base_shift, y2=max_shift)(min(image_seq_len, MU_FIT_MAX_SEQ_LEN))
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
    # Set batch offset to 0 for main image tokens
    img_ids[..., 0] = 0
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    if device.type == "mps":
        img_ids = img_ids.to(orig_dtype)

    return img_ids
