"""FLUX.3 sampling: the timestep schedule, position ids and noise.

The schedule, the rotary position layout and the noise draw follow the reference implementation
(github.com/black-forest-labs/flux-3-image-klein, ``src/flux3/sampling.py``, ``pipeline.py`` and
``autoencoder.py``, Apache-2.0). Latents use FLUX.2's contract, because FLUX.3 decodes with the FLUX.2 VAE:
between nodes they are unpacked ``(B, 32, H/8, W/8)``; the transformer sees them packed into 128-channel
tokens on a ``(H/16, W/16)`` grid by ``pack_flux2``.
"""

import torch

from invokeai.backend.flux2.sampling_utils import unpack_flux2

FLUX3_SCHEDULE_SHIFT = 6.93
"""The constant timestep shift ("alpha") of the reference sampler, for both checkpoints and every resolution."""

FLUX3_IMAGE_ROPE_SCALE = 0.5
"""The model was trained with the row and column position ids of image tokens scaled by this factor."""

FLUX3_PACKED_CHANNELS = 128


def time_shift_flux3(sigmas: list[float], shift: float = FLUX3_SCHEDULE_SHIFT) -> list[float]:
    """Apply the reference sampler's shift ``alpha * t / (1 + (alpha - 1) * t)`` to each sigma."""
    return [shift * t / (1.0 + (shift - 1.0) * t) for t in sigmas]


def unshift_flux3(sigmas: list[float], shift: float = FLUX3_SCHEDULE_SHIFT) -> list[float]:
    """Invert :func:`time_shift_flux3`: ``t = s / (alpha - (alpha - 1) * s)``."""
    return [s / (shift - (shift - 1.0) * s) for s in sigmas]


def get_schedule_flux3(num_steps: int, shift: float = FLUX3_SCHEDULE_SHIFT) -> list[float]:
    """The shifted schedule, descending from 1.0 to 0.0, with ``num_steps + 1`` entries."""
    if num_steps < 1:
        raise ValueError(f"FLUX.3 needs at least one step, got {num_steps}.")
    linear = [1.0 - i / num_steps for i in range(num_steps + 1)]
    return time_shift_flux3(linear, shift)


def redensify_schedule_flux3(
    timesteps: list[float], num_steps: int, shift: float = FLUX3_SCHEDULE_SHIFT
) -> list[float]:
    """Re-space a clipped schedule so that it still takes ``num_steps`` steps.

    Clipping a schedule to an img2img window keeps only the sigmas inside it. The shift pushes sigmas
    towards 1, so a low-strength window keeps very few: at 16 steps and a window starting at sigma 0.5,
    only five survive, and the last has to cover 0.13 -> 0 in one jump. The window is resampled in unshifted
    time and shifted back, so it follows the same curve as the full schedule, sampled more finely. The
    endpoints are kept exactly: the first is the sigma the start latents are blended at, the last is the end
    of the requested range. A window that already holds ``num_steps`` steps is returned unchanged.
    """
    if num_steps < 1 or len(timesteps) < 2 or len(timesteps) - 1 >= num_steps:
        return timesteps

    start, end = unshift_flux3([timesteps[0], timesteps[-1]], shift)
    if start <= end:
        return timesteps

    respaced = time_shift_flux3([start + (end - start) * (i / num_steps) for i in range(num_steps + 1)], shift)
    respaced[0] = timesteps[0]
    respaced[-1] = timesteps[-1]
    return respaced


def image_position_ids(
    packed_height: int, packed_width: int, l_coord: int, device: torch.device | None = None
) -> torch.Tensor:
    """``(t, h, w, l)`` ids of a ``packed_height x packed_width`` token grid, shape ``(h * w, 4)``, float32.

    ``t`` is 0, the row and column carry the image RoPE scale, and ``l`` tells the generated image (0) and the
    reference images (-1, -2, ...) apart.
    """
    rows = torch.arange(packed_height, device=device, dtype=torch.float32)
    cols = torch.arange(packed_width, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(rows, cols, indexing="ij")
    ids = torch.zeros(packed_height, packed_width, 4, device=device, dtype=torch.float32)
    ids[..., 1] = grid_h * FLUX3_IMAGE_ROPE_SCALE
    ids[..., 2] = grid_w * FLUX3_IMAGE_ROPE_SCALE
    ids[..., 3] = float(l_coord)
    return ids.reshape(packed_height * packed_width, 4)


def text_position_ids(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """``(0, 0, 0, i)`` for each text token, shape ``(seq_len, 4)``, float32: text advances along ``l``."""
    ids = torch.zeros(seq_len, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(seq_len, device=device, dtype=torch.float32)
    return ids


def get_noise_flux3(height: int, width: int, seed: int) -> torch.Tensor:
    """Unpacked ``(1, 32, H/8, W/8)`` float32 noise, drawn the way the reference implementation draws it.

    The reference draws image tokens ``(h * w, 128)`` from a seeded CPU generator in float32 and rounds them
    to bf16, so this reproduces its sample for the same seed. Unpacking is a pure rearrange: packing the
    result gives back exactly those tokens.
    """
    if height % 16 or width % 16:
        raise ValueError(f"FLUX.3 width and height must be multiples of 16, got {width}x{height}.")
    packed_h, packed_w = height // 16, width // 16
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randn((packed_h * packed_w, FLUX3_PACKED_CHANNELS), generator=generator, dtype=torch.float32)
    tokens = tokens.to(torch.bfloat16).float()
    return unpack_flux2(tokens.unsqueeze(0), height, width)
