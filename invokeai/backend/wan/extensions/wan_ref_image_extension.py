"""Wan 2.2 I2V reference-image conditioning.

Wan 2.2 I2V-A14B conditions on a reference image by **VAE-encoding** it and
concatenating the resulting latents to the noise latents along the channel
dim — its transformer has ``in_channels=36`` (16 noise + 16 ref-image latents
+ 4 first-frame mask) rather than 16.

This module produces the 20-channel condition tensor ``[B, 20, T_lat, H_lat, W_lat]``
that the denoise loop will concatenate to the 16-channel noise latents each
step, yielding the 36-channel input the I2V transformer expects.

Mirrors diffusers ``WanImageToVideoPipeline.prepare_latents`` lines 423–481
with ``num_frames=1`` and ``expand_timesteps=False`` (the defaults for
single-frame image generation).
"""

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from diffusers.models.autoencoders import AutoencoderKLWan

# Wan 2.2 VAE temporal scale factor — single frame still consumes a 4-position
# slice of the mask tensor, which is why the mask contributes 4 channels.
_WAN_VAE_TEMPORAL_SCALE = 4


def preprocess_reference_image(
    image: Image.Image, width: int, height: int
) -> torch.Tensor:
    """Resize a PIL image to (width, height) and return a normalised [-1, 1]
    tensor of shape ``[1, 3, 1, height, width]`` ready for ``AutoencoderKLWan.encode``."""
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(
            f"Reference-image dimensions must be multiples of 8 (got {width}x{height})."
        )
    resized = image.convert("RGB").resize((width, height), Image.LANCZOS)
    # [0, 1] CHW float tensor.
    pixel = TF.to_tensor(resized)
    # Scale to [-1, 1] to match the Wan VAE's expected input range.
    pixel = pixel * 2.0 - 1.0
    # [3, H, W] -> [1, 3, 1, H, W]: add batch + temporal dims.
    return pixel.unsqueeze(0).unsqueeze(2)


def encode_reference_image_to_condition(
    image: Image.Image,
    vae: AutoencoderKLWan,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the 20-channel I2V condition tensor for a reference image.

    Returns shape ``[1, 20, 1, height // 8, width // 8]`` (4-channel first-frame
    mask concatenated with 16-channel VAE-encoded image latents along the
    channel dim).

    The output should later be concatenated with the 16-channel noise latents
    inside the denoise loop to produce the 36-channel input the I2V transformer
    expects.
    """
    vae_dtype = next(iter(vae.parameters())).dtype
    pixel = preprocess_reference_image(image, width=width, height=height).to(
        device=device, dtype=vae_dtype
    )

    with torch.inference_mode():
        encoded = vae.encode(pixel, return_dict=False)[0]
        latents = encoded.sample()  # [1, 16, 1, H_lat, W_lat]

        # Normalise against the VAE's per-channel mean/std, matching diffusers'
        # ``WanImageToVideoPipeline.prepare_latents`` (lines 440-459). Note the
        # multiplication by 1/std == division by std.
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std)
            .view(1, -1, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latent_condition = (latents - latents_mean) / latents_std

    latent_condition = latent_condition.to(dtype=dtype)

    # First-frame mask: at num_frames=1 every position is "the first frame"
    # (i.e., conditioned). After the temporal-scale expansion the mask is
    # 4 channels of ones at [1, T_lat=1, H_lat, W_lat].
    _, _, t_lat, h_lat, w_lat = latent_condition.shape
    mask = torch.ones(
        1, _WAN_VAE_TEMPORAL_SCALE, t_lat, h_lat, w_lat, device=device, dtype=dtype
    )

    return torch.cat([mask, latent_condition], dim=1)
