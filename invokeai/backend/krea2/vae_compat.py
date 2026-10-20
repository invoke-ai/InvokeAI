"""Compatibility helpers for the Qwen-Image VAE used by Krea-2.

Krea-2 (and Qwen-Image) decode/encode with ``AutoencoderKLQwenImage``. A standalone single-file
``qwen_image_vae.safetensors`` in the native (ComfyUI/Wan) layout is byte-identical to the Anima VAE
and therefore classified with the Anima base, which loads it as ``AutoencoderKLWan``. The two classes
share the exact same diffusers state-dict (identical keys and shapes), so a Wan-loaded VAE can be
used through the same encode/decode path without rebuilding it. Both default configs carry the same
Qwen-Image ``latents_mean`` / ``latents_std`` / ``z_dim`` values read by the Qwen encode/decode nodes.

Also holds the tiling helpers those nodes share, since the tile geometry has to be applied identically
on both classes and restored afterwards — see ``patch_qwen_image_vae_tiling``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

QwenImageCompatibleVAE = AutoencoderKLQwenImage | AutoencoderKLWan


# The Qwen-Image VAE is a 16-channel, 8x-spatial, non-patchified autoencoder. A Wan VAE can only share the
# Qwen-Image encode/decode path if it has that exact geometry. Wan 2.2's VAE is a *different* architecture
# (48 latent channels, patchified) that happens to load as AutoencoderKLWan too, but would fail on the
# 16-vs-48 per-channel normalization (or silently produce incompatible latents), so it must be rejected.
_QWEN_IMAGE_VAE_Z_DIM = 16
_QWEN_IMAGE_VAE_SPATIAL_SCALE = 8


def as_qwen_image_vae(model: Any) -> QwenImageCompatibleVAE:
    """Return a cache-preserving VAE compatible with the Qwen-Image encode/decode path.

    The only expected non-matching input is ``AutoencoderKLWan`` (the same weights loaded via the
    Anima single-file path). A Wan VAE with the Qwen-Image geometry (16 latent channels, 8x spatial, no
    patchification) has identical encode/decode behavior, state-dict layout, and default latent statistics,
    so the cached module can be used directly. A Wan VAE with any other geometry (e.g. Wan 2.2's 48-channel,
    patchified VAE) is rejected here rather than failing deeper in normalization/decode.

    Returning the original object is important: the model cache injects custom modules for partial
    loading before this helper is called, and rebuilding the module from its state dict would discard
    those modules along with any hooks or layerwise-casting configuration.
    """
    if isinstance(model, AutoencoderKLQwenImage):
        return model
    if not isinstance(model, AutoencoderKLWan):
        raise TypeError(f"Expected AutoencoderKLQwenImage or AutoencoderKLWan, got {type(model).__name__}.")

    config = model.config
    z_dim = getattr(config, "z_dim", None)
    patch_size = getattr(config, "patch_size", None)
    spatial_scale = getattr(config, "scale_factor_spatial", _QWEN_IMAGE_VAE_SPATIAL_SCALE)
    if z_dim != _QWEN_IMAGE_VAE_Z_DIM or patch_size is not None or spatial_scale != _QWEN_IMAGE_VAE_SPATIAL_SCALE:
        raise ValueError(
            "AutoencoderKLWan is not Qwen-Image-compatible "
            f"(z_dim={z_dim}, patch_size={patch_size}, scale_factor_spatial={spatial_scale}); "
            f"expected {_QWEN_IMAGE_VAE_Z_DIM} latent channels, {_QWEN_IMAGE_VAE_SPATIAL_SCALE}x spatial, "
            "and no patchification."
        )

    return model


# The stock AutoencoderKLQwenImage tile geometry: 256px tiles advancing in 192px steps, i.e. a 3/4
# stride ratio with a 64px blend band. Both nodes resolve tile_size=0 to QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE
# rather than reading the module's current value, which another invocation may have overwritten.
QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE = 256
_QWEN_IMAGE_VAE_TILE_STRIDE_NUMERATOR = 3
_QWEN_IMAGE_VAE_TILE_STRIDE_DENOMINATOR = 4

# A cost floor, not a correctness one: `_tile_stride_for` keeps the geometry valid all the way down
# (smaller tiles decode and encode to the right size), but the tile *count* grows with the inverse
# square of the stride. At 2560x1440 a 64px tile already emits 1620 tiles; an 8px tile would emit
# 57,600, and the per-tile Python/kernel-launch overhead dominates long before that. Tiles this small
# also blend badly, so the field's low end is clamped rather than honoured literally.
QWEN_IMAGE_VAE_MIN_TILE_SIZE = 64


def resolve_qwen_image_vae_tile_size(tile_size: int) -> int:
    """Resolve a node's ``tile_size`` field to the tile size the VAE will actually use.

    ``tile_size <= 0`` is the nodes' "use the default" sentinel (the workflow UI cannot represent
    ``None`` in a number input and sends 0). Values below ``QWEN_IMAGE_VAE_MIN_TILE_SIZE`` are clamped
    rather than rejected, because the field also has to accept the 0 sentinel and so cannot carry a
    pydantic lower bound. The clamp is about cost, not validity -- see the constant.
    """
    if tile_size <= 0:
        return QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE
    return max(tile_size, QWEN_IMAGE_VAE_MIN_TILE_SIZE)


def _tile_stride_for(tile_size: int) -> int:
    """Return the tile stride to pair with ``tile_size``, keeping the stock 3/4 ratio.

    Rounded down to a multiple of the VAE's 8x spatial compression: ``tiled_encode``/``tiled_decode``
    step the tile loop in one space (pixels for encode, latents for decode) while slicing the
    accumulated tile in the other, so the pixel stride must be exactly 8x the latent stride or the
    two disagree and the output is misaligned.
    """
    stride = tile_size * _QWEN_IMAGE_VAE_TILE_STRIDE_NUMERATOR // _QWEN_IMAGE_VAE_TILE_STRIDE_DENOMINATOR
    return max(_QWEN_IMAGE_VAE_SPATIAL_SCALE, stride // _QWEN_IMAGE_VAE_SPATIAL_SCALE * _QWEN_IMAGE_VAE_SPATIAL_SCALE)


@contextmanager
def patch_qwen_image_vae_tiling(vae: QwenImageCompatibleVAE, tile_size: int | None) -> Iterator[None]:
    """Set the VAE's tiling state for the duration of the block, then restore it.

    Two things make this a context manager rather than a bare ``enable_tiling()`` call:

    - ``enable_tiling`` writes the tile geometry straight onto the module, and ``disable_tiling`` only
      clears ``use_tiling`` — it does not restore the sizes. The module here is the model cache's own
      instance (``as_qwen_image_vae`` deliberately returns it unchanged to keep partial-loading hooks
      intact), so without a restore a tile size set once would persist for the lifetime of the cache
      entry and leak into later invocations — including ``anima_latents_to_image``, which shares the
      same VAE instance when a native-layout ``qwen_image_vae`` single file is loaded.
    - All four parameters are always passed explicitly. ``enable_tiling`` falls back to the module's
      current value for any argument left out, and its ``min``/``stride`` pair must stay consistent:
      the tile loops advance by *stride* but slice each accumulated tile to *min*. A ``min`` below the
      inherited 192px stride silently drops whole bands of the image, and a ``min`` above it grows
      every tile without removing any, making compute scale with ``tile_size**2``.

    ``tile_size=None`` disables tiling for the block.
    """
    original = (
        vae.use_tiling,
        vae.tile_sample_min_height,
        vae.tile_sample_min_width,
        vae.tile_sample_stride_height,
        vae.tile_sample_stride_width,
    )
    try:
        if tile_size is None:
            vae.disable_tiling()
        else:
            stride = _tile_stride_for(tile_size)
            vae.enable_tiling(
                tile_sample_min_height=tile_size,
                tile_sample_min_width=tile_size,
                tile_sample_stride_height=stride,
                tile_sample_stride_width=stride,
            )
        yield
    finally:
        (
            vae.use_tiling,
            vae.tile_sample_min_height,
            vae.tile_sample_min_width,
            vae.tile_sample_stride_height,
            vae.tile_sample_stride_width,
        ) = original
