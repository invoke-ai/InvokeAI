"""FLUX.2 Klein Reference Image Extension for multi-reference image editing.

This module provides the Flux2RefImageExtension for FLUX.2 Klein models,
which handles encoding reference images using the FLUX.2 VAE and
generating the appropriate position IDs for multi-reference image editing.

FLUX.2 Klein has built-in support for reference image editing (unlike FLUX.1
which requires a separate Kontext model).
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import repeat
from PIL import Image

from invokeai.app.invocations.fields import FluxKontextConditioningField
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.latent_norm import bn_normalize
from invokeai.backend.flux2.sampling_utils import pack_flux2
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import (
    estimate_vae_working_memory_flux2,
    flux2_vae_attention_materializes,
)

# Maximum pixel counts for reference images (matches BFL FLUX.2 sampling.py)
# Single reference image: 2024² pixels, Multiple: 1024² pixels
MAX_PIXELS_SINGLE_REF = 2024**2  # ~4.1M pixels
MAX_PIXELS_MULTI_REF = 1024**2  # ~1M pixels

# Tile size (in pixels) forced on the VAE for reference-image encoding, see _prepare_ref_images().
REF_ENCODE_TILE_SIZE = 512
# An untiled reference encode is taken only while its estimated peak stays within this share of the device's
# memory. The rest is for what the cache cannot evict for it: the VAE itself, models locked by the running
# graph, the CUDA context and allocator slack. A 1 MP reference (estimated 2.2 GB, measured 2.1 on an RTX 4090)
# stays untiled on an 8 GB card; a 4 MP one (8.6 GB, measured 8.3) is tiled there and untiled on a 24 GB card.
UNTILED_REF_ENCODE_MAX_VRAM_FRACTION = 0.5


def resize_image_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Resize image to fit within max_pixels while preserving aspect ratio.

    This matches the BFL FLUX.2 sampling.py cap_pixels() behavior.

    Args:
        image: PIL Image to resize.
        max_pixels: Maximum total pixel count (width * height).

    Returns:
        Resized PIL Image (or original if already within bounds).
    """
    width, height = image.size
    pixel_count = width * height

    if pixel_count <= max_pixels:
        return image

    # Calculate scale factor to fit within max_pixels (BFL approach)
    scale = math.sqrt(max_pixels / pixel_count)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Ensure dimensions are at least 1
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def reference_encode_tile_size(
    image_tensor: torch.Tensor,
    vae: torch.nn.Module,
    device: torch.device,
    tiling: Literal["always", "if_attention_materializes"],
) -> int | None:
    """The tile size a reference encode runs at, or None for an untiled encode.

    ``"if_attention_materializes"`` encodes untiled unless the mid-block attention would materialize its score
    matrix (ROCm, MPS) or the untiled peak would not fit the device (see UNTILED_REF_ENCODE_MAX_VRAM_FRACTION).
    """
    if tiling == "always" or flux2_vae_attention_materializes(image_tensor, vae, device):
        return REF_ENCODE_TILE_SIZE
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
    elif device.type == "xpu":
        total_memory = int(torch.xpu.get_device_properties(device).total_memory)
    else:
        return None  # CPU encodes in system RAM.
    untiled_estimate = estimate_vae_working_memory_flux2(
        operation="encode", image_tensor=image_tensor, vae=vae, tile_size=None, device=device
    )
    fits = untiled_estimate.reservation_bytes <= total_memory * UNTILED_REF_ENCODE_MAX_VRAM_FRACTION
    return None if fits else REF_ENCODE_TILE_SIZE


def encode_reference_image(
    vae_info: LoadedModel,
    image: Image.Image,
    tiling: Literal["always", "if_attention_materializes"] = "always",
) -> torch.Tensor:
    """Encode one RGB reference image with the FLUX.2 VAE into unpacked ``(1, 32, H/8, W/8)`` latents.

    Shared by FLUX.2 and FLUX.3, which encode with the same autoencoder. The encode uses the latent
    distribution's mode, so it is deterministic.

    A tiled encode is not the untiled one: each tile's mid-block attention sees only that tile, which moves
    every token, not just the seams (measured on a 1024 px image: RMS error 0.21 against a token RMS of 0.97).
    ``"if_attention_materializes"`` encodes untiled where SDPA runs the attention fused (CUDA), as the
    reference pipelines do, and tiles where it would materialize the score matrix (ROCm, MPS) or where the
    untiled peak would not fit the device (see `reference_encode_tile_size`).
    """
    # Convert from [0, 1] to the [-1, 1] range the VAE expects, and add a batch dimension.
    image_tensor = T.ToTensor()(image) * 2.0 - 1.0
    image_tensor = image_tensor.unsqueeze(0)

    device = TorchDevice.choose_torch_device()
    tile_size = reference_encode_tile_size(image_tensor, vae_info.model, device, tiling)

    # A tiled encode peaks at one tile; tell the cache up front so it frees the room instead of hitting the
    # shortfall as an OOM.
    estimated_working_memory = estimate_vae_working_memory_flux2(
        operation="encode",
        image_tensor=image_tensor,
        vae=vae_info.model,
        tile_size=tile_size,
        device=device,
    )

    with vae_info.model_on_device(working_memory=estimated_working_memory) as (_, vae):
        vae_dtype = next(iter(vae.parameters())).dtype
        image_tensor = image_tensor.to(device=device, dtype=vae_dtype)

        # Where SDPA materializes the encoder mid-block's attention matrix (ROCm, MPS), an untiled
        # encode grows quadratically -- ~15GB at 1024px, hundreds of GB at the 2024px reference cap;
        # on CUDA it stays linear but can still outgrow a small card (see reference_encode_tile_size).
        # A tiled encode is bounded by one tile. The VAE's default tile size equals its sample_size
        # (1024), which still OOMs per tile where the matrix materializes, so force a 512px tile.
        # Save/restore the tiling config because this VAE is a shared, cached instance (e.g.
        # the final image decode must not inherit these settings).
        downsample = 2 ** (len(vae.config.block_out_channels) - 1)
        prev_tiling = (vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size)
        vae.use_tiling = tile_size is not None
        if tile_size is not None:
            vae.tile_sample_min_size = tile_size
            vae.tile_latent_min_size = tile_size // downsample
        try:
            latent_dist = vae.encode(image_tensor, return_dict=False)[0]

            # Use mode() for deterministic encoding (no sampling)
            if hasattr(latent_dist, "mode"):
                latents = latent_dist.mode()
            elif hasattr(latent_dist, "sample"):
                latents = latent_dist.sample()
            else:
                latents = latent_dist
        finally:
            vae.use_tiling, vae.tile_sample_min_size, vae.tile_latent_min_size = prev_tiling

        TorchDevice.empty_cache()

    return latents


def generate_img_ids_flux2_with_offset(
    latent_height: int,
    latent_width: int,
    batch_size: int,
    device: torch.device,
    idx_offset: int = 0,
    h_offset: int = 0,
    w_offset: int = 0,
) -> torch.Tensor:
    """Generate tensor of image position ids with optional offsets for FLUX.2.

    FLUX.2 uses 4D position coordinates (T, H, W, L) for its rotary position embeddings.
    Position IDs use int64 (long) dtype.

    Args:
        latent_height: Height of image in latent space (before packing).
        latent_width: Width of image in latent space (before packing).
        batch_size: Number of images in the batch.
        device: Device to create tensors on.
        idx_offset: Offset for T (time/index) coordinate - use 1 for reference images.
        h_offset: Spatial offset for H coordinate in latent space.
        w_offset: Spatial offset for W coordinate in latent space.

    Returns:
        Image position ids with shape [batch_size, (latent_height//2 * latent_width//2), 4].
    """
    # After packing, the spatial dimensions are halved due to the 2x2 patch structure
    packed_height = latent_height // 2
    packed_width = latent_width // 2

    # Convert spatial offsets from latent space to packed space
    packed_h_offset = h_offset // 2
    packed_w_offset = w_offset // 2

    # Create base tensor for position IDs with shape [packed_height, packed_width, 4]
    # The 4 channels represent: [T, H, W, L]
    img_ids = torch.zeros(packed_height, packed_width, 4, device=device, dtype=torch.long)

    # Set T (time/index offset) for all positions - use 1 for reference images
    img_ids[..., 0] = idx_offset

    # Set H (height/y) coordinates with offset
    h_coords = torch.arange(packed_height, device=device, dtype=torch.long) + packed_h_offset
    img_ids[..., 1] = h_coords[:, None]

    # Set W (width/x) coordinates with offset
    w_coords = torch.arange(packed_width, device=device, dtype=torch.long) + packed_w_offset
    img_ids[..., 2] = w_coords[None, :]

    # L (layer) coordinate stays 0

    # Expand to include batch dimension: [batch_size, (packed_height * packed_width), 4]
    img_ids = img_ids.reshape(1, packed_height * packed_width, 4)
    img_ids = repeat(img_ids, "1 s c -> b s c", b=batch_size)

    return img_ids


class Flux2RefImageExtension:
    """Applies FLUX.2 Klein reference image conditioning.

    This extension handles encoding reference images using the FLUX.2 VAE
    and generating the appropriate 4D position IDs for multi-reference image editing.

    FLUX.2 Klein has built-in support for reference image editing, unlike FLUX.1
    which requires a separate Kontext model.
    """

    def __init__(
        self,
        ref_image_conditioning: list[FluxKontextConditioningField],
        context: InvocationContext,
        vae_field: VAEField,
        device: torch.device,
        dtype: torch.dtype,
        bn_mean: torch.Tensor | None = None,
        bn_std: torch.Tensor | None = None,
    ):
        """Initialize the Flux2RefImageExtension.

        Args:
            ref_image_conditioning: List of reference image conditioning fields.
            context: The invocation context for loading models and images.
            vae_field: The FLUX.2 VAE field for encoding images.
            device: Target device for tensors.
            dtype: Target dtype for tensors.
            bn_mean: BN running mean for normalizing latents (shape: 128).
            bn_std: BN running std for normalizing latents (shape: 128).
        """
        self._context = context
        self._device = device
        self._dtype = dtype
        self._vae_field = vae_field
        self._bn_mean = bn_mean
        self._bn_std = bn_std
        self.ref_image_conditioning = ref_image_conditioning

        # Pre-process and cache the reference image latents and ids upon initialization
        self.ref_image_latents, self.ref_image_ids = self._prepare_ref_images()

    def _prepare_ref_images(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode reference images and prepare their concatenated latents and IDs with spatial tiling."""
        all_latents = []
        all_ids = []

        # Track cumulative dimensions for spatial tiling
        canvas_h = 0
        canvas_w = 0

        vae_info = self._context.models.load(self._vae_field.vae)

        # Determine max pixels based on number of reference images (BFL FLUX.2 approach)
        num_refs = len(self.ref_image_conditioning)
        max_pixels = MAX_PIXELS_SINGLE_REF if num_refs == 1 else MAX_PIXELS_MULTI_REF

        for idx, ref_image_field in enumerate(self.ref_image_conditioning):
            image = self._context.images.get_pil(ref_image_field.image.image_name)
            image = image.convert("RGB")

            # Resize large images to max pixel count (matches BFL FLUX.2 sampling.py)
            image = resize_image_to_max_pixels(image, max_pixels)

            ref_image_latents_unpacked = encode_reference_image(vae_info, image)

            # Extract tensor dimensions (B, 32, H, W for FLUX.2)
            batch_size, _, latent_height, latent_width = ref_image_latents_unpacked.shape

            # Pad latents to be compatible with patch_size=2
            pad_h = (2 - latent_height % 2) % 2
            pad_w = (2 - latent_width % 2) % 2
            if pad_h > 0 or pad_w > 0:
                ref_image_latents_unpacked = F.pad(ref_image_latents_unpacked, (0, pad_w, 0, pad_h), mode="circular")
                _, _, latent_height, latent_width = ref_image_latents_unpacked.shape

            # Pack the latents using FLUX.2 pack function (32 channels -> 128)
            ref_image_latents_packed = pack_flux2(ref_image_latents_unpacked).to(self._device, self._dtype)

            # Apply BN normalization to match the input latents scale
            # This is critical - the transformer expects normalized latents
            if self._bn_mean is not None and self._bn_std is not None:
                ref_image_latents_packed = bn_normalize(ref_image_latents_packed, self._bn_mean, self._bn_std)

            # Determine spatial offsets for this reference image
            h_offset = 0
            w_offset = 0

            if idx > 0:  # First image starts at (0, 0)
                # Calculate potential canvas dimensions for each tiling option
                potential_h_vertical = canvas_h + latent_height
                potential_w_horizontal = canvas_w + latent_width

                # Choose arrangement that minimizes the maximum dimension
                if potential_h_vertical > potential_w_horizontal:
                    # Tile horizontally (to the right)
                    w_offset = canvas_w
                    canvas_w = canvas_w + latent_width
                    canvas_h = max(canvas_h, latent_height)
                else:
                    # Tile vertically (below)
                    h_offset = canvas_h
                    canvas_h = canvas_h + latent_height
                    canvas_w = max(canvas_w, latent_width)
            else:
                canvas_h = latent_height
                canvas_w = latent_width

            # Generate position IDs with 4D format (T, H, W, L)
            # Use T-coordinate offset with scale=10 like diffusers Flux2Pipeline:
            # T = scale + scale * idx (so first ref image is T=10, second is T=20, etc.)
            # The generated image uses T=0, so this clearly separates reference images
            t_offset = 10 + 10 * idx  # scale=10 matches diffusers
            ref_image_ids = generate_img_ids_flux2_with_offset(
                latent_height=latent_height,
                latent_width=latent_width,
                batch_size=batch_size,
                device=self._device,
                idx_offset=t_offset,  # Reference images use T=10, 20, 30...
                h_offset=h_offset,
                w_offset=w_offset,
            )

            all_latents.append(ref_image_latents_packed)
            all_ids.append(ref_image_ids)

        # Concatenate all latents and IDs along the sequence dimension
        concatenated_latents = torch.cat(all_latents, dim=1)
        concatenated_ids = torch.cat(all_ids, dim=1)

        return concatenated_latents, concatenated_ids

    def ensure_batch_size(self, target_batch_size: int) -> None:
        """Ensure the reference image latents and IDs match the target batch size."""
        if self.ref_image_latents.shape[0] != target_batch_size:
            self.ref_image_latents = self.ref_image_latents.repeat(target_batch_size, 1, 1)
            self.ref_image_ids = self.ref_image_ids.repeat(target_batch_size, 1, 1)
