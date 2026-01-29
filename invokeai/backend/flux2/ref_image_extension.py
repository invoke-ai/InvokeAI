"""FLUX.2 Klein Reference Image Extension for multi-reference image editing.

This module provides the Flux2RefImageExtension for FLUX.2 Klein models,
which handles encoding reference images using the FLUX.2 VAE and
generating the appropriate position IDs for multi-reference image editing.

FLUX.2 Klein has built-in support for reference image editing (unlike FLUX.1
which requires a separate Kontext model).
"""

import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import repeat
from PIL import Image

from invokeai.app.invocations.fields import FluxKontextConditioningField
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.sampling_utils import pack_flux2
from invokeai.backend.util.devices import TorchDevice

# Maximum pixel counts for reference images (matches BFL FLUX.2 sampling.py)
# Single reference image: 2024² pixels, Multiple: 1024² pixels
MAX_PIXELS_SINGLE_REF = 2024**2  # ~4.1M pixels
MAX_PIXELS_MULTI_REF = 1024**2  # ~1M pixels


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

    def _bn_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BN normalization to packed latents.

        BN formula (affine=False): y = (x - mean) / std

        Args:
            x: Packed latents of shape (B, seq, 128).

        Returns:
            Normalized latents of same shape.
        """
        assert self._bn_mean is not None and self._bn_std is not None
        bn_mean = self._bn_mean.to(x.device, x.dtype)
        bn_std = self._bn_std.to(x.device, x.dtype)
        return (x - bn_mean) / bn_std

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

            # Convert to tensor using torchvision transforms
            transformation = T.Compose([T.ToTensor()])
            image_tensor = transformation(image)
            # Convert from [0, 1] to [-1, 1] range expected by VAE
            image_tensor = image_tensor * 2.0 - 1.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Encode using FLUX.2 VAE
            with vae_info.model_on_device() as (_, vae):
                vae_dtype = next(iter(vae.parameters())).dtype
                image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

                # FLUX.2 VAE uses diffusers API
                latent_dist = vae.encode(image_tensor, return_dict=False)[0]

                # Use mode() for deterministic encoding (no sampling)
                if hasattr(latent_dist, "mode"):
                    ref_image_latents_unpacked = latent_dist.mode()
                elif hasattr(latent_dist, "sample"):
                    ref_image_latents_unpacked = latent_dist.sample()
                else:
                    ref_image_latents_unpacked = latent_dist

                TorchDevice.empty_cache()

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
                ref_image_latents_packed = self._bn_normalize(ref_image_latents_packed)

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
