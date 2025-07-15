import einops
import numpy as np
import torch
from einops import repeat
from PIL import Image

from invokeai.app.invocations.fields import FluxKontextConditioningField
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import pack
from invokeai.backend.flux.util import PREFERED_KONTEXT_RESOLUTIONS


def generate_img_ids_with_offset(
    latent_height: int,
    latent_width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    idx_offset: int = 0,
) -> torch.Tensor:
    """Generate tensor of image position ids with an optional offset.

    Args:
        latent_height (int): Height of image in latent space (after packing, this becomes h//2).
        latent_width (int): Width of image in latent space (after packing, this becomes w//2).
        batch_size (int): Number of images in the batch.
        device (torch.device): Device to create tensors on.
        dtype (torch.dtype): Data type for the tensors.
        idx_offset (int): Offset to add to the first dimension of the image ids.

    Returns:
        torch.Tensor: Image position ids with shape [batch_size, (latent_height//2 * latent_width//2), 3].
    """

    if device.type == "mps":
        orig_dtype = dtype
        dtype = torch.float16

    # After packing, the spatial dimensions are halved due to the 2x2 patch structure
    packed_height = latent_height // 2
    packed_width = latent_width // 2

    # Create base tensor for position IDs with shape [packed_height, packed_width, 3]
    # The 3 channels represent: [batch_offset, y_position, x_position]
    img_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)

    # Set the batch offset for all positions
    img_ids[..., 0] = idx_offset

    # Create y-coordinate indices (vertical positions)
    y_indices = torch.arange(packed_height, device=device, dtype=dtype)
    # Broadcast y_indices to match the spatial dimensions [packed_height, 1]
    img_ids[..., 1] = y_indices[:, None]

    # Create x-coordinate indices (horizontal positions)
    x_indices = torch.arange(packed_width, device=device, dtype=dtype)
    # Broadcast x_indices to match the spatial dimensions [1, packed_width]
    img_ids[..., 2] = x_indices[None, :]

    # Expand to include batch dimension: [batch_size, (packed_height * packed_width), 3]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    if device.type == "mps":
        img_ids = img_ids.to(orig_dtype)

    return img_ids


class KontextExtension:
    """Applies FLUX Kontext (reference image) conditioning."""

    def __init__(
        self,
        kontext_conditioning: FluxKontextConditioningField,
        context: InvocationContext,
        vae_field: VAEField,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initializes the KontextExtension, pre-processing the reference image
        into latents and positional IDs.
        """
        self._context = context
        self._device = device
        self._dtype = dtype
        self._vae_field = vae_field
        self.kontext_conditioning = kontext_conditioning

        # Pre-process and cache the kontext latents and ids upon initialization.
        self.kontext_latents, self.kontext_ids = self._prepare_kontext()

    def _prepare_kontext(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the reference image and prepares its latents and IDs."""
        image = self._context.images.get_pil(self.kontext_conditioning.image.image_name)

        # Calculate aspect ratio of input image
        width, height = image.size
        aspect_ratio = width / height

        # Find the closest preferred resolution by aspect ratio
        _, target_width, target_height = min(
            ((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS), key=lambda x: x[0]
        )

        # Apply BFL's scaling formula
        # This ensures compatibility with the model's training
        scaled_width = 2 * int(target_width / 16)
        scaled_height = 2 * int(target_height / 16)

        # Resize to the exact resolution used during training
        image = image.convert("RGB")
        final_width = 8 * scaled_width
        final_height = 8 * scaled_height
        image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)

        # Convert to tensor with same normalization as BFL
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0
        image_tensor = einops.rearrange(image_tensor, "h w c -> 1 c h w")
        image_tensor = image_tensor.to(self._device)

        # Continue with VAE encoding
        vae_info = self._context.models.load(self._vae_field.vae)
        kontext_latents_unpacked = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        # Extract tensor dimensions
        batch_size, _, latent_height, latent_width = kontext_latents_unpacked.shape

        # Pack the latents and generate IDs
        kontext_latents_packed = pack(kontext_latents_unpacked).to(self._device, self._dtype)
        kontext_ids = generate_img_ids_with_offset(
            latent_height=latent_height,
            latent_width=latent_width,
            batch_size=batch_size,
            device=self._device,
            dtype=self._dtype,
            idx_offset=1,
        )

        return kontext_latents_packed, kontext_ids

    def ensure_batch_size(self, target_batch_size: int) -> None:
        """Ensures the kontext latents and IDs match the target batch size by repeating if necessary."""
        if self.kontext_latents.shape[0] != target_batch_size:
            self.kontext_latents = self.kontext_latents.repeat(target_batch_size, 1, 1)
            self.kontext_ids = self.kontext_ids.repeat(target_batch_size, 1, 1)
