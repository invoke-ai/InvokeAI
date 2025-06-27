import einops
import torch
from einops import repeat

from invokeai.app.invocations.fields import FluxKontextConditioningField
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import pack
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor


def generate_img_ids_with_offset(
    latent_height: int, latent_width: int, batch_size: int, device: torch.device, dtype: torch.dtype, idx_offset: int = 0
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
        kontext_field: FluxKontextConditioningField,
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
        self.kontext_field = kontext_field

        # Pre-process and cache the kontext latents and ids upon initialization.
        self.kontext_latents, self.kontext_ids = self._prepare_kontext()

    def _prepare_kontext(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the reference image and prepares its latents and IDs."""
        image = self._context.images.get_pil(self.kontext_field.image.image_name)

        # Reuse VAE encoding logic from FluxVaeEncodeInvocation
        vae_info = self._context.models.load(self._vae_field.vae)
        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")
        image_tensor = image_tensor.to(self._device)

        kontext_latents_unpacked = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        # Extract tensor dimensions with descriptive names
        # Latent tensor shape: [batch_size, channels, latent_height, latent_width]
        batch_size, _, latent_height, latent_width = kontext_latents_unpacked.shape

        # Pack the latents and generate IDs. The idx_offset distinguishes these
        # tokens from the main image's tokens, which have an index of 0.
        kontext_latents_packed = pack(kontext_latents_unpacked).to(self._device, self._dtype)
        kontext_ids = generate_img_ids_with_offset(
            latent_height=latent_height,
            latent_width=latent_width,
            batch_size=batch_size,
            device=self._device,
            dtype=self._dtype,
            idx_offset=1,  # Distinguishes reference tokens from main image tokens
        )

        return kontext_latents_packed, kontext_ids

    def apply(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenates the pre-processed kontext data to the main image sequence."""
        # Ensure batch sizes match, repeating kontext data if necessary for batch operations.
        if img.shape[0] != self.kontext_latents.shape[0]:
            self.kontext_latents = self.kontext_latents.repeat(img.shape[0], 1, 1)
            self.kontext_ids = self.kontext_ids.repeat(img.shape[0], 1, 1)

        # Concatenate along the sequence dimension (dim=1)
        combined_img = torch.cat([img, self.kontext_latents], dim=1)
        combined_img_ids = torch.cat([img_ids, self.kontext_ids], dim=1)

        return combined_img, combined_img_ids
