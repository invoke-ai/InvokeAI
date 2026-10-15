import torch
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from einops import repeat

from invokeai.app.invocations.fields import FluxKontextConditioningField
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import pack
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


def generate_img_ids_with_offset(
    latent_height: int,
    latent_width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    idx_offset: int = 0,
    h_offset: int = 0,
    w_offset: int = 0,
) -> torch.Tensor:
    """Generate tensor of image position ids with optional index and spatial offsets.

    Args:
        latent_height (int): Height of image in latent space (after packing, this becomes h//2).
        latent_width (int): Width of image in latent space (after packing, this becomes w//2).
        batch_size (int): Number of images in the batch.
        device (torch.device): Device to create tensors on.
        dtype (torch.dtype): Data type for the tensors.
        idx_offset (int): Offset to add to the first dimension of the image ids (default: 0).
        h_offset (int): Spatial offset for height/y-coordinates in latent space (default: 0).
        w_offset (int): Spatial offset for width/x-coordinates in latent space (default: 0).

    Returns:
        torch.Tensor: Image position ids with shape [batch_size, (latent_height//2 * latent_width//2), 3].
    """

    if device.type == "mps":
        orig_dtype = dtype
        dtype = torch.float16

    # After packing, the spatial dimensions are halved due to the 2x2 patch structure
    packed_height = latent_height // 2
    packed_width = latent_width // 2

    # Convert spatial offsets from latent space to packed space
    packed_h_offset = h_offset // 2
    packed_w_offset = w_offset // 2

    # Create base tensor for position IDs with shape [packed_height, packed_width, 3]
    # The 3 channels represent: [batch_offset, y_position, x_position]
    img_ids = torch.zeros(packed_height, packed_width, 3, device=device, dtype=dtype)

    # Set the batch offset for all positions
    img_ids[..., 0] = idx_offset

    # Create y-coordinate indices (vertical positions) with spatial offset
    y_indices = torch.arange(packed_height, device=device, dtype=dtype) + packed_h_offset
    # Broadcast y_indices to match the spatial dimensions [packed_height, 1]
    img_ids[..., 1] = y_indices[:, None]

    # Create x-coordinate indices (horizontal positions) with spatial offset
    x_indices = torch.arange(packed_width, device=device, dtype=dtype) + packed_w_offset
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
        kontext_conditioning: list[FluxKontextConditioningField],
        context: InvocationContext,
        vae_field: VAEField,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initializes the KontextExtension, pre-processing the reference images
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
        """Encodes the reference images and prepares their concatenated latents and IDs with spatial tiling."""
        all_latents = []
        all_ids = []

        # Track cumulative dimensions for spatial tiling
        # These track the running extent of the virtual canvas in latent space
        canvas_h = 0  # Running canvas height
        canvas_w = 0  # Running canvas width

        vae_info = self._context.models.load(self._vae_field.vae)

        for idx, kontext_field in enumerate(self.kontext_conditioning):
            image = self._context.images.get_pil(kontext_field.image.image_name)

            # Convert to RGB
            image = image.convert("RGB")

            # Convert to tensor using torchvision transforms for consistency
            transformation = T.Compose(
                [
                    T.ToTensor(),  # Converts PIL image to tensor and scales to [0, 1]
                ]
            )
            image_tensor = transformation(image)
            # Convert from [0, 1] to [-1, 1] range expected by VAE
            image_tensor = image_tensor * 2.0 - 1.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self._device)

            # Continue with VAE encoding
            # Don't sample from the distribution for reference images - use the mean (matching ComfyUI)
            estimated_working_memory = estimate_vae_working_memory_flux(
                operation="encode", image_tensor=image_tensor, vae=vae_info.model
            )

            with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
                assert isinstance(vae, AutoencoderKL)
                vae_dtype = next(iter(vae.parameters())).dtype
                image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

                def encode_reference(x: torch.Tensor = image_tensor) -> torch.Tensor:
                    # `.mode()` is the distribution mean without noise -- `sample=False` on the
                    # class this used to be written for. `x` is bound as a default rather than
                    # closed over: this runs inside a loop over the reference images.
                    return vae.encode(x).latent_dist.mode()

                # Nothing on this path resizes: the reference is encoded at whatever resolution the
                # user supplied, so a large one reserves in proportion -- 19.3 GiB at 3072px, which
                # no consumer card has spare beside a resident transformer. Untiled first, because
                # tiling changes the reference latents and with them the generation; tiled only when
                # the untiled attempt has already failed, which turns a dead generation into a
                # slower one. The FLUX.2 sibling forces a tile unconditionally
                # (`flux2/ref_image_extension.py`); whether that trade is right here is a question
                # about reference fidelity, not about memory, and is left open deliberately.
                try:
                    moments = encode_reference()
                except RuntimeError as e:
                    if not is_oom_error(e):
                        raise
                    e.__traceback__ = None
                    TorchDevice.empty_cache()
                    with scoped_vae_tiling(vae, 0):
                        moments = encode_reference()

                shift_factor = getattr(vae.config, "shift_factor", None)
                if shift_factor is not None:
                    moments = moments - shift_factor
                kontext_latents_unpacked = moments * vae.config.scaling_factor
                TorchDevice.empty_cache()

            # Extract tensor dimensions
            batch_size, _, latent_height, latent_width = kontext_latents_unpacked.shape

            # Pad latents to be compatible with patch_size=2
            # This ensures dimensions are even for the pack() function
            pad_h = (2 - latent_height % 2) % 2
            pad_w = (2 - latent_width % 2) % 2
            if pad_h > 0 or pad_w > 0:
                kontext_latents_unpacked = F.pad(kontext_latents_unpacked, (0, pad_w, 0, pad_h), mode="circular")
                # Update dimensions after padding
                _, _, latent_height, latent_width = kontext_latents_unpacked.shape

            # Pack the latents
            kontext_latents_packed = pack(kontext_latents_unpacked).to(self._device, self._dtype)

            # Determine spatial offsets for this reference image
            h_offset = 0
            w_offset = 0

            if idx > 0:  # First image starts at (0, 0)
                # Calculate potential canvas dimensions for each tiling option
                # Option 1: Tile vertically (below existing content)
                potential_h_vertical = canvas_h + latent_height

                # Option 2: Tile horizontally (to the right of existing content)
                potential_w_horizontal = canvas_w + latent_width

                # Choose arrangement that minimizes the maximum dimension
                # This keeps the canvas closer to square, optimizing attention computation
                if potential_h_vertical > potential_w_horizontal:
                    # Tile horizontally (to the right of existing images)
                    w_offset = canvas_w
                    canvas_w = canvas_w + latent_width
                    canvas_h = max(canvas_h, latent_height)
                else:
                    # Tile vertically (below existing images)
                    h_offset = canvas_h
                    canvas_h = canvas_h + latent_height
                    canvas_w = max(canvas_w, latent_width)
            else:
                # First image - just set canvas dimensions
                canvas_h = latent_height
                canvas_w = latent_width

            # Generate IDs with both index offset and spatial offsets
            kontext_ids = generate_img_ids_with_offset(
                latent_height=latent_height,
                latent_width=latent_width,
                batch_size=batch_size,
                device=self._device,
                dtype=self._dtype,
                idx_offset=1,  # All reference images use index=1 (matching ComfyUI implementation)
                h_offset=h_offset,
                w_offset=w_offset,
            )

            all_latents.append(kontext_latents_packed)
            all_ids.append(kontext_ids)

        # Concatenate all latents and IDs along the sequence dimension
        concatenated_latents = torch.cat(all_latents, dim=1)  # Concatenate along sequence dimension
        concatenated_ids = torch.cat(all_ids, dim=1)  # Concatenate along sequence dimension

        return concatenated_latents, concatenated_ids

    def ensure_batch_size(self, target_batch_size: int) -> None:
        """Ensures the kontext latents and IDs match the target batch size by repeating if necessary."""
        if self.kontext_latents.shape[0] != target_batch_size:
            self.kontext_latents = self.kontext_latents.repeat(target_batch_size, 1, 1)
            self.kontext_ids = self.kontext_ids.repeat(target_batch_size, 1, 1)
