import torch


def assert_broadcastable(*shapes):
    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError as e:
        raise AssertionError(f"Shapes {shapes} are not broadcastable.") from e


class RectifiedFlowInpaintExtension:
    """A class for managing inpainting with rectified flow models (e.g. FLUX, SD3, CogView4)."""

    def __init__(self, init_latents: torch.Tensor, inpaint_mask: torch.Tensor, noise: torch.Tensor):
        """Initialize InpaintExtension.

        Args:
            init_latents (torch.Tensor): The initial latents (i.e. un-noised at timestep 0). In 'packed' format.
            inpaint_mask (torch.Tensor): A mask specifying which elements to inpaint. Range [0, 1]. Values of 1 will be
                re-generated. Values of 0 will remain unchanged. Values between 0 and 1 can be used to blend the
                inpainted region with the background. In 'packed' format.
            noise (torch.Tensor): The noise tensor used to noise the init_latents. In 'packed' format.
        """
        assert_broadcastable(init_latents.shape, inpaint_mask.shape, noise.shape)

        self._init_latents = init_latents
        self._inpaint_mask = inpaint_mask
        self._noise = noise

    def _apply_mask_gradient_adjustment(self, t_prev: float) -> torch.Tensor:
        """Applies inpaint mask gradient adjustment and returns the inpaint mask to be used at the current timestep."""
        # As we progress through the denoising process, we promote gradient regions of the mask to have a full weight of
        # 1.0. This helps to produce more coherent seams around the inpainted region.

        # We use a small epsilon to avoid any potential issues with floating point precision.
        eps = 1e-4
        mask = torch.where(self._inpaint_mask >= t_prev + eps, 1.0, 0.0).to(
            dtype=self._inpaint_mask.dtype, device=self._inpaint_mask.device
        )

        return mask

    def merge_intermediate_latents_with_init_latents(
        self, intermediate_latents: torch.Tensor, t_prev: float
    ) -> torch.Tensor:
        """Merge the intermediate latents with the initial latents for the current timestep using the inpaint mask. I.e.
        update the intermediate latents to keep the regions that are not being inpainted on the correct noise
        trajectory.

        This function should be called after each denoising step.
        """
        mask = self._apply_mask_gradient_adjustment(t_prev)

        # Noise the init latents for the current timestep.
        noised_init_latents = self._noise * t_prev + (1.0 - t_prev) * self._init_latents

        # Merge the intermediate latents with the noised_init_latents using the inpaint_mask.
        return intermediate_latents * mask + noised_init_latents * (1.0 - mask)
