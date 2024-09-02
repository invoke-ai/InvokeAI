import torch


class InpaintExtension:
    """A class for managing inpainting with FLUX."""

    def __init__(self, init_latents: torch.Tensor, inpaint_mask: torch.Tensor, noise: torch.Tensor):
        """Initialize InpaintExtension.

        Args:
            init_latents (torch.Tensor): The initial latents (i.e. un-noised at timestep 0). In 'packed' format.
            inpaint_mask (torch.Tensor): A mask specifying which elements to inpaint. Range [0, 1]. Values of 1 will be
                re-generated. Values of 0 will remain unchanged. Values between 0 and 1 can be used to blend the
                inpainted region with the background. In 'packed' format.
            noise (torch.Tensor): The noise tensor used to noise the init_latents. In 'packed' format.
        """
        assert init_latents.shape == inpaint_mask.shape == noise.shape
        self._init_latents = init_latents
        self._inpaint_mask = inpaint_mask
        self._noise = noise

    def merge_intermediate_latents_with_init_latents(
        self, intermediate_latents: torch.Tensor, timestep: float
    ) -> torch.Tensor:
        """Merge the intermediate latents with the initial latents for the current timestep using the inpaint mask. I.e.
        update the intermediate latents to keep the regions that are not being inpainted on the correct noise
        trajectory.

        This function should be called after each denoising step.
        """
        # Noise the init latents for the current timestep.
        noised_init_latents = self._noise * timestep + (1.0 - timestep) * self._init_latents

        # Merge the intermediate latents with the noised_init_latents using the inpaint_mask.
        return intermediate_latents * self._inpaint_mask + noised_init_latents * (1.0 - self._inpaint_mask)
