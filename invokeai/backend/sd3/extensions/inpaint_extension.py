import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


class InpaintExtension:
    """A class for managing inpainting with SD3."""

    def __init__(self, init_latents: torch.Tensor, inpaint_mask: torch.Tensor, noise: torch.Tensor):
        """Initialize InpaintExtension.

        Args:
            init_latents (torch.Tensor): The initial latents (i.e. un-noised at timestep 0).
            inpaint_mask (torch.Tensor): A mask specifying which elements to inpaint. Range [0, 1]. Values of 1 will be
                re-generated. Values of 0 will remain unchanged. Values between 0 and 1 can be used to blend the
                inpainted region with the background.
            noise (torch.Tensor): The noise tensor used to noise the init_latents.
        """
        assert init_latents.dim() == inpaint_mask.dim() == noise.dim() == 4
        assert init_latents.shape[-2:] == inpaint_mask.shape[-2:] == noise.shape[-2:]

        self._init_latents = init_latents
        self._inpaint_mask = inpaint_mask
        self._noise = noise

    # TODO(ryand): Experiment with mask gradient adjustment strategies such as the one used in FLUX:
    # `InpaintExtension._apply_mask_gradient_adjustment()`.

    def merge_intermediate_latents_with_init_latents(
        self, intermediate_latents: torch.Tensor, scheduler: FlowMatchEulerDiscreteScheduler, t_prev: torch.Tensor
    ) -> torch.Tensor:
        """Merge the intermediate latents with the initial latents for the current timestep using the inpaint mask. I.e.
        update the intermediate latents to keep the regions that are not being inpainted on the correct noise
        trajectory.

        This function should be called after each denoising step.
        """

        # Noise the init latents for the current timestep.
        noised_init_latents = self._init_latents

        # Note: scheduler.timesteps does not include the final timestep of 0.0. So, if we are in the final timestep, we
        # simply use self._init_latents directly.
        if t_prev[0] > 1e-6:
            noised_init_latents = scheduler.scale_noise(sample=self._init_latents, timestep=t_prev, noise=self._noise)

        # Merge the intermediate latents with the noised_init_latents using the inpaint_mask.
        return intermediate_latents * self._inpaint_mask + noised_init_latents * (1.0 - self._inpaint_mask)
