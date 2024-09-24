import torch


class TrajectoryGuidanceExtension:
    """An implementation of trajectory guidance for FLUX."""

    def __init__(self, init_latents: torch.Tensor, inpaint_mask: torch.Tensor | None):
        """Initialize TrajectoryGuidanceExtension.

        Args:
            init_latents (torch.Tensor): The initial latents (i.e. un-noised at timestep 0). In 'packed' format.
            inpaint_mask (torch.Tensor | None): A mask specifying which elements to inpaint. Range [0, 1]. Values of 1
                will be re-generated. Values of 0 will remain unchanged. Values between 0 and 1 can be used to blend the
                inpainted region with the background. In 'packed' format. If None, will be treated as a mask of all 1s.
        """
        self._init_latents = init_latents
        if inpaint_mask is None:
            # The inpaing mask is None, so we initialize a mask with a single value of 1.0.
            # This value will be broadcasted and treated as a mask of all 1s.
            self._inpaint_mask = torch.ones(1, device=init_latents.device, dtype=init_latents.dtype)
        else:
            self._inpaint_mask = inpaint_mask

    def _apply_mask_gradient_adjustment(self, t_prev: float) -> torch.Tensor:
        """Applies inpaint mask gradient adjustment and returns the inpaint mask to be used at the current timestep."""
        # As we progress through the denoising process, we promote gradient regions of the mask to have a full weight of
        # 1.0. This helps to produce more coherent seams around the inpainted region. We experimented with a (small)
        # number of promotion strategies (e.g. gradual promotion based on timestep), but found that a simple cutoff
        # threshold worked well.
        # We use a small epsilon to avoid any potential issues with floating point precision.
        eps = 1e-4
        mask_gradient_t_cutoff = 0.5
        if t_prev > mask_gradient_t_cutoff:
            # Early in the denoising process, use the inpaint mask as-is.
            return self._inpaint_mask
        else:
            # After the cut-off, promote all non-zero mask values to 1.0.
            mask = self._inpaint_mask.where(self._inpaint_mask <= (0.0 + eps), 1.0)

        return mask

    def update_noise(
        self, t_curr_latents: torch.Tensor, pred_noise: torch.Tensor, t_curr: float, t_prev: float
    ) -> torch.Tensor:
        # Handle gradient cutoff.
        mask = self._apply_mask_gradient_adjustment(t_prev)

        # NOTE(ryand): During inpainting, it is common to guide the denoising process by noising the initial latents for
        # the current timestep and then blending the predicted intermediate latents with the noised initial latents.
        # For example:
        # ```
        # noised_init_latents = self._noise * t_prev + (1.0 - t_prev) * self._init_latents
        # return t_prev_latents * self._inpaint_mask + noised_init_latents * (1.0 - self._inpaint_mask)
        # ```
        # Instead of guiding based on the noised initial latents, we have decided to guide based on the noise prediction
        # that points towards the initial latents. The difference between these guidance strategies is minor, but
        # qualitatively we found the latter to produce slightly better results. When change_ratio is 0.0 or 1.0 there is
        # no difference between the two strategies.
        #
        # We experimented with a number of related guidance strategies, but not exhaustively. It's entirely possible
        # that there's a much better way to do this.

        # Calculate noise guidance
        # What noise should the model have predicted at this timestep to step towards self._init_latents?
        # Derivation:
        # > t_prev_latents = t_curr_latents + (t_prev - t_curr) * pred_noise
        # > t_0_latents = t_curr_latents + (0 - t_curr) * init_traj_noise
        # > t_0_latents = t_curr_latents - t_curr * init_traj_noise
        # > init_traj_noise = (t_curr_latents - t_0_latents) / t_curr)
        init_traj_noise = (t_curr_latents - self._init_latents) / t_curr

        # Blend the init_traj_noise with the pred_noise according to the inpaint mask and the trajectory guidance.
        noise = pred_noise * mask + init_traj_noise * (1.0 - mask)

        return noise
