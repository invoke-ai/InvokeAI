import torch

from invokeai.backend.util.build_line import build_line


class TrajectoryGuidanceExtension:
    def __init__(
        self, init_latents: torch.Tensor, inpaint_mask: torch.Tensor | None, trajectory_guidance_strength: float
    ):
        """Initialize TrajectoryGuidanceExtension.

        Args:
            init_latents (torch.Tensor): The initial latents (i.e. un-noised at timestep 0). In 'packed' format.
            inpaint_mask (torch.Tensor | None): A mask specifying which elements to inpaint. Range [0, 1]. Values of 1
                will be re-generated. Values of 0 will remain unchanged. Values between 0 and 1 can be used to blend the
                inpainted region with the background. In 'packed' format. If None, will be treated as a mask of all 1s.
            trajectory_guidance_strength (float): A value in [0, 1] specifying the strength of the trajectory guidance.
                A value of 0.0 is equivalent to vanilla image-to-image. A value of 1.0 will guide the denoising process
                very close to the original latents.
        """
        assert 0.0 <= trajectory_guidance_strength <= 1.0
        self._init_latents = init_latents
        self._trajectory_guidance_strength = trajectory_guidance_strength
        if inpaint_mask is None:
            # The inpaing mask is None, so we initialize a mask with a single value of 1.0.
            # This value will be broadcasted and treated as a mask of all 1s.
            self._inpaint_mask = torch.ones(1, device=init_latents.device, dtype=init_latents.dtype)
        else:
            self._inpaint_mask = self._inpaint_mask

    def step(
        self, t_curr_latents: torch.Tensor, pred_noise: torch.Tensor, t_curr: float, t_prev: float
    ) -> torch.Tensor:
        # Handle gradient cutoff.
        # TODO(ryand): This logic is a bit arbitrary. Think about how to clean it up.
        timestep_cutoff = 0.5
        if t_prev > timestep_cutoff:
            # Early in the denoising process, use the smaller mask.
            # I.e. treat gradient values as 0.0.
            mask = self._inpaint_mask.where(self._inpaint_mask >= (1.0 - 1e-3), 0.0)
        else:
            # After the cut-off, use the larger mask.
            # I.e. treat gradient values as 1.0.
            mask = self._inpaint_mask.where(self._inpaint_mask <= (0.0 + 1e-3), 1.0)
            # mask = (self._inpaint_mask > (0.0 + 1e-5)).float()

        # Calculate the change_ratio based on the trajectory guidance strength.
        change_ratio_at_t_1 = build_line(x1=0.0, y1=1.0, x2=1.0, y2=0.0)(self._trajectory_guidance_strength)
        change_ratio_at_cutoff = 1.0
        t_cutoff = build_line(x1=0.0, y1=1.0, x2=1.0, y2=0.5)(self._trajectory_guidance_strength)
        change_ratio = 1.0
        if t_prev > t_cutoff:
            # If we are before the cutoff, lineaarly interpolate between the change_ratio at t=1.0 and the change_ratio
            # at the cutoff.
            change_ratio = build_line(x1=1.0, y1=change_ratio_at_t_1, x2=t_cutoff, y2=change_ratio_at_cutoff)(t_prev)
            # change_ratio = change_ratio_at_t_1 + (change_ratio_at_cutoff - change_ratio_at_t_1) * (1.0 - t_prev) / (
            #     1.0 - t_cutoff
            # )

        mask = mask * change_ratio

        # NOTE(ryand): During inpainting, it is common to guide the denoising process by noising the initial latents for
        # the current timestep and then blending the predicted intermediate latents with the noised initial latents.
        # For example, we could do this here with something like:
        # ```
        # noised_init_latents = self._noise * timestep + (1.0 - timestep) * self._init_latents
        # return intermediate_latents * self._inpaint_mask + noised_init_latents * (1.0 - self._inpaint_mask)
        # ```
        # Instead of guiding based on the noised initial latents, we have decided to guide based on the noise prediction
        # that points towards the initial latents. The difference between these guidance strategies is minor, but
        # qualitatively we found the latter to produce slightly better results. When the guidance strength is 0.0 or 1.0
        # there is no difference between the two strategies.
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

        # Take a denoising step.
        return t_curr_latents + (t_prev - t_curr) * noise
