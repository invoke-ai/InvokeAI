from typing import Callable

import torch


def build_line(x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
    return lambda x: (y2 - y1) / (x2 - x1) * (x - x1) + y1


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
        self, t_curr_latents: torch.Tensor, pred_noise: torch.Tensor, t_curr: float, t_prev: float
    ) -> torch.Tensor:
        """Merge the intermediate latents with the initial latents for the current timestep using the inpaint mask. I.e.
        update the intermediate latents to keep the regions that are not being inpainted on the correct noise
        trajectory.

        This function should be called after each denoising step.
        """

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

        # Max change
        # Scale the mask so that the maximum change follows some schedule.
        # Parameters to control the max change curve:
        # - denoise start: implicitly 0 until this point
        # - max_change_timestep_cutoff: the timestep at which max_change of 1.0 is reached
        # - What curve should we follow in-between? Linear? Step function?

        # This is completely arbitrary that we are using the same value for max_change_timestep_cutoff and max_change = 0.0
        denoise_strength = 0.4

        # # Alg 1
        # # -----

        # # Map denoise_strength to guidance schedule.
        # # Limits:
        # # - denoise_strength = 1.0: max_change_before_cutoff = 1.0, cutoff = 1.0
        # # - denoise_strength = 0.0: max_change_before_cutoff = 0.0, cutoff = 0.0
        # # - denoise_strength = 0.5: max_change_before_cutoff = 0.5, cutoff = 0.25
        # max_change_before_cutoff_at_ds_0 = 0.2
        # max_change_before_cutoff_at_ds_1 = 1.0
        # max_change_before_cutoff = (
        #     max_change_before_cutoff_at_ds_0
        #     + (max_change_before_cutoff_at_ds_1 - max_change_before_cutoff_at_ds_0) * denoise_strength
        # )

        # t_cutoff_at_ds_0 = 0.5
        # t_cutoff_at_ds_1 = 1.0
        # t_cutoff = t_cutoff_at_ds_0 + (t_cutoff_at_ds_1 - t_cutoff_at_ds_0) * denoise_strength
        # t_cutoff = 0.8

        # # Alg 2
        # # -----
        max_change_at_t_1 = build_line(x1=0.0, y1=0.0, x2=1.0, y2=1.0)(denoise_strength)
        max_change_at_cutoff = 1.0
        t_cutoff = build_line(x1=0.0, y1=0.5, x2=1.0, y2=1.0)(denoise_strength)
        # t_cutoff = 0.75

        if t_prev > t_cutoff:
            max_change = max_change_at_t_1 + (max_change_at_cutoff - max_change_at_t_1) * (1.0 - t_prev) / (
                1.0 - t_cutoff
            )

            # max_change = max_change_before_cutoff

            # total_time_to_cutoff = 1.0 - max_change_timestep_cutoff
            # cur_time_elapsed = 1.0 - t_prev
            # max_change = cur_time_elapsed / total_time_to_cutoff
        else:
            # After cut-off, max_change is 1.0 (i.e. no guidance).
            max_change = 1.0
        mask = mask * max_change
        print(f">>> {max_change=}, {denoise_strength=}, {t_cutoff=}")

        # Noise guidance
        # --------------
        # What noise should the model have predicted at this timestep to step towards self._init_latents?
        # Derivation:
        # > Recall the noise model:
        # > t_prev_latents = t_curr_latents + (t_prev - t_curr) * pred_noise
        # > t_0_latents = t_curr_latents + (0 - t_curr) * init_traj_noise
        # > t_0_latents = t_curr_latents - t_curr * init_traj_noise
        # > init_traj_noise = (t_curr_latents - t_0_latents) / t_curr)
        init_traj_noise = (t_curr_latents - self._init_latents) / t_curr

        # Blend the init_traj_noise with the pred_noise according to the inpaint mask.
        noise = pred_noise * mask + init_traj_noise * (1.0 - mask)

        return t_curr_latents + (t_prev - t_curr) * noise

        # # Noise guidance with normaliztion
        # # --------------
        # # What noise should the model have predicted at this timestep to step towards self._init_latents?
        # # Derivation:
        # # > Recall the noise model:
        # # > t_prev_latents = t_curr_latents + (t_prev - t_curr) * pred_noise
        # # > t_0_latents = t_curr_latents + (0 - t_curr) * init_traj_noise
        # # > t_0_latents = t_curr_latents - t_curr * init_traj_noise
        # # > init_traj_noise = (t_curr_latents - t_0_latents) / t_curr)
        # init_traj_noise = (t_curr_latents - self._init_latents) / t_curr

        # # Normalize the init_traj_noise to have the same norm as the pred_noise.
        # init_traj_noise = (
        #     init_traj_noise / (torch.linalg.matrix_norm(init_traj_noise) + 1e-6) * torch.linalg.matrix_norm(pred_noise)
        # )

        # # Blend the init_traj_noise with the pred_noise according to the inpaint mask.
        # noise = pred_noise * mask + init_traj_noise * (1.0 - mask)

        # return t_curr_latents + (t_prev - t_curr) * noise

        # # Trajectory guidance
        # # -------------------
        # # Noise the init latents for the current timestep.
        # noised_init_latents = self._noise * t_prev + (1.0 - t_prev) * self._init_latents

        # # Calculate the predicted intermediate latents.
        # intermediate_latents = t_curr_latents + (t_prev - t_curr) * pred_noise

        # # Blend the predicted intermediate latents with the noised_init_latents using the inpaint_mask.
        # return intermediate_latents * mask + noised_init_latents * (1.0 - mask)

        # # Original noise guidance
        # # -----------------------
        # # Blend the predicted noise with the known noise at this timestep.
        # # This is equivalent to blending the predicted denoising step with the denoising step we would have taken if we
        # # were still on the init trajectory.
        # noise = pred_noise * mask + self._noise * (1.0 - mask)
        # return t_curr_latents + (t_prev - t_curr) * noise
