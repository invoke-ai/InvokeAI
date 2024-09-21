from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.trajectory_guidance_extension import TrajectoryGuidanceExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise(
    model: Flux,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    guidance: float,
    traj_guidance_extension: TrajectoryGuidanceExtension | None,  # noqa: F821
):
    # step 0 is the initial state
    total_steps = len(timesteps) - 1
    step_callback(
        PipelineIntermediateState(
            step=0,
            order=1,
            total_steps=total_steps,
            timestep=int(timesteps[0]),
            latents=img,
        ),
    )
    step = 1
    # guidance_vec is ignored for schnell.
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:], strict=True))):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        if traj_guidance_extension is not None:
            pred = traj_guidance_extension.update_noise(
                t_curr_latents=img, pred_noise=pred, t_curr=t_curr, t_prev=t_prev
            )

        preview_img = img - t_curr * pred
        img = img + (t_prev - t_curr) * pred

        step_callback(
            PipelineIntermediateState(
                step=step,
                order=1,
                total_steps=total_steps,
                timestep=int(t_curr),
                latents=preview_img,
            ),
        )
        step += 1

    return img
