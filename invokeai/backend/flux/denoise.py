from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.flux.inpaint_extension import InpaintExtension
from invokeai.backend.flux.model import Flux
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
    inpaint_extension: InpaintExtension | None,
):
    step = 0
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
        preview_img = img - t_curr * pred
        img = img + (t_prev - t_curr) * pred

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

        step_callback(
            PipelineIntermediateState(
                step=step,
                order=1,
                total_steps=len(timesteps),
                timestep=int(t_curr),
                latents=preview_img,
            ),
        )
        step += 1

    return img
