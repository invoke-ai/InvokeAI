from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.flux.inpaint import merge_intermediate_latents_with_init_latents
from invokeai.backend.flux.model import Flux


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
    step_callback: Callable[[], None],
    guidance: float,
    # For inpainting:
    init_latents: torch.Tensor | None,
    noise: torch.Tensor,
    inpaint_mask: torch.Tensor | None,
):
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

        img = img + (t_prev - t_curr) * pred

        if inpaint_mask is not None:
            assert init_latents is not None
            img = merge_intermediate_latents_with_init_latents(
                init_latents=init_latents,
                intermediate_latents=img,
                timestep=t_prev,
                noise=noise,
                inpaint_mask=inpaint_mask,
            )

        step_callback()

    return img
