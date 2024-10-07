import itertools
from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFluxOutput
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFluxOutput
from invokeai.backend.flux.extensions.inpaint_extension import InpaintExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
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
    xlabs_controlnet_extensions: list[XLabsControlNetExtension],
    instantx_controlnet_extensions: list[InstantXControlNetExtension],
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

        # Run ControlNet models.
        controlnet_residuals: list[XLabsControlNetFluxOutput | InstantXControlNetFluxOutput | None] = []
        for controlnet_extension in itertools.chain(xlabs_controlnet_extensions, instantx_controlnet_extensions):
            controlnet_residuals.append(
                controlnet_extension.run_controlnet(
                    timestep_index=step - 1,
                    total_num_timesteps=total_steps,
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
            )
        xlabs_controlnet_residuals = [res for res in controlnet_residuals if isinstance(res, XLabsControlNetFluxOutput)]
        instantx_controlnet_residuals = [
            res for res in controlnet_residuals if isinstance(res, InstantXControlNetFluxOutput)
        ]

        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            xlabs_controlnet_residuals=xlabs_controlnet_residuals,
            instantx_controlnet_residuals=instantx_controlnet_residuals,
        )

        preview_img = img - t_curr * pred
        img = img + (t_prev - t_curr) * pred

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)
            preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)

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
