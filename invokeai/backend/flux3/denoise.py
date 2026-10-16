"""The FLUX.3 sampling loop.

Euler on a fixed schedule, as in the reference implementation's ``denoise`` / ``denoise_flow_map``
(``src/flux3/sampling.py``, Apache-2.0):

- Klein Base takes classifier-free guidance, ``uncond + scale * (cond - uncond)``. The two predictions run
  as separate forwards; the reference packs them into one sequence with per-sample attention, which gives
  the same result.
- Klein is a flow map: each step also tells the transformer the timestep it jumps to, and it runs one
  forward without guidance.

The sample is kept in float32 across steps and cast to the compute dtype for each forward, as the reference
does. Reference-image tokens are context: they go into every forward and are never stepped.
"""

from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice


def denoise(
    model: torch.nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    flow_map: bool,
    compute_dtype: torch.dtype,
    cfg_scale: float = 1.0,
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    ref_tokens: torch.Tensor | None = None,
    ref_ids: torch.Tensor | None = None,
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
) -> torch.Tensor:
    """Denoise packed, BN-normalized image tokens ``img`` of shape ``(B, S, 128)`` along ``timesteps``.

    Returns the denoised tokens in float32.
    """
    use_cfg = cfg_scale > 1.0
    if use_cfg and flow_map:
        raise ValueError("A flow-map FLUX.3 transformer samples without classifier-free guidance.")
    if use_cfg and (neg_txt is None or neg_txt_ids is None):
        raise ValueError("Classifier-free guidance needs a negative conditioning.")

    img = img.float()
    batch_size = img.shape[0]
    total_steps = len(timesteps) - 1

    def predict(t_vec: torch.Tensor, target_vec: torch.Tensor | None, text: torch.Tensor, text_ids: torch.Tensor):
        output = model(
            hidden_states=img.to(compute_dtype),
            encoder_hidden_states=text,
            timestep=t_vec,
            img_ids=img_ids,
            txt_ids=text_ids,
            image_cond_hidden_states=ref_tokens,
            image_cond_ids=ref_ids,
            target_timestep=target_vec,
            return_dict=False,
        )
        return output[0].float()

    for step_index in tqdm(range(total_steps), desc=f"Denoising{TorchDevice.get_session_device_label()}"):
        t_curr, t_next = timesteps[step_index], timesteps[step_index + 1]
        t_vec = torch.full((batch_size,), t_curr, device=img.device, dtype=compute_dtype)
        target_vec = torch.full((batch_size,), t_next, device=img.device, dtype=compute_dtype) if flow_map else None

        pred = predict(t_vec, target_vec, txt, txt_ids)
        if use_cfg:
            assert neg_txt is not None and neg_txt_ids is not None
            neg_pred = predict(t_vec, None, neg_txt, neg_txt_ids)
            pred = neg_pred + cfg_scale * (pred - neg_pred)

        preview = img - t_curr * pred
        img = img + (t_next - t_curr) * pred

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_next)
            preview = inpaint_extension.merge_intermediate_latents_with_init_latents(preview, 0.0)

        step_callback(
            PipelineIntermediateState(
                step=step_index + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(t_curr * 1000),
                latents=preview,
            )
        )

    return img
