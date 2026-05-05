"""ERNIE-Image denoising loop.

A direct re-implementation of the denoise loop in
`diffusers.pipelines.ernie_image.ErnieImagePipeline.__call__`, factored out so
InvokeAI can drive the transformer with its own scheduling, CFG, and inpainting
machinery instead of going through the upstream pipeline as a black box.
"""

import inspect
import math
from typing import Any, Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

from invokeai.backend.ernie_image.sampling_utils import unpatchify_latents
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise(
    model: torch.nn.Module,
    img: torch.Tensor,
    text_bth: torch.Tensor,
    text_lens: torch.Tensor,
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    cfg_scale: list[float],
    neg_text_bth: Optional[torch.Tensor] = None,
    neg_text_lens: Optional[torch.Tensor] = None,
    scheduler: Any = None,
    inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None,
) -> torch.Tensor:
    """Run the ERNIE-Image denoise loop.

    Args:
        model: `ErnieImageTransformer2DModel` from diffusers.
        img: Patched latents `[B, 128, H/2, W/2]` (already 2x2-patchified, BN-normalized
            if coming from VAE encode -- see `sampling_utils`).
        text_bth: Padded text embeddings `[B, Tmax, text_in_dim]` for the positive prompt.
        text_lens: Actual lengths of each prompt's text encoding `[B]`.
        timesteps: Sigma schedule (descending, in [0,1]). Length = num_steps + 1; the final
            entry is the target sigma after the last step.
        step_callback: Progress callback.
        cfg_scale: Per-step CFG scales (length matches `len(timesteps) - 1` or shorter --
            shorter values are clamped).
        neg_text_bth / neg_text_lens: Negative-prompt conditioning, required when any CFG
            scale is not 1.0.
        scheduler: Optional FlowMatch* scheduler. If None, falls back to manual Euler over
            the supplied sigmas.
        inpaint_extension: Optional inpainting helper that merges the noised init latents
            into the prediction at each step.

    Returns:
        Denoised, still-patched latents `[B, 128, H/2, W/2]`. Caller is responsible for
        BN-denormalization and unpatchify before VAE decode.
    """
    total_steps = len(timesteps) - 1
    use_scheduler = scheduler is not None

    # ERNIE-Image's transformer uses standard sinusoidal time embeddings (`Timesteps(...)` from
    # diffusers) which expect timesteps in `[0, num_train_timesteps]`. Sigmas are in [0, 1].
    # The model gets `sigma * model_timestep_scale`; sigma stays in [0, 1] for the Euler math.
    model_timestep_scale = float(scheduler.config.num_train_timesteps) if use_scheduler else 1000.0

    if use_scheduler:
        # Drop the final sigma -- diffusers schedulers append it themselves.
        sigmas = np.array(timesteps[:-1], dtype=np.float32)
        set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
        if "sigmas" in set_timesteps_sig.parameters:
            scheduler.set_timesteps(sigmas=sigmas.tolist(), device=img.device)
        else:
            # Heun/LCM in diffusers don't accept sigmas -- fall back to step count.
            scheduler.set_timesteps(num_inference_steps=len(sigmas), device=img.device)

        pbar = tqdm(total=total_steps, desc="ERNIE-Image denoising")
        for step_index in range(len(scheduler.timesteps)):
            timestep = scheduler.timesteps[step_index]
            # The scheduler's timestep is already in `[0, num_train_timesteps]`; pass directly.
            t_model = timestep.item()
            t_sigma = t_model / model_timestep_scale
            t_vec = torch.full((img.shape[0],), t_model, dtype=img.dtype, device=img.device)

            pred = _forward(model, img, t_vec, text_bth, text_lens)
            step_cfg = cfg_scale[min(step_index, len(cfg_scale) - 1)]
            if not math.isclose(step_cfg, 1.0):
                if neg_text_bth is None or neg_text_lens is None:
                    raise ValueError("Negative conditioning is required when cfg_scale != 1.0")
                neg_pred = _forward(model, img, t_vec, neg_text_bth, neg_text_lens)
                pred = neg_pred + step_cfg * (pred - neg_pred)

            img = scheduler.step(model_output=pred, timestep=timestep, sample=img).prev_sample

            t_prev = scheduler.sigmas[step_index + 1].item() if step_index + 1 < len(scheduler.sigmas) else 0.0
            if inpaint_extension is not None:
                img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

            pbar.update(1)
            # Predicted x0 estimate, unpatched so the preview decoder can use standard
            # 32-channel latent RGB factors.
            preview = unpatchify_latents(img - t_sigma * pred)
            step_callback(
                PipelineIntermediateState(
                    step=step_index + 1,
                    order=1,
                    total_steps=total_steps,
                    timestep=int(t_model),
                    latents=preview,
                ),
            )
        pbar.close()
        return img

    # Manual Euler over the supplied sigmas. This mirrors the upstream pipeline when no
    # explicit scheduler is configured.
    pbar = tqdm(total=total_steps, desc="ERNIE-Image denoising")
    for i in range(total_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        # Sigmas are [0, 1]; scale up to the model's training-timestep range.
        t_vec = torch.full(
            (img.shape[0],), t_curr * model_timestep_scale, dtype=img.dtype, device=img.device
        )

        pred = _forward(model, img, t_vec, text_bth, text_lens)
        step_cfg = cfg_scale[min(i, len(cfg_scale) - 1)]
        if not math.isclose(step_cfg, 1.0):
            if neg_text_bth is None or neg_text_lens is None:
                raise ValueError("Negative conditioning is required when cfg_scale != 1.0")
            neg_pred = _forward(model, img, t_vec, neg_text_bth, neg_text_lens)
            pred = neg_pred + step_cfg * (pred - neg_pred)

        # Standard rectified-flow Euler step in sigma space.
        img = img + (t_next - t_curr) * pred

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_next)

        pbar.update(1)
        preview = unpatchify_latents(img - t_curr * pred)
        step_callback(
            PipelineIntermediateState(
                step=i + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(t_curr * 1000),
                latents=preview,
            ),
        )
    pbar.close()
    return img


def _forward(
    model: torch.nn.Module,
    img: torch.Tensor,
    t_vec: torch.Tensor,
    text_bth: torch.Tensor,
    text_lens: torch.Tensor,
) -> torch.Tensor:
    out = model(
        hidden_states=img,
        timestep=t_vec,
        text_bth=text_bth,
        text_lens=text_lens,
        return_dict=False,
    )
    return out[0] if isinstance(out, tuple) else out
