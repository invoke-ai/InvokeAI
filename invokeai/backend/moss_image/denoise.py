"""MOSS-Image denoising loop.

A re-implementation of the sampling loop in the reference `MOSSImagePipeline.__call__` (see `__init__`), factored out
so InvokeAI can drive the vendored transformer with its own schedule windows, inpainting and progress previews. Over
the full schedule it computes exactly what the reference computes with its default skip-layer settings.
"""

from typing import Callable, Sequence

import torch
from tqdm import tqdm

from invokeai.backend.ernie_image.sampling_utils import unpatchify_latents
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer, _validate_skip_layers
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice

# The reference feeds the transformer the scheduler's timesteps: sigma * num_train_timesteps, in float32.
_NUM_TRAIN_TIMESTEPS = 1000


def denoise(
    transformer: MOSSImageTransformer,
    latents: torch.Tensor,
    sigmas: list[float],
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    cfg_scale: float,
    uncond_skip_layers: Sequence[int],
    step_callback: Callable[[PipelineIntermediateState], None],
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
) -> torch.Tensor:
    """Denoise model-space latents along `sigmas`.

    Args:
        transformer: The vendored MOSS-Image DiT, in any dtype; it casts its float32 inputs itself.
        latents: float32 model-space latents (B, 128, H/16, W/16) at `sigmas[0]`: the noise, or the img2img blend.
        sigmas: Descending schedule from `get_moss_sigmas`, terminal sigma included.
        positive: Positive text embeddings (B, L, text_dim), unpadded.
        negative: Negative text embeddings (B, L', text_dim). Required when `cfg_scale != 1`, unused otherwise.
        cfg_scale: Classifier-free guidance scale; 1 runs the positive branch alone.
        uncond_skip_layers: Zero-based DiT blocks the negative branch skips (the reference default is [22]); empty
            skips none.
        step_callback: Called after every step with that step's x0 estimate, unpatchified to (B, 32, H/8, W/8) and
            still BN-normalized. Cancelling is up to the callback: InvokeAI's step callbacks raise on cancel.
        inpaint_extension: Model-space inpainting state, merged into the latents after every step.

    Returns:
        float32 model-space latents at `sigmas[-1]`.
    """
    skip_layers = _validate_skip_layers(uncond_skip_layers, len(transformer.layers), "uncond_skip_layers")
    if cfg_scale != 1.0 and negative is None:
        raise ValueError("Negative conditioning is required when cfg_scale != 1.0.")
    # As in the reference, a guidance scale of exactly 1 is the positive branch alone.
    uncond_embeddings = negative if cfg_scale != 1.0 else None

    total_steps = len(sigmas) - 1
    timesteps = torch.tensor(sigmas, dtype=torch.float32, device=latents.device) * _NUM_TRAIN_TIMESTEPS
    for step_index in tqdm(range(total_steps), desc=f"Denoising{TorchDevice.get_session_device_label()}"):
        sigma, sigma_next = sigmas[step_index], sigmas[step_index + 1]
        timestep = timesteps[step_index]

        # Guidance combines float32 predictions even when the DiT runs in bf16, as the reference does.
        pred = transformer(latents, timestep, positive).sample.float()
        if uncond_embeddings is not None:
            uncond = transformer(latents, timestep, uncond_embeddings, skip_layers=skip_layers).sample.float()
            pred = uncond + cfg_scale * (pred - uncond)

        preview = latents - sigma * pred
        # The Euler update FlowMatchEulerDiscreteScheduler.step performs, which the reference steps with.
        latents = latents + (sigma_next - sigma) * pred
        if inpaint_extension is not None:
            latents = inpaint_extension.merge_intermediate_latents_with_init_latents(latents, sigma_next)
            # Show the kept region as the init image itself rather than the model's estimate of it.
            preview = inpaint_extension.merge_intermediate_latents_with_init_latents(preview, 0.0)

        step_callback(
            PipelineIntermediateState(
                step=step_index + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(sigma * _NUM_TRAIN_TIMESTEPS),
                latents=unpatchify_latents(preview),
            )
        )
    return latents
