from math import floor
from typing import Callable, TypeAlias

from PIL import Image

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.architectures import resolve_latent_space
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def calc_percentage(intermediate_state: PipelineIntermediateState) -> float:
    """Calculate the percentage of completion of denoising."""

    step = intermediate_state.step
    total_steps = intermediate_state.total_steps
    order = intermediate_state.order

    if total_steps == 0:
        return 0.0
    if order == 2:
        # Prevent division by zero when total_steps is 1 or 2
        denominator = floor(total_steps / 2)
        if denominator == 0:
            return 0.0
        return floor(step / 2) / denominator
    # order == 1
    return step / total_steps


SignalProgressFunc: TypeAlias = Callable[[str, float | None, Image.Image | None, tuple[int, int] | None], None]


def diffusion_step_callback(
    signal_progress: SignalProgressFunc,
    intermediate_state: PipelineIntermediateState,
    base_model: BaseModelType,
    is_canceled: Callable[[], bool],
) -> None:
    if is_canceled():
        raise CanceledException

    # Some schedulers report not only the noisy latents at the current timestep,
    # but also their estimate so far of what the de-noised latents will be. Use
    # that estimate if it is available.
    if intermediate_state.predicted_original is not None:
        sample = intermediate_state.predicted_original
    else:
        sample = intermediate_state.latents

    # Which projection, which bias, whether to smooth, and how much the VAE downscales are all facts
    # about the architecture, and they live in invokeai/backend/architectures/defs/. This used to be
    # a fifteen-branch if/elif ending in `raise ValueError(f"Unsupported base model: {base_model}")`,
    # which fired on the first preview step — after the model had loaded and generation had started.
    # An architecture that declares no latent space now fails at boot instead.
    latent_space = resolve_latent_space(base_model, sample)
    image = latent_space.preview(sample)

    signal_progress(
        "Denoising",
        calc_percentage(intermediate_state),
        image,
        (image.width * latent_space.spatial_compression, image.height * latent_space.spatial_compression),
    )
