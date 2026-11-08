from math import floor
from typing import Callable, TypeAlias

import torch
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


PreviewFrame: TypeAlias = tuple[Image.Image, tuple[int, int] | None]
"""A preview image and the size to display it at (`None`: the image's own size)."""

PreviewRender: TypeAlias = Callable[[], PreviewFrame]
"""Renders one preview frame. Called only for a frame the preview throttle lets through."""

PreviewRenderer: TypeAlias = Callable[[BaseModelType, torch.Tensor], PreviewFrame]
"""Renders a preview of a latent sample denoised by `base_model`."""

SignalPreviewFunc: TypeAlias = Callable[[str, float | None, PreviewRender], None]


def preview_frame(sample: torch.Tensor) -> torch.Tensor:
    """The latent a preview shows: the last frame of a `(B, C, T, h, w)` window, any other sample as it is.

    A video denoiser may report a window of latent frames so a Tiny AutoEncoder can decode the last one
    with the frames before it as temporal context. Everything that reads a single latent -- the linear
    projection, and `resolve_latent_space`, which takes `shape[-3]` as the channel count -- gets the last.
    """
    return sample[:, :, -1] if sample.dim() == 5 else sample


def render_linear_preview(base_model: BaseModelType, sample: torch.Tensor) -> tuple[Image.Image, tuple[int, int]]:
    """Project `sample` onto RGB with the architecture's linear latent factors.

    The image is one `spatial_compression`-th of the generated image per side; the returned display
    size is the full generated size.
    """
    # Which projection, which bias, whether to smooth, and how much the VAE downscales are all facts
    # about the architecture, and they live in invokeai/backend/architectures/defs/. This used to be
    # a fifteen-branch if/elif ending in `raise ValueError(f"Unsupported base model: {base_model}")`,
    # which fired on the first preview step — after the model had loaded and generation had started.
    # An architecture that declares no latent space now fails at boot instead.
    sample = preview_frame(sample)
    latent_space = resolve_latent_space(base_model, sample)
    image = latent_space.preview(sample)
    return image, (image.width * latent_space.spatial_compression, image.height * latent_space.spatial_compression)


def diffusion_step_callback(
    signal_preview: SignalPreviewFunc,
    intermediate_state: PipelineIntermediateState,
    base_model: BaseModelType,
    is_canceled: Callable[[], bool],
    render: PreviewRenderer = render_linear_preview,
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

    signal_preview("Denoising", calc_percentage(intermediate_state), lambda: render(base_model, sample))
