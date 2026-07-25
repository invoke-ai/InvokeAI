from typing import Literal, Optional

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Ideogram4ConditioningField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.step_callback import (
    FLUX2_LATENT_RGB_BIAS,
    FLUX2_LATENT_RGB_FACTORS,
    sample_to_lowres_estimated_image,
)
from invokeai.backend.ideogram4 import run_ideogram4_denoise
from invokeai.backend.ideogram4.latent_norm import get_latent_norm
from invokeai.backend.ideogram4.sampler_configs import PRESETS
from invokeai.backend.ideogram4.sampling_utils import unpatchify_and_denormalize
from invokeai.backend.ideogram4.transformer_pair import Ideogram4TransformerPair
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Ideogram4ConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# Named sampler presets bundle step count, guidance schedule (with polish tail), and the
# logit-normal schedule mean/std. V4_QUALITY_48 is the reference default.
IDEOGRAM4_SAMPLER_PRESETS = Literal["V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"]


def _effective_guidance_schedule(
    base_schedule: tuple[float, ...], preset_num_steps: int, num_steps: int, guidance_scale: Optional[float]
) -> tuple[float, ...]:
    """Build the per-step guidance schedule for the (possibly overridden) step count.

    The preset schedule is ``(polish_gw,)*N_polish + (main_gw,)*N_main`` in loop-index order
    (index 0 = the final/polish step). A ``guidance_scale`` override replaces the main weight while
    the preset's polish tail is preserved; a changed step count rescales the polish tail
    proportionally (always keeping at least one polish and one main step).

    ``num_steps`` must be >= 2 (enforced by the invocation's ``steps`` field) so both a polish and a
    main step always exist — otherwise a single step would be all-polish and silently drop the
    ``guidance_scale`` override.
    """
    polish_gw = base_schedule[0]
    main_gw = float(guidance_scale) if guidance_scale is not None else float(base_schedule[-1])
    if num_steps == preset_num_steps and guidance_scale is None:
        return base_schedule
    n_polish_base = sum(1 for gw in base_schedule if gw == base_schedule[0])
    # Cap the polish tail at num_steps - 1 so at least one main step always remains and the
    # guidance_scale override is never silently dropped.
    polish_count = max(1, min(round(n_polish_base * num_steps / preset_num_steps), num_steps - 1))
    main_count = num_steps - polish_count
    return (polish_gw,) * polish_count + (main_gw,) * main_count


@invocation(
    "ideogram4_denoise",
    title="Denoise - Ideogram 4",
    tags=["image", "ideogram4"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4DenoiseInvocation(BaseInvocation):
    """Runs the Ideogram 4 dual-branch flow-matching denoising loop (text-to-image)."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: Ideogram4ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    sampler_preset: IDEOGRAM4_SAMPLER_PRESETS = InputField(
        default="V4_QUALITY_48",
        description="Sampler preset (steps + guidance schedule + schedule mean/std).",
        title="Sampler Preset",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    # Optional advanced overrides of the sampler preset. None = use the preset's value.
    steps: Optional[int] = InputField(
        default=None,
        ge=2,
        le=100,
        description="Override the preset's step count (minimum 2, so a polish and a main step both "
        "exist). Leave empty to use the preset.",
    )
    guidance_scale: Optional[float] = InputField(
        default=None,
        ge=1.0,
        le=20.0,
        description="Override the main guidance weight (the preset's polish tail is preserved). "
        "Empty = use the preset.",
    )
    mu: Optional[float] = InputField(
        default=None,
        ge=-4.0,
        le=4.0,
        description="Override the logit-normal schedule mean (resolution-adjusted internally). Empty = use the preset.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        device = TorchDevice.choose_torch_device()
        preset = PRESETS[self.sampler_preset]

        # Apply optional advanced overrides on top of the preset.
        num_steps = self.steps if self.steps is not None else preset.num_steps
        mu = self.mu if self.mu is not None else preset.mu
        guidance_schedule = _effective_guidance_schedule(
            preset.guidance_schedule, preset.num_steps, num_steps, self.guidance_scale
        )

        # Load conditioning (the stacked Qwen3-VL features).
        cond_data = context.conditioning.load(self.positive_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        info = cond_data.conditionings[0]
        assert isinstance(info, Ideogram4ConditioningInfo)
        llm_features = info.prompt_embeds.to(device=device, dtype=torch.float32)

        # Progress-preview setup: Ideogram uses a FLUX.2-style 32-channel VAE, so the FLUX.2
        # latent->RGB factors give a usable (approximate) low-res preview of the forming image at each
        # step, without a full VAE decode. Denormalization params come from get_latent_norm (no VAE).
        latent_shift, latent_scale = get_latent_norm()
        rgb_factors = torch.tensor(FLUX2_LATENT_RGB_FACTORS, dtype=torch.float32)
        rgb_bias = torch.tensor(FLUX2_LATENT_RGB_BIAS, dtype=torch.float32)

        def step_callback(step: int, total: int, packed_latents: torch.Tensor) -> None:
            preview = None
            try:
                # packed_latents: (1, LATENT_DIM, grid_h, grid_w) -> VAE latent (1, 32, H/8, W/8).
                vae_latent = unpatchify_and_denormalize(
                    packed_latents.float(), latent_shift.to(packed_latents.device), latent_scale.to(packed_latents.device)
                )
                preview = sample_to_lowres_estimated_image(
                    samples=vae_latent,
                    latent_rgb_factors=rgb_factors.to(vae_latent.device),
                    latent_rgb_bias=rgb_bias.to(vae_latent.device),
                )
            except Exception:
                # A preview must never break generation — fall back to a plain progress signal.
                preview = None
            if preview is not None:
                context.util.signal_progress(
                    "Running Ideogram 4 denoising",
                    step / total,
                    preview,
                    (preview.width * 8, preview.height * 8),
                )
            else:
                context.util.signal_progress("Running Ideogram 4 denoising", step / total)

        transformer_info = context.models.load(self.transformer.transformer)
        with transformer_info.model_on_device() as (_, transformers):
            assert isinstance(transformers, Ideogram4TransformerPair)
            packed = run_ideogram4_denoise(
                conditional_transformer=transformers.conditional,
                unconditional_transformer=transformers.unconditional,
                llm_features=llm_features,
                height=self.height,
                width=self.width,
                num_steps=num_steps,
                mu=mu,
                std=preset.std,
                guidance_schedule=guidance_schedule,
                seed=self.seed,
                device=device,
                step_callback=step_callback,
            )

        packed = packed.detach().to("cpu")
        name = context.tensors.save(tensor=packed)
        return LatentsOutput.build(latents_name=name, latents=packed, seed=None)
