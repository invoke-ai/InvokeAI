"""Wan 2.2 denoise invocation.

Supports both single-transformer (TI2V-5B) and dual-expert MoE (A14B) denoising.
For A14B the high-noise expert handles timesteps ``t >= boundary_timestep`` and
the low-noise expert handles ``t < boundary_timestep``, where
``boundary_timestep = boundary_ratio * num_train_timesteps`` (typically 1000).

To keep VRAM usage manageable both experts are pinned in the model cache
(system RAM) but only one is GPU-resident at a time. The boundary is normally
crossed once per denoise, so the swap incurs a single CPU→GPU transfer.

Phase 8 will add inpaint via :class:`RectifiedFlowInpaintExtension`.

The transformer call signature mirrors Diffusers' ``WanPipeline``:

    transformer(
        hidden_states=latents_5d,            # [B, C, 1, H/s, W/s]
        timestep=t.expand(B),                # scheduler-time
        encoder_hidden_states=prompt_embeds, # [B, seq_len, 4096]
        attention_kwargs=None,
        return_dict=False,
    )[0]
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WanConditioningField,
)
from invokeai.app.invocations.model import WanTransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, WanVariantType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import WanConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.wan.sampling_utils import get_spatial_scale_factor, make_noise


def _resolve_variant(context: InvocationContext, transformer_field: WanTransformerField) -> WanVariantType:
    """Look up the Wan variant from the main model config that produced this transformer."""
    config = context.models.get_config(transformer_field.transformer)
    variant = getattr(config, "variant", None)
    if not isinstance(variant, WanVariantType):
        raise ValueError(
            f"Could not determine Wan variant from model {config.name!r}: variant is {variant!r}."
        )
    return variant


def _scheduler_path_for_transformer(context: InvocationContext, transformer_field: WanTransformerField) -> Path | None:
    """Return the on-disk ``scheduler/`` directory for the main model, or None."""
    config = context.models.get_config(transformer_field.transformer)
    model_root = context.models.get_absolute_path(config)
    if model_root.is_file():
        return None
    candidate = model_root / "scheduler"
    if (candidate / "scheduler_config.json").exists():
        return candidate
    return None


class _ExpertSwapper:
    """Manages GPU residency of one or two Wan transformer experts.

    Both experts are kept in the model cache (system RAM); only one is on
    device at a time. ``get(label)`` returns the model for the requested label,
    swapping GPU residency when the label changes. The first ``get`` call also
    enters the underlying ``model_on_device`` context for the requested expert.
    """

    HIGH = "high"
    LOW = "low"

    def __init__(self, high_info: Any, low_info: Any | None) -> None:
        self._high_info = high_info
        self._low_info = low_info
        self._active_label: str | None = None
        self._active_ctx: Any | None = None
        self._active_model: Any | None = None

    def get(self, label: str) -> Any:
        if label not in (self.HIGH, self.LOW):
            raise ValueError(f"Unknown expert label: {label!r}")
        if label == self.LOW and self._low_info is None:
            raise ValueError("Low-noise expert was requested but is not available.")
        if label == self._active_label:
            assert self._active_model is not None
            return self._active_model

        # Release current GPU residency before bringing the other expert on device.
        self._release()

        info = self._high_info if label == self.HIGH else self._low_info
        ctx = info.model_on_device()
        _cached, model = ctx.__enter__()
        self._active_label = label
        self._active_ctx = ctx
        self._active_model = model
        return model

    def _release(self) -> None:
        if self._active_ctx is not None:
            self._active_ctx.__exit__(None, None, None)
        self._active_label = None
        self._active_ctx = None
        self._active_model = None

    def close(self) -> None:
        self._release()


@invocation(
    "wan_denoise",
    title="Denoise - Wan 2.2",
    tags=["image", "wan"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanDenoiseInvocation(BaseInvocation):
    """Run the denoising process with a Wan 2.2 model.

    Drives a flow-matching Euler schedule via Diffusers'
    ``FlowMatchEulerDiscreteScheduler``. CFG is supported when negative
    conditioning is provided and ``guidance_scale != 1.0``.

    For Wan 2.2 A14B the high-noise expert handles timesteps at and above
    ``boundary_ratio * num_train_timesteps``; the low-noise expert handles
    timesteps below. Both experts share the model cache; only the active one is
    GPU-resident at any time.
    """

    transformer: WanTransformerField = InputField(
        description="Wan transformer field (transformer + optional dual-expert metadata).",
        input=Input.Connection,
        title="Transformer",
    )
    positive_conditioning: WanConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[WanConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )

    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )

    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")

    guidance_scale: float = InputField(
        default=4.0,
        ge=1.0,
        description="Classifier-free guidance scale. 4.0 is the Wan 2.2 default for A14B; "
        "TI2V-5B can tolerate higher values up to ~5.5.",
        title="Guidance Scale",
    )
    guidance_scale_low_noise: Optional[float] = InputField(
        default=None,
        ge=1.0,
        description="Optional separate CFG scale for the low-noise expert (Wan 2.2 A14B only). "
        "If unset, the primary 'Guidance Scale' is reused. Ignored for TI2V-5B.",
        title="Guidance Scale (Low Noise)",
    )
    width: int = InputField(default=1024, multiple_of=8, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=8, description="Height of the generated image.")
    steps: int = InputField(default=40, gt=0, description="Number of denoising steps.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        if self.denoising_start >= self.denoising_end:
            raise ValueError(
                f"denoising_start ({self.denoising_start}) must be less than denoising_end ({self.denoising_end})."
            )

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        variant = _resolve_variant(context, self.transformer)
        spatial_scale = get_spatial_scale_factor(variant)

        scheduler = self._build_scheduler(context, device)

        pos_cond = self._load_conditioning(
            context, self.positive_conditioning, device=device, dtype=inference_dtype
        )
        do_cfg = self.guidance_scale != 1.0 and self.negative_conditioning is not None
        neg_cond: WanConditioningInfo | None = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_cond = self._load_conditioning(
                context, self.negative_conditioning, device=device, dtype=inference_dtype
            )

        # Schedule timesteps. set_timesteps populates scheduler.timesteps and
        # scheduler.sigmas (where sigmas is in [0, 1] flow-matching space).
        scheduler.set_timesteps(num_inference_steps=self.steps, device=device)
        timesteps = scheduler.timesteps
        # sigmas has length steps + 1.
        sigmas = scheduler.sigmas

        # Apply denoising_start / denoising_end clipping.
        if self.denoising_start > 0 or self.denoising_end < 1:
            start_idx = int(self.denoising_start * self.steps)
            end_idx = int(self.denoising_end * self.steps)
            timesteps = timesteps[start_idx:end_idx]
            sigmas = sigmas[start_idx : end_idx + 1]
        total_steps = len(timesteps)

        # Load init latents (img2img) and convert 4D → 5D.
        init_latents_5d: torch.Tensor | None = None
        if self.latents is not None:
            loaded = context.tensors.load(self.latents.latents_name).to(device=device, dtype=inference_dtype)
            if loaded.ndim == 4:
                loaded = loaded.unsqueeze(2)
            init_latents_5d = loaded

        # Determine the latent channel count. Prefer init_latents shape; otherwise
        # fall back to the variant default. (We avoid loading the transformer just
        # to read .config.in_channels; the variant gives us the right answer.)
        latent_channels = (
            init_latents_5d.shape[1]
            if init_latents_5d is not None
            else (48 if variant == WanVariantType.TI2V_5B else 16)
        )

        noise = make_noise(
            batch_size=1,
            latent_channels=latent_channels,
            height=self.height,
            width=self.width,
            spatial_scale_factor=spatial_scale,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )

        # Combine init latents + noise per the schedule's starting sigma.
        if init_latents_5d is not None:
            if self.add_noise:
                s_0 = float(sigmas[0])
                latents = s_0 * noise + (1.0 - s_0) * init_latents_5d
            else:
                latents = init_latents_5d
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        if total_steps <= 0:
            return latents.squeeze(2)

        # Inpaint extension (4D space — the existing extension is shape-agnostic
        # but operates on the squeezed-T shape we use for masks).
        inpaint_mask = self._prep_inpaint_mask(context, latents.squeeze(2))
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            if init_latents_5d is None:
                raise ValueError("Initial latents are required when using an inpaint mask (img2img inpainting).")
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents_5d.squeeze(2),
                inpaint_mask=inpaint_mask,
                noise=noise.squeeze(2),
            )

        step_callback = self._build_step_callback(context)

        # Resolve experts and the boundary timestep that triggers the MoE swap.
        high_info = context.models.load(self.transformer.transformer)
        low_info = (
            context.models.load(self.transformer.transformer_low_noise)
            if self.transformer.transformer_low_noise is not None
            else None
        )
        # FlowMatchEulerDiscreteScheduler stores num_train_timesteps in its config
        # (default 1000). Diffusers' WanPipeline computes:
        #   boundary_timestep = boundary_ratio * num_train_timesteps
        num_train_timesteps = int(scheduler.config.num_train_timesteps)
        boundary_timestep = (
            self.transformer.boundary_ratio * num_train_timesteps if low_info is not None else None
        )

        with ExitStack() as exit_stack:
            swapper = _ExpertSwapper(high_info, low_info)
            exit_stack.callback(swapper.close)

            for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising (Wan 2.2)", total=total_steps)):
                timestep = t.expand(latents.shape[0])

                # Pick the active expert: high-noise for t >= boundary_timestep,
                # low-noise below. Single-transformer models always use HIGH.
                if low_info is not None and float(t) < float(boundary_timestep):
                    active_label = _ExpertSwapper.LOW
                    active_cfg = (
                        self.guidance_scale_low_noise
                        if self.guidance_scale_low_noise is not None
                        else self.guidance_scale
                    )
                else:
                    active_label = _ExpertSwapper.HIGH
                    active_cfg = self.guidance_scale

                transformer = swapper.get(active_label)

                noise_pred_cond = transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=pos_cond.prompt_embeds.unsqueeze(0),
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

                if do_cfg and neg_cond is not None:
                    noise_pred_uncond = transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=neg_cond.prompt_embeds.unsqueeze(0),
                        attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + active_cfg * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if inpaint_extension is not None:
                    sigma_prev = float(sigmas[step_idx + 1])
                    latents_4d = latents.squeeze(2)
                    latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(
                        latents_4d, sigma_prev
                    )
                    latents = latents_4d.unsqueeze(2)

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t.item()),
                        latents=latents.squeeze(2),
                    )
                )

        # Squeeze T for downstream 4D consumers.
        return latents.squeeze(2)

    def _build_scheduler(self, context: InvocationContext, device: torch.device):
        """Construct ``FlowMatchEulerDiscreteScheduler`` for this run.

        Loads the model's on-disk scheduler config when available so per-model
        ``shift`` settings are honoured; falls back to defaults otherwise.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_dir = _scheduler_path_for_transformer(context, self.transformer)
        if scheduler_dir is not None:
            return FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(scheduler_dir), local_files_only=True
            )
        return FlowMatchEulerDiscreteScheduler()

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: WanConditioningField,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> WanConditioningInfo:
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        assert len(cond_data.conditionings) == 1
        cond_info = cond_data.conditionings[0]
        assert isinstance(cond_info, WanConditioningInfo)
        return cond_info.to(device=device, dtype=dtype)

    def _prep_inpaint_mask(self, context: InvocationContext, latents_4d: torch.Tensor) -> torch.Tensor | None:
        """Resize the user-supplied mask down to latent resolution.

        Convention matches Anima/FLUX: the original mask has 0 = preserve and
        1 = denoise; the extension expects the inverted form.
        """
        if self.denoise_mask is None:
            return None
        mask = context.tensors.load(self.denoise_mask.mask_name)
        mask = 1.0 - mask
        _, _, latent_h, latent_w = latents_4d.shape
        mask = tv_resize(
            img=mask,
            size=[latent_h, latent_w],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        return mask.to(device=latents_4d.device, dtype=latents_4d.dtype)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Wan)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator:
        # Phase 5 will populate this with the actual LoRA application path.
        return iter([])
