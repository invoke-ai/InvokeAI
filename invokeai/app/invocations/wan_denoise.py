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

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

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
    WanRefImageConditioningField,
)
from invokeai.app.invocations.model import LoRAField, WanTransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, WanVariantType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.wan_lora_constants import WAN_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import WanConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.wan.sampling_utils import get_spatial_scale_factor, make_noise

# Type alias: a factory that produces a fresh iterator of (LoRA patch, weight)
# pairs each time it is called. We need fresh iterators because the patcher
# consumes the iterator once per ``apply_smart_model_patches`` invocation, and
# the expert may be swapped (and re-entered) multiple times in a render.
LoRAIteratorFactory = Callable[[], Iterable[Tuple[ModelPatchRaw, float]]]


def _resolve_variant(context: InvocationContext, transformer_field: WanTransformerField) -> WanVariantType:
    """Look up the Wan variant from the main model config that produced this transformer."""
    config = context.models.get_config(transformer_field.transformer)
    variant = getattr(config, "variant", None)
    if not isinstance(variant, WanVariantType):
        raise ValueError(f"Could not determine Wan variant from model {config.name!r}: variant is {variant!r}.")
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
    """Manages GPU residency and LoRA patching of one or two Wan transformer experts.

    Both experts are kept in the model cache (system RAM); only one is on
    device at a time. ``get(label)`` returns the model for the requested label,
    swapping GPU residency when the label changes and applying that expert's
    LoRA patches via ``LayerPatcher.apply_smart_model_patches``.

    Ordering on swap: exit the active expert's LoRA context (restores weights)
    -> exit ``model_on_device`` (returns expert to RAM) -> load the new expert
    (fresh handle) -> enter its device context -> apply its LoRAs. This
    mirrors the pattern used by ``flux_denoise``/``anima_denoise`` but adds
    the extra context layer needed for dual experts.

    Model handles are obtained lazily inside ``get()`` rather than cached at
    construction. With dual ~9 GB GGUF experts plus a UMT5-XXL encoder
    competing for the RAM cache, holding both ``LoadedModel`` handles upfront
    can leave one of them stale by the time the swap happens — InvokeAI's
    model cache emits a ``has already been dropped from the RAM cache``
    warning and reloads from disk per swap. See issue #7513 for the broader
    pattern.
    """

    HIGH = "high"
    LOW = "low"

    def __init__(
        self,
        context: InvocationContext,
        high_model: Any,
        low_model: Any | None,
        inference_dtype: torch.dtype,
        high_lora_factory: LoRAIteratorFactory | None = None,
        low_lora_factory: LoRAIteratorFactory | None = None,
        high_is_quantized: bool = False,
        low_is_quantized: bool = False,
    ) -> None:
        self._context = context
        self._high_model = high_model
        self._low_model = low_model
        self._inference_dtype = inference_dtype
        self._high_lora_factory = high_lora_factory
        self._low_lora_factory = low_lora_factory
        self._high_is_quantized = high_is_quantized
        self._low_is_quantized = low_is_quantized
        self._active_label: str | None = None
        self._active_device_ctx: Any | None = None
        self._active_lora_ctx: Any | None = None
        self._active_model: Any | None = None

    def get(self, label: str) -> Any:
        if label not in (self.HIGH, self.LOW):
            raise ValueError(f"Unknown expert label: {label!r}")
        if label == self.LOW and self._low_model is None:
            raise ValueError("Low-noise expert was requested but is not available.")
        if label == self._active_label:
            assert self._active_model is not None
            return self._active_model

        # Release current GPU residency before bringing the other expert on device.
        self._release()

        # Load the requested expert lazily so its ``LoadedModel`` handle is
        # always fresh — see class docstring for the cache-eviction reasoning.
        model_id = self._high_model if label == self.HIGH else self._low_model
        info = self._context.models.load(model_id)
        device_ctx = info.model_on_device()
        cached_weights, model = device_ctx.__enter__()

        # Apply LoRA patches for this expert. GGUF transformers need sidecar
        # patching since direct patching of GGMLTensors isn't supported.
        lora_factory = self._high_lora_factory if label == self.HIGH else self._low_lora_factory
        is_quantized = self._high_is_quantized if label == self.HIGH else self._low_is_quantized
        lora_ctx: Any | None = None
        if lora_factory is not None:
            lora_ctx = LayerPatcher.apply_smart_model_patches(
                model=model,
                patches=lora_factory(),
                prefix=WAN_LORA_TRANSFORMER_PREFIX,
                dtype=self._inference_dtype,
                cached_weights=cached_weights,
                force_sidecar_patching=is_quantized,
            )
            lora_ctx.__enter__()

        self._active_label = label
        self._active_device_ctx = device_ctx
        self._active_lora_ctx = lora_ctx
        self._active_model = model
        return model

    def _release(self) -> None:
        # LoRA context first so weights are restored before the model leaves GPU.
        if self._active_lora_ctx is not None:
            self._active_lora_ctx.__exit__(None, None, None)
        if self._active_device_ctx is not None:
            self._active_device_ctx.__exit__(None, None, None)
        self._active_label = None
        self._active_device_ctx = None
        self._active_lora_ctx = None
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

    ref_image: Optional[WanRefImageConditioningField] = InputField(
        default=None,
        description=FieldDescriptions.wan_ref_image,
        input=Input.Connection,
        title="Reference Image",
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
        ge=0.0,
        description="Optional separate CFG scale for the low-noise expert (Wan 2.2 A14B only). "
        "Values below 1.0 (including 0) fall back to the primary 'Guidance Scale'. "
        "Ignored for TI2V-5B.",
        title="Guidance Scale (Low Noise)",
    )
    # Wan transformer has ``patch_size=(1, 2, 2)``: combined with the VAE's
    # 8x spatial scale, generated H/W must be a multiple of 16 (not just 8)
    # or the patch round-trip lands off-by-one and the scheduler step fails
    # with a spatial-dim mismatch.
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
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

        pos_cond = self._load_conditioning(context, self.positive_conditioning, device=device, dtype=inference_dtype)
        do_cfg = self.guidance_scale != 1.0 and self.negative_conditioning is not None
        neg_cond: WanConditioningInfo | None = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_cond = self._load_conditioning(
                context, self.negative_conditioning, device=device, dtype=inference_dtype
            )

        # Reference-image conditioning (Wan 2.2 I2V-A14B only). The condition
        # tensor is 20 channels (4 mask + 16 VAE-encoded image latents); it
        # gets concatenated to the 16-channel noise latents each step,
        # yielding the 36-channel input the I2V transformer expects.
        ref_condition: torch.Tensor | None = None
        if self.ref_image is not None:
            if variant != WanVariantType.I2V_A14B:
                raise ValueError(
                    f"Reference-image conditioning is only supported by the Wan 2.2 I2V variant. "
                    f"The selected transformer is {variant.value!r}. Remove the Reference Image input "
                    "or load an I2V model."
                )
            if self.ref_image.width != self.width or self.ref_image.height != self.height:
                raise ValueError(
                    f"Reference-image dimensions ({self.ref_image.width}x{self.ref_image.height}) must "
                    f"match denoise dimensions ({self.width}x{self.height})."
                )
            if self.ref_image.num_frames > 1:
                # The image denoise produces single-frame output; concatenating a multi-frame
                # condition to a single-frame noise tensor mismatches the temporal dim and the
                # downstream tensor-shape error would be unhelpful.
                raise ValueError(
                    f"This denoise node produces a single-frame image but the reference image was "
                    f"encoded for {self.ref_image.num_frames} frames. Use the Denoise Video - Wan 2.2 "
                    "node for video I2V, or set num_frames=1 on the Reference Image node."
                )
            ref_condition = context.tensors.load(self.ref_image.condition_tensor_name).to(
                device=device, dtype=inference_dtype
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

        # Latents stay in fp32 throughout the denoise loop to avoid accumulating
        # bf16 quantization across the scheduler's small per-step deltas. We
        # cast to bf16 only when calling the transformer, matching Diffusers'
        # WanPipeline (which calls ``prepare_latents(..., dtype=torch.float32)``
        # then ``latent_model_input = latents.to(transformer_dtype)``).
        latent_dtype = torch.float32

        # Load init latents (img2img) and convert 4D → 5D.
        init_latents_5d: torch.Tensor | None = None
        if self.latents is not None:
            loaded = context.tensors.load(self.latents.latents_name).to(device=device, dtype=latent_dtype)
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
            dtype=latent_dtype,
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
        #
        # We deliberately do NOT call ``context.models.load(...)`` for the
        # transformer experts here — that would put both ~9 GB GGUF handles
        # in the model cache concurrently. With UMT5-XXL (~10 GB) competing
        # for the same cache, the LRU policy can drop one of them by the
        # time the denoise loop swaps in, producing the
        # "has already been dropped from the RAM cache" warning and forcing
        # a disk reload per swap. The swapper calls ``models.load`` lazily
        # inside each ``get()`` instead, so handles are always fresh.
        #
        # The config metadata (variant / format) is fine to read upfront —
        # ``get_config`` doesn't touch the weights cache.
        high_model = self.transformer.transformer
        low_model = self.transformer.transformer_low_noise
        low_config = context.models.get_config(low_model) if low_model is not None else None
        # FlowMatchEulerDiscreteScheduler stores num_train_timesteps in its config
        # (default 1000). Diffusers' WanPipeline computes:
        #   boundary_timestep = boundary_ratio * num_train_timesteps
        num_train_timesteps = int(scheduler.config.num_train_timesteps)
        boundary_timestep = self.transformer.boundary_ratio * num_train_timesteps if low_model is not None else None

        # LoRA wiring. The high-noise expert uses ``transformer.loras``; the
        # low-noise expert uses ``transformer.loras_low_noise``, falling back
        # to the primary list if empty (matches the WanTransformerField semantics).
        # Quantized (GGUF) experts force sidecar patching so GGMLTensor weights
        # aren't touched directly.
        high_loras = self.transformer.loras
        low_loras = self.transformer.loras_low_noise or self.transformer.loras
        high_config = context.models.get_config(high_model)
        high_is_quantized = high_config.format == ModelFormat.GGUFQuantized
        low_is_quantized = low_config.format == ModelFormat.GGUFQuantized if low_config is not None else False

        def high_lora_factory() -> Iterable[Tuple[ModelPatchRaw, float]]:
            return self._lora_iterator(context, high_loras)

        def low_lora_factory() -> Iterable[Tuple[ModelPatchRaw, float]]:
            return self._lora_iterator(context, low_loras)

        with ExitStack() as exit_stack:
            swapper = _ExpertSwapper(
                context=context,
                high_model=high_model,
                low_model=low_model,
                inference_dtype=inference_dtype,
                high_lora_factory=high_lora_factory if high_loras else None,
                low_lora_factory=low_lora_factory if low_loras else None,
                high_is_quantized=high_is_quantized,
                low_is_quantized=low_is_quantized,
            )
            exit_stack.callback(swapper.close)

            for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising (Wan 2.2)", total=total_steps)):
                timestep = t.expand(latents.shape[0])

                # Pick the active expert: high-noise for t >= boundary_timestep,
                # low-noise below. Single-transformer models always use HIGH.
                if low_model is not None and float(t) < float(boundary_timestep):
                    active_label = _ExpertSwapper.LOW
                    # Treat None or values below 1.0 (incl. the FE's default 0)
                    # as "use the primary guidance_scale".
                    low_cfg = self.guidance_scale_low_noise
                    active_cfg = low_cfg if (low_cfg is not None and low_cfg >= 1.0) else self.guidance_scale
                else:
                    active_label = _ExpertSwapper.HIGH
                    active_cfg = self.guidance_scale

                transformer = swapper.get(active_label)

                # Cast latents to the transformer's dtype only for the forward
                # pass; keep the scheduler-level latents in fp32.
                latent_model_input = latents.to(dtype=inference_dtype)

                # For I2V, concatenate the ref-image condition (4-ch mask + 16-ch
                # image latents) along the channel dim, producing the 36-channel
                # input the I2V transformer's patch_embedding expects.
                if ref_condition is not None:
                    latent_model_input = torch.cat([latent_model_input, ref_condition], dim=1)

                noise_pred_cond = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=pos_cond.prompt_embeds.unsqueeze(0),
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

                if do_cfg and neg_cond is not None:
                    noise_pred_uncond = transformer(
                        hidden_states=latent_model_input,
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
                    latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(latents_4d, sigma_prev)
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
        """Construct the scheduler matching the model's on-disk ``scheduler_config.json``.

        Wan model variants ship different schedulers — e.g. TI2V-5B uses
        ``UniPCMultistepScheduler`` with ``flow_shift=5.0``, while the
        standard A14B reference uses ``FlowMatchEulerDiscreteScheduler``.
        We dispatch on ``_class_name`` so the noise schedule matches what the
        model was trained against. Falls back to ``FlowMatchEulerDiscreteScheduler``
        defaults when no on-disk config is available.
        """
        import json

        import diffusers
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_dir = _scheduler_path_for_transformer(context, self.transformer)
        if scheduler_dir is None:
            return FlowMatchEulerDiscreteScheduler()

        # Read the on-disk class name and instantiate that class. Diffusers'
        # SchedulerMixin.from_pretrained does class dispatch internally, but
        # only when called from the abstract base; calling a concrete subclass
        # silently builds the wrong type. Resolve it explicitly.
        config_path = scheduler_dir / "scheduler_config.json"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            class_name = cfg.get("_class_name")
            scheduler_cls = getattr(diffusers, class_name, None) if class_name else None
        except (OSError, json.JSONDecodeError):
            scheduler_cls = None

        if scheduler_cls is None:
            scheduler_cls = FlowMatchEulerDiscreteScheduler

        return scheduler_cls.from_pretrained(str(scheduler_dir), local_files_only=True)

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

    def _lora_iterator(
        self, context: InvocationContext, loras: list[LoRAField]
    ) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Yield (ModelPatchRaw, weight) pairs for the given LoRA list.

        The caller passes either ``transformer.loras`` (high-noise expert) or
        ``transformer.loras_low_noise`` (low-noise expert) — the fallback to
        the primary list when low-noise is empty is handled at the call site.
        """
        for lora_field in loras:
            lora_info = context.models.load(lora_field.lora)
            assert isinstance(lora_info.model, ModelPatchRaw), (
                f"Wan LoRA model must be ModelPatchRaw, got {type(lora_info.model).__name__}"
            )
            yield (lora_info.model, lora_field.weight)
            del lora_info
