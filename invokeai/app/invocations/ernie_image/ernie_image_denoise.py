from contextlib import ExitStack
from typing import Optional

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    ErnieImageConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ernie_image import sampling_utils
from invokeai.backend.ernie_image.denoise import denoise as ernie_denoise
from invokeai.backend.flux.schedulers import (
    ERNIE_IMAGE_SCHEDULER_LABELS,
    ERNIE_IMAGE_SCHEDULER_MAP,
    ERNIE_IMAGE_SCHEDULER_NAME_VALUES,
    ERNIE_IMAGE_SHIFT,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ErnieImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import FP8_STORAGE_DTYPES

# What one sidecar-patched layer holds while it computes its residual -- a second copy of the layer's output,
# or for a LoKR/LoHA layer its full weight delta (at most ~0.2 GB here, `adaLN_modulation.1`). Same margin as
# LTX-2 uses.
_SIDECAR_TRANSIENT_BYTES = 1024**3


@invocation(
    "ernie_image_denoise",
    title="Denoise - ERNIE-Image",
    tags=["latents", "denoise", "ernie-image"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class ErnieImageDenoiseInvocation(BaseInvocation):
    """Run the ERNIE-Image denoising loop and emit packed latents."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: ErnieImageConditioningField = InputField(
        description="Positive prompt conditioning", input=Input.Connection
    )
    negative_conditioning: Optional[ErnieImageConditioningField] = InputField(
        default=None,
        description="Negative prompt conditioning (required when guidance_scale != 1.0)",
        input=Input.Connection,
    )
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=(
            "Optional starting latents. Must be patched ERNIE-Image latents: either clean ones "
            "(VAE-encoded, BN-normalized, patchified) with `add_noise` on, or the output of an "
            "earlier ERNIE-Image denoise stage with `add_noise` off."
        ),
        input=Input.Connection,
    )
    add_noise: bool = InputField(
        default=True,
        description=(
            "Noise the starting latents up to `denoising_start` before denoising. Turn this off to "
            "continue an earlier denoise stage, whose output already sits at that sigma. Ignored "
            "when no starting latents are connected."
        ),
    )

    width: int = InputField(default=1024, multiple_of=16, description="Generation width.")
    height: int = InputField(default=1024, multiple_of=16, description="Generation height.")
    steps: int = InputField(default=50, gt=0, description="Denoising steps. Use 8 for ERNIE-Image-Turbo.")
    guidance_scale: float = InputField(
        default=4.0,
        ge=1.0,
        description="Classifier-free guidance scale. 4.0 for ERNIE-Image, 1.0 (no CFG) for Turbo.",
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    seed: int = InputField(default=0, description="Random seed for noise generation.")
    scheduler: ERNIE_IMAGE_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler used during denoising.",
        ui_choice_labels=ERNIE_IMAGE_SCHEDULER_LABELS,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        pos_info = self._load_conditioning(context, self.positive_conditioning, dtype, device)
        neg_info: Optional[ErnieImageConditioningInfo] = None
        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            if self.negative_conditioning is None:
                raise ValueError("Negative conditioning is required when guidance_scale > 1.0")
            neg_info = self._load_conditioning(context, self.negative_conditioning, dtype, device)

        transformer_info = context.models.load(self.transformer.transformer)
        transformer_config = context.models.get_config(self.transformer.transformer)
        # Loaded before the lock, because the reservation below is sized from them.
        lora_patch_specs = self._materialize_lora_patches(context)
        # A GGUF build keeps its weights quantized, so the LoRA has to ride along as a sidecar. Asked of the
        # unlocked model, and only when there is something to patch.
        force_sidecar = bool(lora_patch_specs) and requires_sidecar_patching(
            transformer_info.model, transformer_config.format
        )
        # The patcher also picks the sidecar on its own for fp8 weights (FP8 Storage), which cannot be added to.
        lora_on_device = force_sidecar or (
            bool(lora_patch_specs) and any(p.dtype in FP8_STORAGE_DTYPES for p in transformer_info.model.parameters())
        )

        # A GGUF build dequantizes each Linear per forward, a transient its resident size does not cover. This node
        # has no activation estimate, so the cache floors the request at `device_working_mem_gb`, which then stands
        # in for the activations. Sidecar LoRA patches stay on the device for the whole run, so they come on top of
        # that floor rather than out of it. A plain build patched directly merges them into its weights instead.
        working_mem_bytes = peak_dequant_transient_bytes(transformer_info.model, dtype)
        if lora_on_device:
            floor = int(context.config.get().device_working_mem_gb * 2**30)
            lora_bytes = sum(spec[0].calc_size() for spec in lora_patch_specs)
            working_mem_bytes = max(working_mem_bytes, floor) + lora_bytes + _SIDECAR_TRANSIENT_BYTES

        with ExitStack() as exit_stack:
            (cached_weights, transformer) = exit_stack.enter_context(
                transformer_info.model_on_device(working_mem_bytes=working_mem_bytes)
            )
            if lora_patch_specs:
                exit_stack.enter_context(
                    LayerPatcher.apply_smart_model_patches(
                        model=transformer,
                        patches=lora_patch_specs,
                        prefix=ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
                        dtype=dtype,
                        cached_weights=cached_weights,
                        force_sidecar_patching=force_sidecar,
                    )
                )

            text_in_dim = int(transformer.config.text_in_dim)
            in_channels = int(transformer.config.in_channels)  # 128 -- already patched

            text_bth, text_lens = sampling_utils.pad_text(
                [pos_info.prompt_embeds], device=device, dtype=dtype, text_in_dim=text_in_dim
            )
            neg_text_bth = neg_text_lens = None
            if neg_info is not None:
                neg_text_bth, neg_text_lens = sampling_utils.pad_text(
                    [neg_info.prompt_embeds], device=device, dtype=dtype, text_in_dim=text_in_dim
                )

            latent_h = self.height // sampling_utils.VAE_SCALE_FACTOR
            latent_w = self.width // sampling_utils.VAE_SCALE_FACTOR

            sigmas = sampling_utils.get_schedule(
                self.steps, denoising_start=self.denoising_start, denoising_end=self.denoising_end
            )
            timesteps = sigmas.tolist()
            cfg_scale = [self.guidance_scale] * (len(timesteps) - 1)

            # Always generate noise on the same device and dtype, then cast, so a given seed
            # produces the same image regardless of the execution device (CUDA/ROCm/MPS/CPU).
            rand_device = "cpu"
            rand_dtype = torch.float32
            noise = torch.randn(
                (1, in_channels, latent_h, latent_w),
                generator=torch.Generator(device=rand_device).manual_seed(self.seed),
                device=rand_device,
                dtype=rand_dtype,
            ).to(device=device, dtype=dtype)

            init_latents = self._load_init_latents(context, noise, device=device, dtype=dtype)
            if init_latents is not None and not self.add_noise:
                # The latents already sit at the schedule's first sigma (they came out of an
                # earlier denoise stage), so start from them directly instead of re-noising.
                img, init_latents = init_latents, None
            else:
                img = noise

            scheduler = self._build_scheduler(context)

            def _step_callback(state: PipelineIntermediateState) -> None:
                context.util.sd_step_callback(state, BaseModelType.ErnieImage)

            img = ernie_denoise(
                model=transformer,
                img=img,
                text_bth=text_bth,
                text_lens=text_lens,
                timesteps=timesteps,
                step_callback=_step_callback,
                cfg_scale=cfg_scale,
                neg_text_bth=neg_text_bth,
                neg_text_lens=neg_text_lens,
                scheduler=scheduler,
                init_latents=init_latents,
                # Stochastic schedulers (LCM) re-noise the sample on every step. Without an
                # explicit generator they draw from the global RNG, so the `seed` field would only
                # control the initial latent and the same seed would not reproduce the image.
                # XOR mirrors `denoise_latents.py` and keeps the step noise decorrelated from the
                # initial noise, which is drawn from `self.seed` directly.
                generator=torch.Generator(device=rand_device).manual_seed(self.seed ^ 0xFFFFFFFF),
            )

        latents = img.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        # `LatentsOutput.build` assumes unpatched latents at 1/8 scale. ERNIE's latents are
        # 2x2-patchified on top of the 8x VAE downscale, so report the requested size directly.
        return LatentsOutput(
            latents=LatentsField(latents_name=name, seed=self.seed),
            width=self.width,
            height=self.height,
        )

    def _materialize_lora_patches(self, context: InvocationContext) -> list[PatchSpec]:
        """Load every LoRA on the transformer field into (patch, weight, cache-pin) specs."""
        patch_specs: list[PatchSpec] = []
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            patch_specs.append((lora_info.model, lora.weight, lora_info.model_in_ram()))
        return patch_specs

    def _load_init_latents(
        self,
        context: InvocationContext,
        noise: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Load and validate the optional image-to-image starting latents.

        The blend with `noise` deliberately happens in the denoise loop rather than here: only that
        layer knows the *post-shift* first sigma, since `get_schedule` emits raw schedule values and
        the scheduler applies its `shift` inside `set_timesteps`.
        """
        if self.latents is None:
            if self.denoising_start > 0:
                raise ValueError(
                    "denoising_start must be 0 when no initial latents are provided. There is nothing to "
                    "partially denoise, and starting from full-magnitude noise at a reduced sigma tells the "
                    "model the sample is already partly denoised, which produces garbage."
                )
            return None

        init_latents = context.tensors.load(self.latents.latents_name).to(device=device, dtype=dtype)
        if init_latents.shape != noise.shape:
            raise ValueError(
                f"Input latents have shape {tuple(init_latents.shape)} but this graph expects "
                f"{tuple(noise.shape)} (batch, patched channels, height, width). ERNIE-Image latents must be "
                "VAE-encoded, BN-normalized (`sampling_utils.vae_normalize`) and 2x2-patchified "
                "(`sampling_utils.patchify_latents`) before they can be denoised."
            )
        return init_latents

    def _build_scheduler(self, context: InvocationContext) -> SchedulerMixin:
        """Instantiate the selected scheduler from the pipeline's own `scheduler/` config.

        The diffusers defaults (`shift=1.0`, `num_train_timesteps=1000`) are not necessarily what
        the checkpoint ships, and both feed the sampling math (`set_timesteps` applies `shift` to
        the sigmas, and `num_train_timesteps` scales the timestep fed to the transformer). Falling
        back to a default-constructed scheduler would silently diverge from the reference pipeline.
        """
        scheduler_cls = ERNIE_IMAGE_SCHEDULER_MAP[self.scheduler]
        config = context.models.get_config(self.transformer.transformer)
        # Model paths in the record store are relative to `models_path` for Invoke-managed models.
        model_path = (context.config.get().models_path / config.path).resolve()
        scheduler_dir = model_path / "scheduler"
        if not scheduler_dir.is_dir():
            # A single file (safetensors or GGUF) has no `scheduler/` by construction -- `config.path`
            # is the file itself -- so only a Diffusers folder missing one is worth warning about. What
            # matters is the value: both released pipelines ship `shift=4.0`, and the driver hands
            # the scheduler raw sigmas expecting it to apply that shift, so default-constructing
            # would silently denoise on the unshifted schedule. Turbo feels it worst at 8 steps.
            if config.format is ModelFormat.Diffusers:
                context.logger.warning(
                    f"No scheduler config found at {scheduler_dir}; using {scheduler_cls.__name__} "
                    f"with the released ERNIE-Image shift={ERNIE_IMAGE_SHIFT}."
                )
            return scheduler_cls(shift=ERNIE_IMAGE_SHIFT)
        try:
            return scheduler_cls.from_pretrained(model_path, subfolder="scheduler")
        except Exception as e:
            context.logger.warning(
                f"Failed to load scheduler config from {scheduler_dir} ({e}); "
                f"falling back to {scheduler_cls.__name__} with the released ERNIE-Image "
                f"shift={ERNIE_IMAGE_SHIFT}."
            )
            return scheduler_cls(shift=ERNIE_IMAGE_SHIFT)

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: ErnieImageConditioningField,
        dtype: torch.dtype,
        device: torch.device,
    ) -> ErnieImageConditioningInfo:
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        if len(cond_data.conditionings) != 1:
            raise ValueError(f"Expected exactly one conditioning, got {len(cond_data.conditionings)}")
        info = cond_data.conditionings[0]
        if not isinstance(info, ErnieImageConditioningInfo):
            raise TypeError(
                f"Expected ErnieImageConditioningInfo, got {type(info).__name__}. "
                "Connect an ERNIE-Image text encoder to this input."
            )
        return info.to(device=device, dtype=dtype)
