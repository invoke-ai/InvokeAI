import json
import math
import statistics
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torchvision.transforms as tv_transforms
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    Krea2ConditioningField,
    Krea2StyleReferenceField,
    LatentsField,
    MetadataField,
    WithMetadata,
)
from invokeai.app.invocations.metadata import (
    LatentsMetaOutput,
    build_denoise_metadata,
    serialize_lora_metadata,
)
from invokeai.app.invocations.model import LoRAField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.attention import (
    Krea2RegionalPromptingState,
    build_krea2_attention_processors,
    resolve_krea2_sdpa_backends,
)
from invokeai.backend.krea2.denoise_working_memory import estimate_krea2_denoise_working_memory
from invokeai.backend.krea2.reference_edit import (
    Krea2ReferenceKVCache,
    forward_with_references,
    precompute_reference_kv,
)
from invokeai.backend.krea2.regional_prompting import (
    Krea2RegionalPromptingExtension,
    Krea2TextConditioning,
)
from invokeai.backend.krea2.sampling_utils import (
    KREA2_DISTILLED_MU,
    KREA2_SCHEDULER_CONFIG,
    build_sigmas,
    pack_latents,
    prepare_position_ids,
    prepare_position_ids_with_references,
    prepare_reference_position_ids,
    unpack_latents,
)
from invokeai.backend.krea2.style_reference_extension import Krea2StyleReferenceExtension
from invokeai.backend.krea2.text_encoding import KREA2_MAX_REFERENCE_IMAGES
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.rectified_flow.schedule_shift import ShiftFit, load_scheduler
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Krea2ConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.sage_attention import sage_attention_scope

# Krea-2 latent channels (Qwen-Image VAE z_dim). The packed transformer in_channels is 16 * patch_size**2 = 64.
KREA2_LATENT_CHANNELS = 16


logger = InvokeAILogger.get_logger(__name__)


class _Krea2StepBenchmark:
    """Per-step timing for the Krea-2 denoise loop, for A/B-ing SDPA backends.

    Off unless KREA2_SDPA_BACKEND_ENV_VAR is set, and off on non-CUDA devices. That matters: the
    `cuda.synchronize()` calls on both sides of a step serialise the loop, so the default path must
    not reach them at all.
    """

    def __init__(self, device: torch.device, label: str) -> None:
        self._device = device
        self._label = label
        self._step_ms: list[float] = []
        self._step_started_at = 0.0
        # Reset here, immediately before the loop, so the reported peak is the loop's and not the
        # model load's.
        torch.cuda.reset_peak_memory_stats(device)

    @classmethod
    def create(cls, device: torch.device) -> "_Krea2StepBenchmark | None":
        override = resolve_krea2_sdpa_backends().override
        if override is None or device.type != "cuda":
            return None
        return cls(device, override)

    def start_step(self) -> None:
        torch.cuda.synchronize(self._device)
        self._step_started_at = time.perf_counter()

    def end_step(self) -> None:
        torch.cuda.synchronize(self._device)
        self._step_ms.append((time.perf_counter() - self._step_started_at) * 1000)

    def log_summary(self) -> None:
        if not self._step_ms:
            return
        peak_gib = torch.cuda.max_memory_allocated(self._device) / 2**30
        first_ms = self._step_ms[0]
        # The first step absorbs kernel selection and allocator warmup -- on a cold cuDNN run it has
        # been seen at 2606 ms against a 1451 ms steady mean -- so it is reported, not averaged in.
        steady = self._step_ms[1:] or self._step_ms
        logger.info(
            f"Krea-2 SDPA benchmark [{self._label}]: {len(self._step_ms)} steps | "
            f"first {first_ms:.0f} ms | "
            f"steady mean {statistics.mean(steady):.0f} ms, median {statistics.median(steady):.0f} ms, "
            f"min {min(steady):.0f} ms, max {max(steady):.0f} ms | "
            f"peak torch.cuda.max_memory_allocated {peak_gib:.3f} GiB"
        )


@invocation(
    "krea2_denoise",
    title="Denoise - Krea-2",
    tags=["image", "krea2", "krea-2"],
    category="image",
    version="1.4.0",
    classification=Classification.Prototype,
)
class Krea2DenoiseInvocation(BaseInvocation, WithMetadata):
    """Run the denoising process with a Krea-2 model."""

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    # denoise_mask is used for image-to-image inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.krea2_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: Krea2ConditioningField | list[Krea2ConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Krea2ConditioningField | list[Krea2ConditioningField] | None = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # CFG uses the standard formulation (uncond + cfg_scale*(cond-uncond)); cfg_scale <= 1 disables it.
    # Krea-2-Turbo is distilled and runs with CFG disabled (cfg_scale=1.0).
    cfg_scale: float | list[float] = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, gt=0, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, gt=0, multiple_of=16, description="Height of the generated image.")
    steps: int = InputField(default=8, gt=0, description=FieldDescriptions.steps)
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    shift: Optional[float] = InputField(
        default=None,
        description="Override the resolution-aware timestep shift (mu). Leave unset -- or 0, which is what "
        "the node editor fills in for an untouched optional float -- to use the model default: mu=1.15 for "
        "the distilled Turbo checkpoint, resolution-dependent for Raw up to the area of 1280x1280 and held above it.",
    )
    style_reference: Optional[Krea2StyleReferenceField] = InputField(
        default=None,
        description="Training-free style reference. Adds one reference forward per step, so generation "
        "takes roughly twice as long, and retains the reference's attention keys/values for the whole "
        "step (~0.5 GB at 1024x1024, ~1.7 GB at 2560x1440). At 1440p the combined footprint no longer "
        "fits a 24 GB card alongside the model. A style_strength of 0 is ignored entirely and costs "
        "nothing.",
        input=Input.Connection,
        title="Style Reference",
    )
    style_reference_conditioning: Optional[Krea2ConditioningField] = InputField(
        default=None,
        description="Prompt for the style-reference pass. Leave unconnected to reuse the positive prompt; "
        "a short neutral prompt describing the reference can give a purer style transfer.",
        input=Input.Connection,
        title="Style Reference Prompt",
    )
    reference_latents: LatentsField | list[LatentsField] | None = InputField(
        default=None,
        description="Edit references for Krea-2 edit LoRAs, encoded by 'Image to Latents - Qwen Image' with "
        f"'reference' enabled. Up to {KREA2_MAX_REFERENCE_IMAGES}. They join every transformer pass as clean "
        "(t=0) image tokens, about 4096 tokens per megapixel each; with Reference K/V Cache on they run once "
        "instead of in every pass. Not combinable with style reference or regional prompts.",
        input=Input.Connection,
        title="Reference Latents",
    )
    reference_kv_cache: bool = InputField(
        default=False,
        description="For edit LoRAs trained with ai-toolkit's 'kv_cache' option, where reference tokens attend only "
        "to each other. The references then run through the transformer once per generation instead of in every "
        "pass, which is faster. Leave off for normal edit LoRAs. The LoRA file does not say which kind it is; "
        "check its model card. No effect without reference latents.",
        title="Reference K/V Cache",
    )

    @field_validator("cfg_scale")
    @classmethod
    def validate_cfg_scale_is_finite(cls, value: float | list[float]) -> float | list[float]:
        values = value if isinstance(value, list) else [value]
        if not all(math.isfinite(item) for item in values):
            raise ValueError("cfg_scale values must be finite.")
        return value

    @field_validator("shift")
    @classmethod
    def validate_shift_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("shift must be finite.")
        return value

    def _build_metadata(self) -> MetadataField:
        updates: dict[str, Any] = {
            "generation_mode": (
                "krea2_inpaint"
                if getattr(self, "denoise_mask", None) is not None
                else ("krea2_img2img" if getattr(self, "latents", None) is not None else "krea2_txt2img")
            ),
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_start": self.denoising_start,
            "denoising_end": self.denoising_end,
            "seed": self.seed,
            "model": self.transformer.transformer,
        }
        if self.shift is not None:
            updates["shift"] = self.shift
        if getattr(self, "style_reference", None) is not None and self.style_reference.style_strength > 0:
            updates["krea2_style_strength"] = self.style_reference.style_strength

        loras = serialize_lora_metadata(getattr(self.transformer, "loras", None))
        if loras:
            updates["loras"] = loras
        if getattr(self, "reference_latents", None) is not None and self._reference_latent_fields() and loras:
            updates["krea2_reference_kv_cache"] = self.reference_kv_cache

        return build_denoise_metadata(getattr(self, "metadata", None), updates)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsMetaOutput.build(
            latents_name=name,
            latents=latents,
            seed=None,
            metadata=self._build_metadata(),
            width=self.width,
            height=self.height,
        )

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        if self.denoise_mask is None:
            return None
        mask = context.tensors.load(self.denoise_mask.mask_name)
        mask = 1.0 - mask
        _, _, latent_height, latent_width = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_height, latent_width],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        return mask

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        conditioning_field: Krea2ConditioningField | list[Krea2ConditioningField],
        grid_height: int,
        grid_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Krea2RegionalPromptingExtension:
        conditioning_fields = (
            [conditioning_field] if isinstance(conditioning_field, Krea2ConditioningField) else conditioning_field
        )
        if not conditioning_fields:
            raise ValueError("At least one Krea-2 conditioning is required.")

        text_conditionings: list[Krea2TextConditioning] = []
        for field in conditioning_fields:
            cond_data = context.conditioning.load(field.conditioning_name)
            assert len(cond_data.conditionings) == 1
            conditioning = cond_data.conditionings[0]
            assert isinstance(conditioning, Krea2ConditioningInfo)
            conditioning = conditioning.to(dtype=dtype, device=device)
            embeds = conditioning.prompt_embeds
            if conditioning.prompt_embeds_mask is not None:
                mask = conditioning.prompt_embeds_mask.to(device=device, dtype=torch.bool)
                if mask.shape != embeds.shape[:2]:
                    raise ValueError(
                        f"Krea-2 conditioning mask shape {tuple(mask.shape)} does not match "
                        f"prompt embedding shape {tuple(embeds.shape[:2])}."
                    )
                valid_token_counts = mask.sum(dim=1)
                if not torch.equal(valid_token_counts, valid_token_counts[:1].expand_as(valid_token_counts)):
                    raise ValueError("All Krea-2 conditioning batch items must have the same valid token count.")
                embeds = torch.stack(
                    [batch_embeds[batch_mask] for batch_embeds, batch_mask in zip(embeds, mask, strict=True)]
                )
            regional_mask = None
            if field.mask is not None:
                mask = context.tensors.load(field.mask.tensor_name)
                regional_mask = Krea2RegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask=mask,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    dtype=dtype,
                    device=device,
                )
            text_conditionings.append(Krea2TextConditioning(prompt_embeds=embeds, mask=regional_mask))

        # Masked padding does not contribute to attention. Remove it before concatenation to avoid multiplying
        # the text sequence length by the encoder's fixed 512-token allocation for every conditioning.
        return Krea2RegionalPromptingExtension.from_text_conditionings(
            text_conditionings=text_conditionings,
            image_seq_len=grid_height * grid_width,
        )

    def _get_noise(self, height: int, width: int, dtype: torch.dtype, device: torch.device, seed: int) -> torch.Tensor:
        rand_device = "cpu"
        return torch.randn(
            1,
            KREA2_LATENT_CHANNELS,
            int(height) // LATENT_SCALE_FACTOR,
            int(width) // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=torch.float32,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to(device=device, dtype=dtype)

    def _prepare_cfg_scale(self, num_timesteps: int) -> list[float]:
        if isinstance(self.cfg_scale, float):
            return [self.cfg_scale] * num_timesteps
        if isinstance(self.cfg_scale, list):
            if len(self.cfg_scale) != num_timesteps:
                raise ValueError(
                    f"cfg_scale list has {len(self.cfg_scale)} values but the model is configured for "
                    f"{num_timesteps} steps. Provide one CFG value per configured step (or a single float)."
                )
            return self.cfg_scale
        raise ValueError(f"Invalid CFG scale type: {type(self.cfg_scale)}")

    def _step_window(self, num_steps: int) -> tuple[int, int]:
        """The ``[start, end)`` indices of the configured steps that denoising_start/denoising_end select."""
        return int(round(self.denoising_start * num_steps)), int(round(self.denoising_end * num_steps))

    def _active_cfg_scale(self, num_steps: int) -> list[float]:
        """The CFG of each step this node runs.

        Built against the full step count, then clipped to the active window, so a caller-supplied per-step list
        (one value per configured step) survives the reduction caused by denoising_start/denoising_end.
        """
        start_idx, end_idx = self._step_window(num_steps)
        return self._prepare_cfg_scale(num_steps)[start_idx:end_idx]

    def _negative_prompt_runs(self, num_steps: int) -> bool:
        """Whether a step this node runs applies CFG, the only steps that read the negative prompt."""
        return bool(self.negative_conditioning) and any(value > 1.0 for value in self._active_cfg_scale(num_steps))

    @staticmethod
    def _should_apply_cfg_for_step(cfg_scale: float, *, has_negative_conditioning: bool) -> bool:
        return has_negative_conditioning and cfg_scale > 1.0

    @staticmethod
    def _validate_effective_schedule(*, start_idx: int, end_idx: int) -> None:
        if end_idx <= start_idx:
            raise ValueError(
                "The requested denoising range does not contain any effective denoising steps at the configured "
                "step count. Increase denoising_end, decrease denoising_start, or increase steps."
            )

    def _validate_inputs(self) -> None:
        if self.denoising_start >= self.denoising_end:
            raise ValueError("denoising_start must be less than denoising_end.")
        if self.denoise_mask is not None and self.latents is None:
            raise ValueError("Initial latents are required when a denoise mask is provided.")
        self._validate_reference_latent_combination()

    def _reference_latent_fields(self) -> list[LatentsField]:
        if self.reference_latents is None:
            return []
        return [self.reference_latents] if isinstance(self.reference_latents, LatentsField) else self.reference_latents

    def _validate_reference_latent_combination(self) -> None:
        reference_fields = self._reference_latent_fields()
        if not reference_fields:
            return
        if len(reference_fields) > KREA2_MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Krea-2 accepts at most {KREA2_MAX_REFERENCE_IMAGES} reference latents, got {len(reference_fields)}."
            )
        # Style reference splices its keys/values in assuming the image tokens are the tail of the sequence,
        # and a regional mask is sized for exactly [text | target]. Reference tokens break both.
        if self.style_reference is not None and self.style_reference.style_strength > 0:
            raise ValueError("Krea-2 reference latents cannot be combined with a style reference. Disconnect one.")
        conditioning_fields = [
            field
            for conditioning in (self.positive_conditioning, self.negative_conditioning)
            if conditioning is not None
            for field in (conditioning if isinstance(conditioning, list) else [conditioning])
        ]
        if any(field.mask is not None for field in conditioning_fields):
            raise ValueError(
                "Krea-2 reference latents cannot be combined with regional (masked) prompts. Remove the masks "
                "or disconnect the reference latents."
            )
        if type(self)._install_attention_processors is not Krea2DenoiseInvocation._install_attention_processors:
            # A subclass's processors are sized for [text | target]: joint references would reach them as extra
            # image tokens, and the K/V cache lives in this node's own processors, which a subclass replaces.
            raise ValueError(
                f"'{type(self).__name__}' installs its own Krea-2 attention processors, which cannot take reference "
                "latents. Use the stock 'Denoise - Krea-2' node for edit references."
            )

    def _validate_reference_counts(self, context: InvocationContext) -> None:
        """Each prompt that runs must be a single prompt that saw as many references as the denoise is given.

        The encoder numbers the references it saw ("Picture 1", ...) and the transformer numbers the reference
        latents by their position ids. An edit LoRA is trained with the two agreeing; a mismatch -- a reference
        missing on one side, or a prompt encoded without them -- would condition on an image the other side
        never sees. Only the count can be checked: the denoise gets latents, not the images they came from, so
        keeping the same images in the same order on both sides is the workflow's job.

        Several prompts are concatenated into one text sequence, so each would bring the references' vision tokens
        in again, a layout no edit LoRA is trained on.
        """
        reference_count = len(self._reference_latent_fields())
        checked = [("positive", self.positive_conditioning)]
        # An unused negative prompt need not match. `self.steps` is the configured step count `_run_diffusion`
        # builds its schedule against.
        if self._negative_prompt_runs(self.steps):
            checked.append(("negative", self.negative_conditioning))
        for label, conditioning in checked:
            if conditioning is None:
                continue
            fields = conditioning if isinstance(conditioning, list) else [conditioning]
            if reference_count and len(fields) > 1:
                raise ValueError(
                    f"The {label} Krea-2 prompt is a collection of {len(fields)} prompts. With reference latents, "
                    "connect a single 'Prompt - Krea-2' node instead: each prompt would add the reference images "
                    "to the text sequence again, which edit LoRAs are not trained on."
                )
            for field in fields:
                info = context.conditioning.load(field.conditioning_name).conditionings[0]
                encoded_count = getattr(info, "num_reference_images", 0)
                if encoded_count != reference_count:
                    raise ValueError(
                        f"The {label} Krea-2 prompt was encoded with {encoded_count} reference image(s), but the "
                        f"denoise node has {reference_count} reference latents. Connect the same reference "
                        "images, in the same order, to the prompt and (through 'Image to Latents - Qwen Image') to "
                        "the denoise node."
                    )

    def _load_reference_latents(
        self, context: InvocationContext, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor | None, list[tuple[int, int]]]:
        """Load and pack the reference latents: every reference's tokens in order, and each one's patch grid."""
        packed: list[torch.Tensor] = []
        grids: list[tuple[int, int]] = []
        for index, field in enumerate(self._reference_latent_fields(), start=1):
            latents = context.tensors.load(field.latents_name)
            if latents.dim() == 5 and latents.shape[2] == 1:
                # The Qwen-Image VAE emits a single-frame axis.
                latents = latents.squeeze(2)
            if latents.dim() != 4 or latents.shape[0] != 1 or latents.shape[1] != KREA2_LATENT_CHANNELS:
                raise ValueError(
                    f"Krea-2 reference latents {index} have shape {tuple(latents.shape)}; expected "
                    f"(1, {KREA2_LATENT_CHANNELS}, height, width) from the Qwen-Image VAE."
                )
            _, _, height, width = latents.shape
            if height == 0 or width == 0 or height % 2 or width % 2:
                raise ValueError(
                    f"Krea-2 reference latents {index} are {height}x{width}; the transformer packs 2x2 patches, so "
                    "both sides must be even. Encode the reference at a multiple of 16 pixels."
                )
            latents = latents.to(device=device, dtype=dtype)
            packed.append(pack_latents(latents, 1, KREA2_LATENT_CHANNELS, height, width))
            grids.append((height // 2, width // 2))
        if not packed:
            return None, []
        return torch.cat(packed, dim=1), grids

    def _is_distilled(self, context: InvocationContext) -> bool:
        """Whether the transformer is the distilled Turbo checkpoint (fixed mu) vs. Raw (dynamic mu).

        Prefer the classified variant (works for diffusers, single-file and GGUF alike); fall back to
        the pipeline-level ``is_distilled`` flag in model_index.json, then default to distilled.

        A failed config lookup is a real error and is allowed to propagate — silently defaulting to the
        Turbo shift would apply the wrong sampling schedule to a Raw model.
        """
        from invokeai.backend.model_manager.taxonomy import Krea2VariantType

        config = context.models.get_config(self.transformer.transformer)
        variant = getattr(config, "variant", None)
        if variant is not None:
            return variant != Krea2VariantType.Base
        # No classified variant (unexpected for Krea-2) — fall back to the pipeline-level flag. Only a
        # missing/malformed model_index.json is tolerated here; it defaults to the distilled behavior.
        try:
            model_index = context.models.get_absolute_path(config) / "model_index.json"
            if model_index.is_file():
                with open(model_index) as f:
                    return bool(json.load(f).get("is_distilled", False))
        except (OSError, ValueError):
            pass
        return True

    def _run_diffusion(self, context: InvocationContext):
        self._validate_inputs()

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        # Before the transformer loads, so a malformed reference fails before anything is spent on the model.
        self._validate_reference_counts(context)
        reference_tokens, reference_grids = self._load_reference_latents(context, inference_dtype, device)
        reference_seq_len = 0 if reference_tokens is None else reference_tokens.shape[1]
        # K/V-cache mode keeps the references out of the per-step sequence: they run once, alone, before the loop.
        cache_references = self.reference_kv_cache and reference_tokens is not None
        joint_references = None if cache_references else reference_tokens

        transformer_info = context.models.load(self.transformer.transformer)

        latent_height = self.height // LATENT_SCALE_FACTOR
        latent_width = self.width // LATENT_SCALE_FACTOR
        grid_height = latent_height // 2
        grid_width = latent_width // 2
        image_seq_len = grid_height * grid_width
        pos_extension = self._load_text_conditioning(
            context,
            self.positive_conditioning,
            grid_height,
            grid_width,
            inference_dtype,
            device,
        )
        pos_prompt_embeds = pos_extension.regional_text_conditioning.prompt_embeds

        model_path = context.models.get_absolute_path(context.models.get_config(self.transformer.transformer))
        scheduler = load_scheduler(Path(model_path), KREA2_SCHEDULER_CONFIG)

        # A non-positive shift means "unset", not "no time shift". The node editor builds every float field
        # with `schemaObject.default ?? 0`, so an untouched optional shift arrives here as 0.0 rather than
        # None -- which would otherwise silently replace the model's mu and leave Turbo visibly grainy.
        # mu <= 0 is not a useful Krea-2 schedule anyway, so there is no legitimate value being shadowed.
        if self.shift is not None and self.shift > 0:
            mu = self.shift
        elif self._is_distilled(context):
            mu = KREA2_DISTILLED_MU
        else:
            # Resolution-aware shift, held at the end of its fit (see `KREA2_SCHEDULER_CONFIG`), from the loaded
            # scheduler's config: a Raw checkpoint that ships its own scheduler_config.json samples with its fit.
            mu = ShiftFit.from_scheduler_config(scheduler.config).mu(image_seq_len)

        init_sigmas = build_sigmas(self.steps)
        scheduler.set_timesteps(sigmas=init_sigmas, mu=mu, device=device)

        # Clip the schedule based on denoising_start/denoising_end for img2img strength.
        sigmas_sched = scheduler.sigmas  # (N+1,) including terminal 0
        is_clipped = self.denoising_start > 0 or self.denoising_end < 1
        start_idx, end_idx = self._step_window(self.steps)
        self._validate_effective_schedule(start_idx=start_idx, end_idx=end_idx)
        if is_clipped:
            sigmas_sched = sigmas_sched[start_idx : end_idx + 1]
            timesteps_sched = sigmas_sched[:-1] * scheduler.config.num_train_timesteps
        else:
            timesteps_sched = scheduler.timesteps

        total_steps = len(timesteps_sched)

        cfg_scale = self._active_cfg_scale(self.steps)

        # Load negative conditioning only if at least one active step actually uses CFG. CFG is still
        # decided per step below so values at or below 1.0 use the conditional prediction directly.
        do_cfg = self._negative_prompt_runs(self.steps)
        neg_extension: Krea2RegionalPromptingExtension | None = None
        neg_prompt_embeds = None
        if do_cfg and self.negative_conditioning is not None:
            neg_extension = self._load_text_conditioning(
                context,
                self.negative_conditioning,
                grid_height,
                grid_width,
                inference_dtype,
                device,
            )
            neg_prompt_embeds = neg_extension.regional_text_conditioning.prompt_embeds

        # Load initial latents (img2img).
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=torch.float32)
            if init_latents.dim() == 5:
                init_latents = init_latents.squeeze(2)

        # Keep the sampler state in fp32; only the transformer input is cast to its inference dtype.
        # Round-tripping the latents through bf16 every step loses low-order bits that show up as grain.
        noise = self._get_noise(self.height, self.width, torch.float32, device, self.seed)

        if init_latents is not None:
            s_0 = sigmas_sched[0].item()
            latents = s_0 * noise + (1.0 - s_0) * init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        # Pack latents into 2x2 patches: (B, C, H, W) -> (B, grid_h*grid_w, C*4).
        latents = pack_latents(latents, 1, KREA2_LATENT_CHANNELS, latent_height, latent_width)

        # Position ids: text tokens at origin, image tokens carry their grid coords, reference i its (i, h, w).
        def build_position_ids(text_seq_len: int) -> torch.Tensor:
            if joint_references is None:
                return prepare_position_ids(text_seq_len, grid_height, grid_width, device)
            return prepare_position_ids_with_references(text_seq_len, grid_height, grid_width, reference_grids, device)

        position_ids = build_position_ids(pos_prompt_embeds.shape[1])
        # The negative prompt can tokenize to a different length than the positive prompt, so it needs its
        # own position ids. Reusing the positive ids would leave the rotary embedding (text + image tokens)
        # a different length than the uncond query sequence and crash in the transformer's apply_rotary_emb.
        neg_position_ids = build_position_ids(neg_prompt_embeds.shape[1]) if neg_prompt_embeds is not None else None

        # Inpaint extension operates in 4D, so unpack/repack around each merge.
        inpaint_mask = self._prep_inpaint_mask(context, noise)
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents, inpaint_mask=inpaint_mask, noise=noise
            )

        # Style reference: load + validate before the transformer is placed, so a size mismatch fails
        # before we have spent anything on the model.
        style_extension: Krea2StyleReferenceExtension | None = None
        style_ref_prompt_embeds = None
        style_ref_position_ids = None
        # A style strength of 0 is documented as disabling the reference, so treat it as if nothing were
        # connected: no latents load, no K/V cache, no capture pass, no working-memory reservation. Running
        # the machinery for a mix of 0 would roughly double generation time for no visible effect.
        if self.style_reference is not None and self.style_reference.style_strength > 0:
            style_extension = Krea2StyleReferenceExtension.from_field(
                context,
                self.style_reference,
                denoise_width=self.width,
                denoise_height=self.height,
                dtype=inference_dtype,
                device=device,
            )
            if self.style_reference_conditioning is not None:
                style_ref_extension = self._load_text_conditioning(
                    context,
                    self.style_reference_conditioning,
                    grid_height,
                    grid_width,
                    inference_dtype,
                    device,
                )
                style_ref_prompt_embeds = style_ref_extension.regional_text_conditioning.prompt_embeds
                style_ref_position_ids = prepare_position_ids(
                    style_ref_prompt_embeds.shape[1], grid_height, grid_width, device
                )
            else:
                # Reusing the positive conditioning costs nothing and needs no extra encode. A dedicated,
                # neutral reference prompt is exposed as an override because it measurably changes how much
                # of the reference's *subject* bleeds into the style.
                style_ref_prompt_embeds = pos_prompt_embeds
                style_ref_position_ids = position_ids

        step_callback = self._build_step_callback(context)
        step_callback(
            PipelineIntermediateState(
                step=0,
                order=1,
                total_steps=total_steps,
                timestep=int(timesteps_sched[0].item()) if total_steps > 0 else 0,
                latents=unpack_latents(latents, latent_height, latent_width),
            ),
        )

        transformer_config = context.models.get_config(self.transformer.transformer)
        num_train_timesteps = scheduler.config.num_train_timesteps

        # Loaded once: the estimate prices these patches and the patcher applies the same ones. Each load reports its
        # progress, and the un-entered wrappers keep the LoRA records shielded from eviction until the patcher pins them.
        loaded_loras = [(lora, self._load_lora(context, lora)) for lora in self.transformer.loras]

        # The transformer's lock reserves what the whole denoise needs beside the weights: the forward, the cached
        # reference or style K/V, regional masks and the LoRA patches applied under it.
        working_memory = estimate_krea2_denoise_working_memory(
            transformer_info.model,
            image_seq_len=image_seq_len,
            text_seq_len=max(
                pos_prompt_embeds.shape[1], neg_prompt_embeds.shape[1] if neg_prompt_embeds is not None else 0
            ),
            style_text_seq_len=style_ref_prompt_embeds.shape[1] if style_ref_prompt_embeds is not None else 0,
            reference_seq_len=reference_seq_len,
            cache_references=cache_references,
            cfg=do_cfg,
            loras=[(lora_info.model, lora_info.config.file_size) for _, lora_info in loaded_loras],
            regional_mask_bytes=self._regional_attention_mask_bytes(pos_extension, neg_extension, inference_dtype),
            style_kv_bytes=style_extension.kv_cache_bytes(inference_dtype) if style_extension is not None else 0,
            device=device,
            dtype=inference_dtype,
        )

        with ExitStack() as exit_stack:
            (cached_weights, transformer) = exit_stack.enter_context(
                transformer_info.model_on_device(working_memory=working_memory)
            )

            # Whether LoRA has to go on as a sidecar. Asked of the loaded module tree rather than the
            # format, which cannot see an int8_tensorwise build's Int8ConvrotLinear layers.
            model_is_quantized = requires_sidecar_patching(transformer, transformer_config.format)

            attention_state = self._install_attention_processors(transformer, exit_stack)
            if style_extension is not None:
                self._install_style_reference_processors(transformer, exit_stack, attention_state, style_extension)
            reference_kv = (
                self._install_reference_kv_processors(transformer, exit_stack, attention_state)
                if cache_references
                else None
            )

            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=(
                        (lora_info.model, lora.weight, lora_info.model_in_ram()) for lora, lora_info in loaded_loras
                    ),
                    prefix=KREA2_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            pos_attention_payload = self._build_attention_payload(pos_extension, inference_dtype)
            neg_attention_payload = (
                self._build_attention_payload(neg_extension, inference_dtype) if neg_extension is not None else None
            )

            if style_extension is not None:
                # Built after the LoRA patcher so the reference trajectory is anchored to the same weights
                # the sampler will use.
                style_extension.prepare([sigma.item() for sigma in sigmas_sched[:total_steps]])

            if reference_kv is not None:
                assert reference_tokens is not None
                # After the LoRA patcher, whose weights shape the reference K/V. Once per generation: the cache
                # depends on neither the prompt nor the timestep, so it serves every step and both CFG passes.
                context.util.signal_progress("Precomputing reference K/V")
                precompute_reference_kv(
                    transformer,
                    reference_tokens,
                    prepare_reference_position_ids(reference_grids, device),
                    reference_kv,
                )

            # An INVOKE_KREA2_SDPA_BACKEND override measures PyTorch's kernels -- one pinned kernel, or the
            # process-wide order for `default` -- so SageAttention must not serve those calls instead.
            if resolve_krea2_sdpa_backends().override is None:
                exit_stack.enter_context(sage_attention_scope())

            benchmark = _Krea2StepBenchmark.create(device)

            def predict(
                model_latents: torch.Tensor,
                prompt_embeds: torch.Tensor,
                timestep: torch.Tensor,
                pass_position_ids: torch.Tensor,
            ) -> torch.Tensor:
                if joint_references is None:
                    # Cached references enter through the attention processors.
                    return transformer(
                        hidden_states=model_latents,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=None,
                        timestep=timestep,
                        position_ids=pass_position_ids,
                        return_dict=False,
                    )[0]
                # Both CFG passes carry the references (as in Qwen-Image-Edit), so guidance contrasts the
                # prompts and not the presence of the edit reference.
                return forward_with_references(
                    transformer, model_latents, prompt_embeds, timestep, pass_position_ids, joint_references
                )

            for step_idx, t in enumerate(tqdm(timesteps_sched)):
                if benchmark is not None:
                    benchmark.start_step()
                # The pipeline passes timestep / num_train_timesteps to the transformer.
                timestep = (t / num_train_timesteps).expand(latents.shape[0]).to(inference_dtype)
                # The sampler state stays fp32; the transformer gets its own cast copy.
                model_latents = latents.to(inference_dtype)

                style_pass = nullcontext()
                if style_extension is not None:
                    # Pass A: run the reference alone and stash its image-token K/V. It never sees the
                    # target, so this is equivalent to upstream's doubled batch — see style_reference.py.
                    self._install_attention_payload(attention_state, None)
                    with style_extension.capture():
                        transformer(
                            hidden_states=style_extension.reference_latents_for_step(step_idx),
                            encoder_hidden_states=style_ref_prompt_embeds,
                            encoder_attention_mask=None,
                            timestep=timestep,
                            position_ids=style_ref_position_ids,
                            return_dict=False,
                        )
                    style_pass = style_extension.inject(
                        Krea2StyleReferenceExtension.progress_for_step(step_idx, total_steps)
                    )

                # Pass B: the target. CFG runs inside the same injection — styling only the conditional
                # pass would make CFG amplify (styled_cond - plain_uncond), which overshoots badly above
                # cfg ~4. The single reference pass above is shared by both.
                with style_pass:
                    self._install_attention_payload(attention_state, pos_attention_payload)
                    noise_pred_cond = predict(model_latents, pos_prompt_embeds, timestep, position_ids)

                    if self._should_apply_cfg_for_step(
                        cfg_scale[step_idx], has_negative_conditioning=neg_prompt_embeds is not None
                    ):
                        self._install_attention_payload(attention_state, neg_attention_payload)
                        noise_pred_uncond = predict(model_latents, neg_prompt_embeds, timestep, neg_position_ids)
                        noise_pred = noise_pred_uncond + cfg_scale[step_idx] * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                # Euler step using the (possibly clipped) sigma schedule.
                sigma_curr = sigmas_sched[step_idx]
                sigma_next = sigmas_sched[step_idx + 1]
                dt = (sigma_next - sigma_curr).to(torch.float32)
                latents = latents + dt * noise_pred.to(torch.float32)

                if inpaint_extension is not None:
                    sigma_next_f = sigmas_sched[step_idx + 1].item()
                    latents_4d = unpack_latents(latents, latent_height, latent_width)
                    latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(
                        latents_4d, sigma_next_f
                    )
                    latents = pack_latents(latents_4d, 1, KREA2_LATENT_CHANNELS, latent_height, latent_width)

                if benchmark is not None:
                    benchmark.end_step()

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t.item()),
                        latents=unpack_latents(latents, latent_height, latent_width),
                    ),
                )

        if benchmark is not None:
            benchmark.log_summary()

        # Unpack to 4D then add a frame dim for the Qwen-Image VAE: (B, C, 1, H, W).
        latents = unpack_latents(latents, latent_height, latent_width)
        latents = latents.unsqueeze(2)
        return latents

    def _install_attention_processors(
        self, transformer: torch.nn.Module, exit_stack: ExitStack
    ) -> Krea2RegionalPromptingState:
        """Swap in the memory-efficient attention processors and return the state they share.

        Krea-2 uses grouped-query attention (48 query / 12 KV heads). The stock attention processor asks
        SDPA for enable_gqa=True, which PyTorch only supports on the math backend — that materializes the
        full O(seq^2) score matrix (~5.7 GB per attention at 1280x720, ~40 GB at 2560x1440) and OOMs. The
        replacement expands the KV heads and uses the O(seq) SDPA kernel instead.

        Extension point: subclasses may install their own processors and a richer state object here, as
        long as the state still carries the attention mask this node installs per pass.

        The signature is part of that contract — style reference deliberately does *not* extend it, and
        installs itself in ``_install_style_reference_processors`` instead.
        """
        state = Krea2RegionalPromptingState()
        transformer.set_attn_processor(build_krea2_attention_processors(transformer, state))
        # The processors remain installed on the cached transformer after this invocation. Do not let them
        # retain a potentially multi-GB regional mask between generations, including when denoising raises.
        exit_stack.callback(self._clear_attention_state, state)
        return state

    def _install_style_reference_processors(
        self,
        transformer: torch.nn.Module,
        exit_stack: ExitStack,
        attention_state: Krea2RegionalPromptingState,
        style_extension: Krea2StyleReferenceExtension,
    ) -> None:
        """Re-install the processors so the styled blocks also carry the style-reference state.

        Kept out of ``_install_attention_processors`` on purpose. Subclasses (the prompt-weighting node
        pack, for one) override that seam with their own state type and their own processors; widening
        its signature would break every generation they run, style reference or not.

        Because this replaces whatever that seam installed, it is only valid when the seam has *not* been
        overridden — hence the explicit refusal below rather than silently discarding the subclass's
        processors.
        """
        if type(self)._install_attention_processors is not Krea2DenoiseInvocation._install_attention_processors:
            raise ValueError(
                f"'{type(self).__name__}' installs its own Krea-2 attention processors, which style reference "
                "would replace. Use the stock 'Denoise - Krea-2' node for style reference."
            )

        style_state = style_extension.build_state(transformer)
        transformer.set_attn_processor(
            build_krea2_attention_processors(
                transformer,
                attention_state,
                style_reference_state=style_state,
                style_reference_blocks=style_extension.block_indices,
            )
        )
        # Same hazard as the regional mask, an order of magnitude larger: the captured reference K/V run
        # to ~1.7 GiB at 2560x1440 and would otherwise stay on the cached transformer.
        exit_stack.callback(style_state.clear)

    def _install_reference_kv_processors(
        self,
        transformer: torch.nn.Module,
        exit_stack: ExitStack,
        attention_state: Krea2RegionalPromptingState,
    ) -> Krea2ReferenceKVCache:
        """Re-install the processors so every main block also carries the edit-reference K/V cache.

        Like style reference, kept out of ``_install_attention_processors`` so that seam's signature stays as node
        packs know it; subclasses overriding it are refused in validation.
        """
        cache = Krea2ReferenceKVCache()
        transformer.set_attn_processor(
            build_krea2_attention_processors(transformer, attention_state, reference_kv_cache=cache)
        )
        # ~0.66 GiB per 1 MP reference in bf16, retained on the cached transformer's processors unless dropped here.
        exit_stack.callback(cache.clear)
        return cache

    @staticmethod
    def _clear_attention_state(state: Krea2RegionalPromptingState) -> None:
        """Drop every tensor the shared attention state retains. Override alongside the state object."""
        state.set_attention_mask(None)

    def _build_attention_payload(self, extension: Krea2RegionalPromptingExtension, inference_dtype: torch.dtype) -> Any:
        """Build the per-conditioning attention payload once, before the denoise loop.

        Returns the regional attention mask (or None). Subclasses may return a richer payload; whatever
        comes back is handed straight to ``_install_attention_payload`` before the matching pass.
        """
        return extension.get_attention_mask()

    @staticmethod
    def _install_attention_payload(state: Krea2RegionalPromptingState, payload: Any) -> None:
        """Install one conditioning's payload for the upcoming transformer call.

        Called separately for the conditional and unconditional pass, so nothing leaks between them.
        """
        state.set_attention_mask(payload)

    @staticmethod
    def _regional_attention_mask_bytes(
        pos_extension: Krea2RegionalPromptingExtension,
        neg_extension: Krea2RegionalPromptingExtension | None,
        attention_dtype: torch.dtype,
    ) -> int:
        """Return final masks plus peak construction scratch and SDPA's transient additive bias."""
        extensions = [pos_extension] if neg_extension is None else [pos_extension, neg_extension]
        final_mask_bytes = sum(extension.attention_mask_numel for extension in extensions)
        peak_build_scratch_bytes = max(extension.attention_mask_build_scratch_numel for extension in extensions)
        attention_dtype_bytes = torch.empty((), dtype=attention_dtype).element_size()
        peak_attention_bias_bytes = max(extension.attention_mask_numel for extension in extensions)
        peak_attention_bias_bytes *= attention_dtype_bytes
        return final_mask_bytes + peak_build_scratch_bytes + peak_attention_bias_bytes

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Krea2)

        return step_callback

    @staticmethod
    def _load_lora(context: InvocationContext, lora: LoRAField) -> LoadedModel:
        lora_info = context.models.load(lora.lora)
        if not isinstance(lora_info.model, ModelPatchRaw):
            raise TypeError(f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}.")
        return lora_info
