import json
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import torch
import torchvision.transforms as tv_transforms
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
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.sampling_utils import (
    KREA2_DISTILLED_MU,
    build_sigmas,
    calculate_shift,
    pack_latents,
    prepare_position_ids,
    unpack_latents,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Krea2ConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# Krea-2 latent channels (Qwen-Image VAE z_dim). The packed transformer in_channels is 16 * patch_size**2 = 64.
KREA2_LATENT_CHANNELS = 16


@invocation(
    "krea2_denoise",
    title="Denoise - Krea-2",
    tags=["image", "krea2", "krea-2"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2DenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
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
    positive_conditioning: Krea2ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[Krea2ConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # CFG uses the standard formulation (uncond + cfg_scale*(cond-uncond)); cfg_scale <= 1 disables it.
    # Krea-2-Turbo is distilled and runs with CFG disabled (cfg_scale=1.0).
    cfg_scale: float | list[float] = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    steps: int = InputField(default=8, gt=0, description=FieldDescriptions.steps)
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    shift: Optional[float] = InputField(
        default=None,
        description="Override the resolution-aware timestep shift (mu). Leave unset to use the model default "
        "(mu=1.15 for the distilled Turbo checkpoint).",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

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
        conditioning_name: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cond_data = context.conditioning.load(conditioning_name)
        assert len(cond_data.conditionings) == 1
        conditioning = cond_data.conditionings[0]
        assert isinstance(conditioning, Krea2ConditioningInfo)
        conditioning = conditioning.to(dtype=dtype, device=device)
        return conditioning.prompt_embeds, conditioning.prompt_embeds_mask

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

    def _validate_inputs(self) -> None:
        if self.denoising_start >= self.denoising_end:
            raise ValueError("denoising_start must be less than denoising_end.")
        if self.denoise_mask is not None and self.latents is None:
            raise ValueError("Initial latents are required when a denoise mask is provided.")

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
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

        self._validate_inputs()

        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        transformer_info = context.models.load(self.transformer.transformer)

        pos_prompt_embeds, pos_prompt_mask = self._load_text_conditioning(
            context, self.positive_conditioning.conditioning_name, inference_dtype, device
        )

        # CFG: standard formulation, enabled only when cfg_scale > 1 and negative conditioning is provided.
        if isinstance(self.cfg_scale, list):
            any_cfg_above_one = any(v > 1.0 for v in self.cfg_scale)
        else:
            any_cfg_above_one = self.cfg_scale > 1.0
        do_cfg = self.negative_conditioning is not None and any_cfg_above_one
        neg_prompt_embeds = None
        neg_prompt_mask = None
        if do_cfg:
            neg_prompt_embeds, neg_prompt_mask = self._load_text_conditioning(
                context, self.negative_conditioning.conditioning_name, inference_dtype, device
            )

        latent_height = self.height // LATENT_SCALE_FACTOR
        latent_width = self.width // LATENT_SCALE_FACTOR
        grid_height = latent_height // 2
        grid_width = latent_width // 2
        image_seq_len = grid_height * grid_width

        # Scheduler: load from the model's scheduler/ dir if present, else construct with Krea-2 defaults.
        model_path = context.models.get_absolute_path(context.models.get_config(self.transformer.transformer))
        scheduler_path = Path(model_path) / "scheduler"
        if scheduler_path.is_dir() and (scheduler_path / "scheduler_config.json").exists():
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                base_shift=0.5,
                max_shift=1.15,
                base_image_seq_len=256,
                max_image_seq_len=6400,
                num_train_timesteps=1000,
                time_shift_type="exponential",
            )

        if self.shift is not None:
            mu = self.shift
        elif self._is_distilled(context):
            mu = KREA2_DISTILLED_MU
        else:
            mu = calculate_shift(image_seq_len)

        init_sigmas = build_sigmas(self.steps)
        scheduler.set_timesteps(sigmas=init_sigmas, mu=mu, device=device)

        # Clip the schedule based on denoising_start/denoising_end for img2img strength.
        sigmas_sched = scheduler.sigmas  # (N+1,) including terminal 0
        total_sigmas = len(sigmas_sched) - 1  # == self.steps
        is_clipped = self.denoising_start > 0 or self.denoising_end < 1
        start_idx = int(round(self.denoising_start * total_sigmas))
        end_idx = int(round(self.denoising_end * total_sigmas))
        if is_clipped:
            sigmas_sched = sigmas_sched[start_idx : end_idx + 1]
            timesteps_sched = sigmas_sched[:-1] * scheduler.config.num_train_timesteps
        else:
            timesteps_sched = scheduler.timesteps

        total_steps = len(timesteps_sched)

        # Build the CFG schedule against the FULL step count, then clip it to the active window. This way a
        # caller-supplied per-step CFG list (one value per configured step) survives the reduction caused
        # by denoising_start/denoising_end; a scalar is simply broadcast.
        full_cfg_scale = self._prepare_cfg_scale(total_sigmas)
        cfg_scale = full_cfg_scale[start_idx:end_idx] if is_clipped else full_cfg_scale

        # Load initial latents (img2img).
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)
            if init_latents.dim() == 5:
                init_latents = init_latents.squeeze(2)

        noise = self._get_noise(self.height, self.width, inference_dtype, device, self.seed)

        if init_latents is not None:
            s_0 = sigmas_sched[0].item()
            latents = s_0 * noise + (1.0 - s_0) * init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        if total_steps <= 0:
            return latents.unsqueeze(2)

        # Pack latents into 2x2 patches: (B, C, H, W) -> (B, grid_h*grid_w, C*4).
        latents = pack_latents(latents, 1, KREA2_LATENT_CHANNELS, latent_height, latent_width)

        # Position ids: text tokens at origin, image tokens carry their grid coords.
        text_seq_len = pos_prompt_embeds.shape[1]
        position_ids = prepare_position_ids(text_seq_len, grid_height, grid_width, device)

        # Inpaint extension operates in 4D, so unpack/repack around each merge.
        inpaint_mask = self._prep_inpaint_mask(context, noise)
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents, inpaint_mask=inpaint_mask, noise=noise
            )

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
        model_is_quantized = transformer_config.format in (ModelFormat.GGUFQuantized,)
        num_train_timesteps = scheduler.config.num_train_timesteps

        # Estimate the peak working memory (activations) the transformer forward needs and ask the model
        # cache to keep that much VRAM free. The cache offloads as much of the (resident) model to RAM as
        # required to honor this — only consequential at higher resolutions, where the activation footprint
        # over text+image tokens grows enough that a fully-resident ~12B model would otherwise leave no
        # headroom. Without this hint the cache reserves only the small default working memory and places
        # the model before LoRA patches are applied, so a model+LoRA combination that just fits the base
        # forward OOMs once the LoRA's extra activations are added.
        estimated_working_memory = self._estimate_working_memory(
            image_seq_len=image_seq_len,
            do_cfg=do_cfg,
            num_loras=len(self.transformer.loras),
        )

        with ExitStack() as exit_stack:
            (cached_weights, transformer) = exit_stack.enter_context(
                transformer_info.model_on_device(working_mem_bytes=estimated_working_memory)
            )

            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=KREA2_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            for step_idx, t in enumerate(tqdm(timesteps_sched)):
                # The pipeline passes timestep / num_train_timesteps to the transformer.
                timestep = (t / num_train_timesteps).expand(latents.shape[0]).to(inference_dtype)

                noise_pred_cond = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=pos_prompt_embeds,
                    encoder_attention_mask=pos_prompt_mask,
                    timestep=timestep,
                    position_ids=position_ids,
                    return_dict=False,
                )[0]

                if do_cfg and neg_prompt_embeds is not None:
                    noise_pred_uncond = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=neg_prompt_embeds,
                        encoder_attention_mask=neg_prompt_mask,
                        timestep=timestep,
                        position_ids=position_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + cfg_scale[step_idx] * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Euler step using the (possibly clipped) sigma schedule.
                sigma_curr = sigmas_sched[step_idx]
                sigma_next = sigmas_sched[step_idx + 1]
                dt = sigma_next - sigma_curr
                latents = latents.to(torch.float32) + dt * noise_pred.to(torch.float32)
                latents = latents.to(inference_dtype)

                if inpaint_extension is not None:
                    sigma_next_f = sigmas_sched[step_idx + 1].item()
                    latents_4d = unpack_latents(latents, latent_height, latent_width)
                    latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(
                        latents_4d, sigma_next_f
                    )
                    latents = pack_latents(latents_4d, 1, KREA2_LATENT_CHANNELS, latent_height, latent_width)

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t.item()),
                        latents=unpack_latents(latents, latent_height, latent_width),
                    ),
                )

        # Unpack to 4D then add a frame dim for the Qwen-Image VAE: (B, C, 1, H, W).
        latents = unpack_latents(latents, latent_height, latent_width)
        latents = latents.unsqueeze(2)
        return latents

    def _estimate_working_memory(self, image_seq_len: int, do_cfg: bool, num_loras: int) -> int:
        """Estimate peak transformer activation memory (bytes) so the model cache reserves enough headroom.

        The MMDiT activation footprint scales with the number of image tokens. The per-token figure is
        calibrated empirically against the Krea-2-Turbo transformer in bf16 (~2.6 MiB/token covers the
        attention + feed-forward intermediates and the transient fp8->bf16 weight casts). LoRA sidecar
        patches add their own (small) weights plus an extra activation branch per patched layer, so we add
        a fixed margin per LoRA on top.
        """
        GB = 1024**3
        per_token_bytes = int(2.6 * 1024 * 1024)
        estimated = image_seq_len * per_token_bytes
        if do_cfg:
            # Conditional/unconditional passes are sequential, but the larger combined sequence and extra
            # transient buffers warrant a modest bump.
            estimated = int(estimated * 1.1)
        if num_loras > 0:
            estimated += int((1.5 + 0.5 * num_loras) * GB)
        return estimated

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Krea2)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
