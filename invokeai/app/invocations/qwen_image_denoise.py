from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
import torchvision.transforms as tv_transforms
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    QwenImageConditioningField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import QwenImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "qwen_image_denoise",
    title="Denoise - Qwen Image",
    tags=["image", "qwen_image"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImageDenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run the denoising process with a Qwen Image model."""

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    # Reference image latents (encoded through VAE) to concatenate with noisy latents.
    reference_latents: Optional[LatentsField] = InputField(
        default=None,
        description="Reference image latents to guide generation. Encoded through the VAE.",
        input=Input.Connection,
    )
    # denoise_mask is used for image-to-image inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.qwen_image_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: QwenImageConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[QwenImageConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    cfg_scale: float | list[float] = InputField(default=4.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    steps: int = InputField(default=40, gt=0, description=FieldDescriptions.steps)
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    shift: Optional[float] = InputField(
        default=None,
        description="Override the sigma schedule shift. "
        "When set, uses a fixed shift (e.g. 3.0 for Lightning LoRAs) instead of the default dynamic shifting. "
        "Leave unset for the base model's default schedule.",
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
        assert isinstance(conditioning, QwenImageConditioningInfo)
        conditioning = conditioning.to(dtype=dtype, device=device)
        return conditioning.prompt_embeds, conditioning.prompt_embeds_mask

    def _get_noise(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> torch.Tensor:
        rand_device = "cpu"
        rand_dtype = torch.float16

        return torch.randn(
            batch_size,
            num_channels_latents,
            int(height) // LATENT_SCALE_FACTOR,
            int(width) // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to(device=device, dtype=dtype)

    def _prepare_cfg_scale(self, num_timesteps: int) -> list[float]:
        if isinstance(self.cfg_scale, float):
            cfg_scale = [self.cfg_scale] * num_timesteps
        elif isinstance(self.cfg_scale, list):
            assert len(self.cfg_scale) == num_timesteps
            cfg_scale = self.cfg_scale
        else:
            raise ValueError(f"Invalid CFG scale type: {type(self.cfg_scale)}")
        return cfg_scale

    def _compute_sigmas(self, image_seq_len: int, num_steps: int, shift_override: float | None = None) -> list[float]:
        """Compute sigmas matching the diffusers FlowMatchEulerDiscreteScheduler.

        When shift_override is None, reproduces the full base-model pipeline:
        linspace → dynamic exponential time_shift → stretch_shift_to_terminal → append 0.

        When shift_override is set (e.g. 3.0 for Lightning LoRAs), uses a fixed mu = log(shift)
        with no shift_terminal stretching.
        """
        import math

        import numpy as np

        # 1. Initial sigmas: N values from 1.0 to 1/N (same as diffusers pipeline)
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float64)

        if shift_override is not None:
            # Fixed shift (e.g. Lightning LoRA): mu = log(shift), no terminal stretching
            mu = math.log(shift_override)
        else:
            # Dynamic shift from scheduler config
            base_shift = 0.5
            max_shift = 0.9
            base_image_seq_len = 256
            max_image_seq_len = 8192

            m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
            b = base_shift - m * base_image_seq_len
            mu = image_seq_len * m + b

        # 2. Exponential time shift
        sigmas = np.array([math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0)) for s in sigmas])

        # 3. Stretch shift to terminal (only for base model schedule)
        if shift_override is None:
            shift_terminal = 0.02
            one_minus = 1.0 - sigmas
            scale_factor = one_minus[-1] / (1.0 - shift_terminal)
            sigmas = 1.0 - (one_minus / scale_factor)

        # 4. Append terminal 0
        sigmas = np.append(sigmas, 0.0)

        return sigmas.tolist()

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int
    ) -> torch.Tensor:
        """Pack 4D latents (B, C, H, W) into 2x2-patched 3D (B, H/2*W/2, C*4)."""
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unpack 3D patched latents (B, seq, C*4) back to 4D (B, C, H, W)."""
        batch_size, _num_patches, channels = latents.shape
        # height/width are in latent space; they must be divisible by 2 for packing
        h = 2 * (height // 2)
        w = 2 * (width // 2)
        latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, h, w)
        return latents

    def _run_diffusion(self, context: InvocationContext):
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        transformer_info = context.models.load(self.transformer.transformer)
        assert isinstance(transformer_info.model, QwenImageTransformer2DModel)

        # Load conditioning
        pos_prompt_embeds, pos_prompt_mask = self._load_text_conditioning(
            context=context,
            conditioning_name=self.positive_conditioning.conditioning_name,
            dtype=inference_dtype,
            device=device,
        )

        neg_prompt_embeds = None
        neg_prompt_mask = None
        # Match the diffusers pipeline: only enable CFG when cfg_scale > 1 AND negative conditioning is provided.
        # With cfg_scale <= 1, the negative prediction is unused, so skip it entirely.
        cfg_scale_value = self.cfg_scale if isinstance(self.cfg_scale, float) else self.cfg_scale[0]
        do_classifier_free_guidance = self.negative_conditioning is not None and cfg_scale_value > 1.0
        if do_classifier_free_guidance:
            neg_prompt_embeds, neg_prompt_mask = self._load_text_conditioning(
                context=context,
                conditioning_name=self.negative_conditioning.conditioning_name,
                dtype=inference_dtype,
                device=device,
            )

        # Prepare the timestep / sigma schedule
        patch_size = transformer_info.model.config.patch_size
        assert isinstance(patch_size, int)
        # Output channels is 16 (the actual latent channels)
        out_channels = transformer_info.model.config.out_channels
        assert isinstance(out_channels, int)

        latent_height = self.height // LATENT_SCALE_FACTOR
        latent_width = self.width // LATENT_SCALE_FACTOR
        image_seq_len = (latent_height * latent_width) // (patch_size**2)

        # Use the actual FlowMatchEulerDiscreteScheduler to compute sigmas/timesteps,
        # exactly matching the diffusers pipeline.
        import math

        import numpy as np
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

        # Try to load the scheduler config from the model's directory (Diffusers models
        # have a scheduler/ subdir). For GGUF models this path doesn't exist, so fall
        # back to instantiating the scheduler with the known Qwen Image defaults.
        model_path = context.models.get_absolute_path(context.models.get_config(self.transformer.transformer))
        scheduler_path = model_path / "scheduler"
        if scheduler_path.is_dir() and (scheduler_path / "scheduler_config.json").exists():
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                base_shift=0.5,
                max_shift=0.9,
                base_image_seq_len=256,
                max_image_seq_len=8192,
                shift_terminal=0.02,
                num_train_timesteps=1000,
                time_shift_type="exponential",
            )

        if self.shift is not None:
            # Lightning LoRA: fixed shift
            mu = math.log(self.shift)
        else:
            # Default dynamic shifting
            # Linear interpolation matching diffusers' calculate_shift
            base_shift = scheduler.config.get("base_shift", 0.5)
            max_shift = scheduler.config.get("max_shift", 0.9)
            base_seq = scheduler.config.get("base_image_seq_len", 256)
            max_seq = scheduler.config.get("max_image_seq_len", 4096)
            m = (max_shift - base_shift) / (max_seq - base_seq)
            b = base_shift - m * base_seq
            mu = image_seq_len * m + b

        init_sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps).tolist()
        scheduler.set_timesteps(sigmas=init_sigmas, mu=mu, device=device)

        # Clip the schedule based on denoising_start/denoising_end to support img2img strength.
        # The scheduler's sigmas go from high (noisy) to 0 (clean). We clip to the fractional range.
        sigmas_sched = scheduler.sigmas  # (N+1,) including terminal 0
        if self.denoising_start > 0 or self.denoising_end < 1:
            total_sigmas = len(sigmas_sched) - 1  # exclude terminal
            start_idx = int(round(self.denoising_start * total_sigmas))
            end_idx = int(round(self.denoising_end * total_sigmas))
            sigmas_sched = sigmas_sched[start_idx : end_idx + 1]  # +1 to include the next sigma for dt
            # Rebuild timesteps from clipped sigmas (exclude terminal 0)
            timesteps_sched = sigmas_sched[:-1] * scheduler.config.num_train_timesteps
        else:
            timesteps_sched = scheduler.timesteps

        total_steps = len(timesteps_sched)

        cfg_scale = self._prepare_cfg_scale(total_steps)

        # Load initial latents if provided (for img2img)
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)
            if init_latents.dim() == 5:
                init_latents = init_latents.squeeze(2)

        # Load reference image latents if provided
        ref_latents = None
        if self.reference_latents is not None:
            ref_latents = context.tensors.load(self.reference_latents.latents_name)
            ref_latents = ref_latents.to(device=device, dtype=inference_dtype)
            # The VAE encoder produces 5D latents (B, C, 1, H, W); squeeze the frame dim
            # so we have 4D (B, C, H, W) for packing.
            if ref_latents.dim() == 5:
                ref_latents = ref_latents.squeeze(2)

        # Generate noise (16 channels - the output latent channels)
        noise = self._get_noise(
            batch_size=1,
            num_channels_latents=out_channels,
            height=self.height,
            width=self.width,
            dtype=inference_dtype,
            device=device,
            seed=self.seed,
        )

        # Prepare input latent image
        if init_latents is not None:
            s_0 = sigmas_sched[0].item()
            latents = s_0 * noise + (1.0 - s_0) * init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        if total_steps <= 0:
            return latents

        # Pack latents into 2x2 patches: (B, C, H, W) -> (B, H/2*W/2, C*4)
        latents = self._pack_latents(latents, 1, out_channels, latent_height, latent_width)

        # Determine whether the model uses reference latent conditioning (zero_cond_t).
        # Edit models (zero_cond_t=True) expect [noisy_patches ; ref_patches] in the sequence.
        # Txt2img models (zero_cond_t=False) only take noisy patches.
        has_zero_cond_t = getattr(transformer_info.model, "zero_cond_t", False) or getattr(
            transformer_info.model.config, "zero_cond_t", False
        )
        use_ref_latents = has_zero_cond_t

        ref_latents_packed = None
        if use_ref_latents:
            if ref_latents is not None:
                _, ref_ch, rh, rw = ref_latents.shape
                if rh != latent_height or rw != latent_width:
                    ref_latents = torch.nn.functional.interpolate(
                        ref_latents, size=(latent_height, latent_width), mode="bilinear"
                    )
            else:
                # No reference image provided — use zeros so the model still gets the
                # expected sequence layout.
                ref_latents = torch.zeros(
                    1, out_channels, latent_height, latent_width, device=device, dtype=inference_dtype
                )
            ref_latents_packed = self._pack_latents(ref_latents, 1, out_channels, latent_height, latent_width)

        # img_shapes tells the transformer the spatial layout of patches.
        if use_ref_latents:
            img_shapes = [
                [
                    (1, latent_height // 2, latent_width // 2),
                    (1, latent_height // 2, latent_width // 2),
                ]
            ]
        else:
            img_shapes = [
                [
                    (1, latent_height // 2, latent_width // 2),
                ]
            ]

        # Prepare inpaint extension (operates in 4D space, so unpack/repack around it)
        inpaint_mask = self._prep_inpaint_mask(context, noise)  # noise has the right 4D shape
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents,
                inpaint_mask=inpaint_mask,
                noise=noise,
            )

        step_callback = self._build_step_callback(context)

        step_callback(
            PipelineIntermediateState(
                step=0,
                order=1,
                total_steps=total_steps,
                timestep=int(timesteps_sched[0].item()) if len(timesteps_sched) > 0 else 0,
                latents=self._unpack_latents(latents, latent_height, latent_width),
            ),
        )

        noisy_seq_len = latents.shape[1]

        # Determine if the model is quantized — GGUF models need sidecar patching for LoRAs
        transformer_config = context.models.get_config(self.transformer.transformer)
        model_is_quantized = transformer_config.format in (ModelFormat.GGUFQuantized,)

        with ExitStack() as exit_stack:
            (cached_weights, transformer) = exit_stack.enter_context(transformer_info.model_on_device())
            assert isinstance(transformer, QwenImageTransformer2DModel)

            # Apply LoRA patches to the transformer
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            for step_idx, t in enumerate(tqdm(timesteps_sched)):
                # The pipeline passes timestep / 1000 to the transformer
                timestep = t.expand(latents.shape[0]).to(inference_dtype)

                # For edit models: concatenate noisy and reference patches along the sequence dim
                # For txt2img models: just use noisy patches
                if ref_latents_packed is not None:
                    model_input = torch.cat([latents, ref_latents_packed], dim=1)
                else:
                    model_input = latents

                noise_pred_cond = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=pos_prompt_embeds,
                    encoder_hidden_states_mask=pos_prompt_mask,
                    timestep=timestep / 1000,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                # Only keep the noisy-latent portion of the output
                noise_pred_cond = noise_pred_cond[:, :noisy_seq_len]

                if do_classifier_free_guidance and neg_prompt_embeds is not None:
                    noise_pred_uncond = transformer(
                        hidden_states=model_input,
                        encoder_hidden_states=neg_prompt_embeds,
                        encoder_hidden_states_mask=neg_prompt_mask,
                        timestep=timestep / 1000,
                        img_shapes=img_shapes,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = noise_pred_uncond[:, :noisy_seq_len]

                    noise_pred = noise_pred_uncond + cfg_scale[step_idx] * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Euler step using the (possibly clipped) sigma schedule
                sigma_curr = sigmas_sched[step_idx]
                sigma_next = sigmas_sched[step_idx + 1]
                dt = sigma_next - sigma_curr
                latents = latents.to(torch.float32) + dt * noise_pred.to(torch.float32)
                latents = latents.to(inference_dtype)

                if inpaint_extension is not None:
                    sigma_next = sigmas_sched[step_idx + 1].item()
                    latents_4d = self._unpack_latents(latents, latent_height, latent_width)
                    latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(latents_4d, sigma_next)
                    latents = self._pack_latents(latents_4d, 1, out_channels, latent_height, latent_width)

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t.item()),
                        latents=self._unpack_latents(latents, latent_height, latent_width),
                    ),
                )

        # Unpack back to 4D then add frame dim for the video-style VAE: (B, C, 1, H, W)
        latents = self._unpack_latents(latents, latent_height, latent_width)
        latents = latents.unsqueeze(2)
        return latents

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.QwenImage)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the transformer."""
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
