"""Flux2 Klein Denoise Invocation.

Run denoising process with a FLUX.2 Klein transformer model.
Uses Qwen3 conditioning instead of CLIP+T5.
"""

from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    FluxKontextConditioningField,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux2.denoise import denoise
from invokeai.backend.flux2.ref_image_extension import Flux2RefImageExtension
from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    generate_img_ids_flux2,
    get_noise_flux2,
    get_schedule_flux2,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_denoise",
    title="FLUX2 Denoise",
    tags=["image", "flux", "flux2", "klein", "denoise"],
    category="image",
    version="1.4.0",
    classification=Classification.Prototype,
)
class Flux2DenoiseInvocation(BaseInvocation):
    """Run denoising process with a FLUX.2 Klein transformer model.

    This node is designed for FLUX.2 Klein models which use Qwen3 as the text encoder.
    It does not support ControlNet, IP-Adapters, or regional prompting.
    """

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
    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(
        default=1.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_end,
    )
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    positive_text_conditioning: FluxConditioningField = InputField(
        description=FieldDescriptions.positive_cond,
        input=Input.Connection,
    )
    negative_text_conditioning: Optional[FluxConditioningField] = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    cfg_scale: float = InputField(
        default=1.0,
        description=FieldDescriptions.cfg_scale,
        title="CFG Scale",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(
        default=4,
        description="Number of diffusion steps. Use 4 for distilled models, 28+ for base models.",
    )
    scheduler: FLUX_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process. 'euler' is fast and standard. "
        "'heun' is 2nd-order (better quality, 2x slower). 'lcm' is optimized for few steps.",
        ui_choice_labels=FLUX_SCHEDULER_LABELS,
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    vae: VAEField = InputField(
        description="FLUX.2 VAE model (required for BN statistics).",
        input=Input.Connection,
    )
    kontext_conditioning: FluxKontextConditioningField | list[FluxKontextConditioningField] | None = InputField(
        default=None,
        description="FLUX Kontext conditioning (reference images for multi-reference image editing).",
        input=Input.Connection,
        title="Reference Images",
    )

    def _get_bn_stats(self, context: InvocationContext) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract BN statistics from the FLUX.2 VAE.

        The FLUX.2 VAE uses batch normalization on the patchified 128-channel representation.
        IMPORTANT: BFL FLUX.2 VAE uses affine=False, so there are NO learnable weight/bias.

        BN formula (affine=False): y = (x - mean) / std
        Inverse: x = y * std + mean

        Returns:
            Tuple of (bn_mean, bn_std) tensors of shape (128,), or None if BN layer not found.
        """
        with context.models.load(self.vae.vae).model_on_device() as (_, vae):
            # Ensure VAE is in eval mode to prevent BN stats from being updated
            vae.eval()

            # Try to find the BN layer - it may be at different locations depending on model format
            bn_layer = None
            if hasattr(vae, "bn"):
                bn_layer = vae.bn
            elif hasattr(vae, "batch_norm"):
                bn_layer = vae.batch_norm
            elif hasattr(vae, "encoder") and hasattr(vae.encoder, "bn"):
                bn_layer = vae.encoder.bn

            if bn_layer is None:
                return None

            # Verify running statistics are initialized
            if bn_layer.running_mean is None or bn_layer.running_var is None:
                return None

            # Get BN running statistics from VAE
            bn_mean = bn_layer.running_mean.clone()  # Shape: (128,)
            bn_var = bn_layer.running_var.clone()  # Shape: (128,)
            bn_eps = bn_layer.eps if hasattr(bn_layer, "eps") else 1e-4  # BFL uses 1e-4
            bn_std = torch.sqrt(bn_var + bn_eps)

        return bn_mean, bn_std

    def _bn_normalize(
        self,
        x: torch.Tensor,
        bn_mean: torch.Tensor,
        bn_std: torch.Tensor,
    ) -> torch.Tensor:
        """Apply BN normalization to packed latents.

        BN formula (affine=False): y = (x - mean) / std

        Args:
            x: Packed latents of shape (B, seq, 128).
            bn_mean: BN running mean of shape (128,).
            bn_std: BN running std of shape (128,).

        Returns:
            Normalized latents of same shape.
        """
        # x: (B, seq, 128), params: (128,) -> broadcast over batch and sequence dims
        bn_mean = bn_mean.to(x.device, x.dtype)
        bn_std = bn_std.to(x.device, x.dtype)
        return (x - bn_mean) / bn_std

    def _bn_denormalize(
        self,
        x: torch.Tensor,
        bn_mean: torch.Tensor,
        bn_std: torch.Tensor,
    ) -> torch.Tensor:
        """Apply BN denormalization to packed latents (inverse of normalization).

        Inverse BN (affine=False): x = y * std + mean

        Args:
            x: Packed latents of shape (B, seq, 128).
            bn_mean: BN running mean of shape (128,).
            bn_std: BN running std of shape (128,).

        Returns:
            Denormalized latents of same shape.
        """
        # x: (B, seq, 128), params: (128,) -> broadcast over batch and sequence dims
        bn_mean = bn_mean.to(x.device, x.dtype)
        bn_std = bn_std.to(x.device, x.dtype)
        return x * bn_std + bn_mean

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        # Get BN statistics from VAE for latent denormalization (optional)
        # BFL FLUX.2 VAE uses affine=False, so only mean/std are needed
        # Some VAE formats (e.g. diffusers) may not expose BN stats directly
        bn_stats = self._get_bn_stats(context)
        bn_mean, bn_std = bn_stats if bn_stats is not None else (None, None)

        # Load the input latents, if provided
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        # Prepare input noise (FLUX.2 uses 32 channels)
        noise = get_noise_flux2(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )
        b, _c, latent_h, latent_w = noise.shape
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        # Load the conditioning data
        pos_cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(pos_cond_data.conditionings) == 1
        pos_flux_conditioning = pos_cond_data.conditionings[0]
        assert isinstance(pos_flux_conditioning, FLUXConditioningInfo)
        pos_flux_conditioning = pos_flux_conditioning.to(dtype=inference_dtype, device=device)

        # Qwen3 stacked embeddings (stored in t5_embeds field for compatibility)
        txt = pos_flux_conditioning.t5_embeds

        # Generate text position IDs (4D format for FLUX.2: T, H, W, L)
        # FLUX.2 uses 4D position coordinates for its rotary position embeddings
        # IMPORTANT: Position IDs must be int64 (long) dtype
        # Diffusers uses: T=0, H=0, W=0, L=0..seq_len-1
        seq_len = txt.shape[1]
        txt_ids = torch.zeros(1, seq_len, 4, device=device, dtype=torch.long)
        txt_ids[..., 3] = torch.arange(seq_len, device=device, dtype=torch.long)  # L coordinate varies

        # Load negative conditioning if provided
        neg_txt = None
        neg_txt_ids = None
        if self.negative_text_conditioning is not None:
            neg_cond_data = context.conditioning.load(self.negative_text_conditioning.conditioning_name)
            assert len(neg_cond_data.conditionings) == 1
            neg_flux_conditioning = neg_cond_data.conditionings[0]
            assert isinstance(neg_flux_conditioning, FLUXConditioningInfo)
            neg_flux_conditioning = neg_flux_conditioning.to(dtype=inference_dtype, device=device)
            neg_txt = neg_flux_conditioning.t5_embeds
            # For text tokens: T=0, H=0, W=0, L=0..seq_len-1 (only L varies per token)
            neg_seq_len = neg_txt.shape[1]
            neg_txt_ids = torch.zeros(1, neg_seq_len, 4, device=device, dtype=torch.long)
            neg_txt_ids[..., 3] = torch.arange(neg_seq_len, device=device, dtype=torch.long)

        # Validate transformer config
        transformer_config = context.models.get_config(self.transformer.transformer)
        assert transformer_config.base == BaseModelType.Flux2 and transformer_config.type == ModelType.Main

        # Calculate the timestep schedule using FLUX.2 specific schedule
        # This matches diffusers' Flux2Pipeline implementation
        # Note: Schedule shifting is handled by the scheduler via mu parameter
        image_seq_len = packed_h * packed_w
        timesteps = get_schedule_flux2(
            num_steps=self.num_steps,
            image_seq_len=image_seq_len,
        )
        # Compute mu for dynamic schedule shifting (used by FlowMatchEulerDiscreteScheduler)
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.num_steps)

        # Clip the timesteps schedule based on denoising_start and denoising_end
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

        # Prepare input latent image
        if init_latents is not None:
            if self.add_noise:
                t_0 = timesteps[0]
                x = t_0 * noise + (1.0 - t_0) * init_latents
            else:
                x = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            x = noise

        # If len(timesteps) == 1, then short-circuit
        if len(timesteps) <= 1:
            return x

        # Generate image position IDs (FLUX.2 uses 4D coordinates)
        # Position IDs use int64 dtype like diffusers
        img_ids = generate_img_ids_flux2(h=latent_h, w=latent_w, batch_size=b, device=device)

        # Prepare inpaint mask
        inpaint_mask = self._prep_inpaint_mask(context, x)

        # Pack all latent tensors
        init_latents_packed = pack_flux2(init_latents) if init_latents is not None else None
        inpaint_mask_packed = pack_flux2(inpaint_mask) if inpaint_mask is not None else None
        noise_packed = pack_flux2(noise)
        x = pack_flux2(x)

        # BN normalization for img2img/inpainting:
        # - The init_latents from VAE encode are NOT BN-normalized
        # - The transformer operates in BN-normalized space
        # - We must normalize x, init_latents, AND noise for InpaintExtension
        # - Output MUST be denormalized after denoising before VAE decode
        #
        # This ensures that:
        # 1. x starts in the correct normalized space for the transformer
        # 2. When InpaintExtension merges intermediate_latents with noised_init_latents,
        #    both are in the same scale/space (noise and init_latents must be in same space
        #    for the linear interpolation: noised = noise * t + init * (1-t))
        if bn_mean is not None and bn_std is not None:
            if init_latents_packed is not None:
                init_latents_packed = self._bn_normalize(init_latents_packed, bn_mean, bn_std)
                # Also normalize noise for InpaintExtension - it's used to compute
                # noised_init_latents = noise * t + init_latents * (1-t)
                # Both operands must be in the same normalized space
                noise_packed = self._bn_normalize(noise_packed, bn_mean, bn_std)
            # For img2img/inpainting, x is computed from init_latents and must also be normalized
            # For txt2img, x is pure noise (already N(0,1)) - normalizing it would be incorrect
            # We detect img2img by checking if init_latents was provided
            if init_latents is not None:
                x = self._bn_normalize(x, bn_mean, bn_std)

        # Verify packed dimensions
        assert packed_h * packed_w == x.shape[1]

        # Prepare inpaint extension
        inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None
        if inpaint_mask_packed is not None:
            assert init_latents_packed is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents_packed,
                inpaint_mask=inpaint_mask_packed,
                noise=noise_packed,
            )

        # Prepare CFG scale list
        num_steps = len(timesteps) - 1
        cfg_scale_list = [self.cfg_scale] * num_steps

        # Check if we're doing inpainting (have a mask or a clipped schedule)
        is_inpainting = self.denoise_mask is not None or self.denoising_start > 1e-5

        # Create scheduler with FLUX.2 Klein configuration
        # For inpainting/img2img, use manual Euler stepping to preserve the exact timestep schedule
        # For txt2img, use the scheduler with dynamic shifting for optimal results
        scheduler = None
        if self.scheduler in FLUX_SCHEDULER_MAP and not is_inpainting:
            # Only use scheduler for txt2img - use manual Euler for inpainting to preserve exact timesteps
            scheduler_class = FLUX_SCHEDULER_MAP[self.scheduler]
            # FlowMatchHeunDiscreteScheduler only supports num_train_timesteps and shift parameters
            # FlowMatchEulerDiscreteScheduler and FlowMatchLCMScheduler support dynamic shifting
            if self.scheduler == "heun":
                scheduler = scheduler_class(
                    num_train_timesteps=1000,
                    shift=3.0,
                )
            else:
                scheduler = scheduler_class(
                    num_train_timesteps=1000,
                    shift=3.0,
                    use_dynamic_shifting=True,
                    base_shift=0.5,
                    max_shift=1.15,
                    base_image_seq_len=256,
                    max_image_seq_len=4096,
                    time_shift_type="exponential",
                )

        # Prepare reference image extension for FLUX.2 Klein built-in editing
        ref_image_extension = None
        if self.kontext_conditioning:
            ref_image_extension = Flux2RefImageExtension(
                context=context,
                ref_image_conditioning=self.kontext_conditioning
                if isinstance(self.kontext_conditioning, list)
                else [self.kontext_conditioning],
                vae_field=self.vae,
                device=device,
                dtype=inference_dtype,
                bn_mean=bn_mean,
                bn_std=bn_std,
            )

        with ExitStack() as exit_stack:
            # Load the transformer model
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            config = transformer_config

            # Determine if the model is quantized
            if config.format in [ModelFormat.Diffusers]:
                model_is_quantized = False
            elif config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                model_is_quantized = False

            # Apply LoRA models to the transformer
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Prepare reference image conditioning if provided
            img_cond_seq = None
            img_cond_seq_ids = None
            if ref_image_extension is not None:
                # Ensure batch sizes match
                ref_image_extension.ensure_batch_size(x.shape[0])
                img_cond_seq, img_cond_seq_ids = (
                    ref_image_extension.ref_image_latents,
                    ref_image_extension.ref_image_ids,
                )

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                cfg_scale=cfg_scale_list,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                scheduler=scheduler,
                mu=mu,
                inpaint_extension=inpaint_extension,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
            )

        # Apply BN denormalization if BN stats are available
        # The diffusers Flux2KleinPipeline applies: latents = latents * bn_std + bn_mean
        # This transforms latents from normalized space to VAE's expected input space
        if bn_mean is not None and bn_std is not None:
            x = self._bn_denormalize(x, bn_mean, bn_std)

        x = unpack_flux2(x.float(), self.height, self.width)
        return x

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> Optional[torch.Tensor]:
        """Prepare the inpaint mask."""
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
        return mask.expand_as(latents)

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply.

        Converts BFL-format LoRA keys to diffusers format if needed, since FLUX.2 Klein
        uses Flux2Transformer2DModel (diffusers naming) but LoRAs may have been loaded
        with BFL naming (e.g. when a Klein 4B LoRA is misidentified as FLUX.1).
        """
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            converted = convert_bfl_lora_patch_to_diffusers(lora_info.model)
            yield (converted, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        """Build a callback for step progress updates."""

        def step_callback(state: PipelineIntermediateState) -> None:
            latents = state.latents.float()
            state.latents = unpack_flux2(latents, self.height, self.width).squeeze()
            context.util.flux2_step_callback(state)

        return step_callback
