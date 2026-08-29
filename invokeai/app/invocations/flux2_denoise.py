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
from invokeai.app.invocations.latent_noise import validate_noise_tensor_shape
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux2.denoise import denoise
from invokeai.backend.flux2.extensions.regional_prompting_extension import Flux2RegionalPromptingExtension
from invokeai.backend.flux2.ref_image_extension import Flux2RefImageExtension
from invokeai.backend.flux2.sampling_utils import (
    compute_empirical_mu,
    generate_img_ids_flux2,
    get_noise_flux2,
    get_schedule_flux2,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.flux2.text_conditioning import Flux2TextConditioning
from invokeai.backend.model_manager.configs.flux2_variant import flux2_hidden_size
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.flux_bfl_peft_lora_conversion_utils import (
    convert_bfl_lora_patch_to_diffusers,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice

# FLUX.2 attention geometry. The head dim is 128 across every variant and the head count follows
# the hidden size (Klein 4B: 3072/24, Klein 9B: 4096/32, [dev] 6144/48), so the width is the single
# number that describes both. Only the head dim decides which SDPA kernel is eligible; the head
# count scales the `math` fallback's score matrix.
FLUX2_ATTENTION_HEAD_DIM = 128
# The width the per-token constant below was measured on. Estimates scale off this.
FLUX2_REFERENCE_HIDDEN_SIZE = 4096
# The widest variant, used when the config does not tell us which one this is -- over-reserving on
# an unknown model beats under-reserving on the largest one.
FLUX2_MAX_HIDDEN_SIZE = 6144


@invocation(
    "flux2_denoise",
    title="FLUX2 Denoise",
    tags=["image", "flux", "flux2", "klein", "denoise"],
    category="latents",
    version="1.6.0",
    classification=Classification.Prototype,
)
class Flux2DenoiseInvocation(BaseInvocation):
    """Run denoising process with a FLUX.2 Klein transformer model.

    This node is designed for FLUX.2 Klein models which use Qwen3 as the text encoder.
    Regional prompting is supported via per-conditioning masks (single mask is applied
    to every transformer block via `joint_attention_kwargs`). ControlNet and IP-Adapters
    are not supported. Regional masking is skipped when reference images are attached.
    """

    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    noise: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.noise,
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
    positive_text_conditioning: FluxConditioningField | list[FluxConditioningField] = InputField(
        description=FieldDescriptions.positive_cond,
        input=Input.Connection,
    )
    negative_text_conditioning: Optional[FluxConditioningField] = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    guidance: float = InputField(
        default=4.0,
        ge=0,
        le=20,
        description="Guidance strength for distilled guidance-embedding models. "
        "Inert for all current FLUX.2 Klein variants (their guidance_embeds weights are absent/zero); "
        "kept for node-graph compatibility and future guidance-embedded models.",
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

        # Prepare input noise (FLUX.2 uses 32 channels).
        # If noise will never be consumed, avoid validating/loading it.
        should_ignore_noise = init_latents is not None and not self.add_noise and self.denoise_mask is None
        noise: Optional[torch.Tensor]
        if should_ignore_noise:
            noise = None
            b, _c, latent_h, latent_w = init_latents.shape
        else:
            noise = self._prepare_noise_tensor(context, inference_dtype, device)
            b, _c, latent_h, latent_w = noise.shape
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        # Load the positive conditioning(s). Supports a single field or a list of regional
        # fields (with optional per-conditioning masks). Masks are preprocessed against the
        # packed latent grid and combined into a single attention mask via the regional
        # extension.
        pos_cond_fields = (
            self.positive_text_conditioning
            if isinstance(self.positive_text_conditioning, list)
            else [self.positive_text_conditioning]
        )
        pos_text_conditionings = self._load_text_conditioning(
            context=context,
            cond_fields=pos_cond_fields,
            packed_height=packed_h,
            packed_width=packed_w,
            dtype=inference_dtype,
            device=device,
        )
        regional_extension = Flux2RegionalPromptingExtension.from_text_conditionings(
            text_conditionings=pos_text_conditionings,
            img_seq_len=packed_h * packed_w,
        )
        txt = regional_extension.regional_text_conditioning.txt_embeddings
        txt_ids = regional_extension.regional_text_conditioning.txt_ids

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
                assert noise is not None
                # Noise the init latents using the first timestep from the clipped
                # InvokeAI schedule.
                #
                # Known limitation: if a scheduler later uses a different first
                # effective timestep/sigma than this precomputed schedule, the
                # img2img preblend below may not match that scheduler exactly.
                # This is an existing pipeline limitation and applies to both
                # seed-generated noise and externally supplied noise.
                t_0 = timesteps[0]
                x = t_0 * noise + (1.0 - t_0) * init_latents
            else:
                x = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            assert noise is not None
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
        noise_packed = pack_flux2(noise) if noise is not None else None
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
                if noise_packed is not None:
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
            assert noise_packed is not None
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
        # For inpainting/img2img, use manual Euler stepping to preserve the exact
        # clipped timestep schedule used for the initial latent/noise preblend.
        # For txt2img, use the scheduler with dynamic shifting for optimal results.
        #
        # This split is intentional. Reusing a scheduler for img2img here can
        # change the first effective timestep/sigma and break parity with the
        # preblend computed above.
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

        # Estimate the peak activation memory the transformer forward will need and ask the model cache
        # to keep that much VRAM free. Without this hint the cache reserves only the small default
        # working memory and fills the rest of the card with the model, so anything beyond a plain
        # low-resolution generation OOMs. Reference images are the dominant term: their latents are
        # concatenated onto the image stream, so three 1024x1024 references quadruple the sequence
        # (and with it the activation footprint) of a 1024x1024 generation.
        ref_image_seq_len = ref_image_extension.ref_image_latents.shape[1] if ref_image_extension is not None else 0
        # The additive bias is skipped entirely when reference images are present (see below), so the
        # mask only costs anything -- storage, and possibly a materialized score matrix -- without them.
        regional_attn_mask = regional_extension.restricted_attn_mask if ref_image_seq_len == 0 else None
        estimated_working_memory = self._estimate_working_memory(
            image_seq_len=packed_h * packed_w,
            ref_image_seq_len=ref_image_seq_len,
            text_seq_len=max(txt.shape[1], neg_txt.shape[1] if neg_txt is not None else 0),
            num_loras=len(self.transformer.loras),
            # Taken from `x`, not from `b`. `b` is the *noise* tensor's batch, which this node builds
            # at 1 from width/height/seed even when the init latents carry more; the img2img preblend
            # above then broadcasts the two, so `x` is the only thing that knows how many samples
            # actually go through the transformer. Reference latents are repeated to match it
            # (`ensure_batch_size` below), so they scale with it too.
            batch_size=x.shape[0],
            # Activation cost per token scales with the transformer's width, and [dev] is 1.5x
            # Klein 9B. Fall back to the widest variant when the config cannot tell us.
            hidden_size=flux2_hidden_size(getattr(transformer_config, "variant", None)) or FLUX2_MAX_HIDDEN_SIZE,
            # The mask itself is already allocated; only the additive bias built per forward is new.
            regional_attention_bias_bytes=(
                regional_attn_mask.numel() * torch.empty((), dtype=inference_dtype).element_size()
                if regional_attn_mask is not None
                else 0
            ),
            has_regional_attention_mask=regional_attn_mask is not None,
            device=device,
            dtype=inference_dtype,
        )

        with ExitStack() as exit_stack:
            # Load the transformer model
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device(
                    working_mem_bytes=estimated_working_memory
                )
            )
            config = transformer_config

            # Determine if the model is quantized
            if config.format in [ModelFormat.Diffusers]:
                model_is_quantized = False
            elif config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
                ModelFormat.SDNQQuantized,
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

            # Regional attention mask is shaped against (txt_len + img_seq_len). When
            # reference images are concatenated to the image stream their tokens are not
            # represented in the mask, so SDPA would error. Skip masking in that case.
            pos_joint_attention_kwargs = None
            if img_cond_seq is None:
                pos_joint_attention_kwargs = regional_extension.get_joint_attention_kwargs(dtype=inference_dtype)
            elif regional_extension.restricted_attn_mask is not None:
                context.logger.warning(
                    "FLUX.2 regional prompting is not supported together with reference images. "
                    "Regional masks will be ignored for this generation."
                )

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                guidance=self.guidance,
                cfg_scale=cfg_scale_list,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                scheduler=scheduler,
                mu=mu,
                inpaint_extension=inpaint_extension,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
                pos_joint_attention_kwargs=pos_joint_attention_kwargs,
            )

        # Apply BN denormalization if BN stats are available
        # The diffusers Flux2KleinPipeline applies: latents = latents * bn_std + bn_mean
        # This transforms latents from normalized space to VAE's expected input space
        if bn_mean is not None and bn_std is not None:
            x = self._bn_denormalize(x, bn_mean, bn_std)

        x = unpack_flux2(x.float(), self.height, self.width)
        return x

    def _prepare_noise_tensor(
        self, context: InvocationContext, inference_dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self.noise is not None:
            noise = context.tensors.load(self.noise.latents_name).to(device=device, dtype=inference_dtype)
            validate_noise_tensor_shape(noise, "FLUX.2", self.width, self.height)
            return noise

        return get_noise_flux2(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )

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

    def _estimate_working_memory(
        self,
        image_seq_len: int,
        ref_image_seq_len: int,
        text_seq_len: int,
        num_loras: int,
        batch_size: int = 1,
        hidden_size: int = FLUX2_REFERENCE_HIDDEN_SIZE,
        regional_attention_bias_bytes: int = 0,
        has_regional_attention_mask: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Estimate peak transformer activation memory (bytes) so the model cache reserves enough headroom.

        FLUX.2 attention runs through SDPA without materializing the O(seq^2) score matrix, so the
        activation footprint scales *linearly* with the total attended sequence -- text tokens, image
        tokens, and reference-image tokens alike. Measured on the Klein 9B geometry in bf16 as peak
        reserved memory, that slope is ~0.39 MB per token and holds from 1.5k to 28k tokens; it is
        also independent of the block count (a no-grad forward frees each block's intermediates).

        It is *not* independent of the transformer's width, which is why ``hidden_size`` is a
        parameter rather than a constant. Measured slope between 4608 and 9216 tokens, block count
        and everything else held fixed:

            3072 (Klein 4B) 0.291 MB/tok    4096 (Klein 9B) 0.386    6144 ([dev]) 0.555

        which is 0.755 / 1.000 / 1.438 against width ratios of 0.75 / 1.00 / 1.50 -- linear in width,
        and slightly sub-linear at the top so scaling by width stays an upper bound. Calibrating on
        Klein 9B alone would have under-reserved [dev] by a third: 1024x1024 with three 1024x1024
        references is 16896 tokens, ~7.6GB reserved against ~10GB needed. The head count follows the
        same width, so the score-matrix term gets the real one instead of the widest.

        The reference-image term is what makes this estimate necessary rather than merely nice to
        have: a 1024x1024 generation is 4096 image tokens (~1.7GB), but attaching three 1024x1024
        references adds 12288 more for ~6.5GB, and a 1328px tile with three 1328px references reaches
        ~10.9GB -- against a default ``device_working_mem_gb`` of 3.

        A fixed base covers resolution-independent overhead (transient fp8/GGUF -> bf16 weight casts
        during the forward, and allocator slack across many steps). LoRA sidecar patches add an extra
        activation branch per patched layer, so we add a per-LoRA margin.

        Batch multiplies the token count and nothing else. A batch of B is B independent sequences,
        so it enters the linear term exactly as extra sequence does -- measured on the Klein geometry
        with a reduced block count: 4608 tokens at B=1 peaks at 2570MB, the same 4608 at B=2 (9216
        tokens) at 5126MB, and 9728 tokens at B=1 at 5584MB. Batch and sequence are interchangeable
        to within the noise. Reference latents are repeated per sample (`ensure_batch_size`), so they
        scale with it too, and the score matrix is shaped (batch, heads, S, S). The fixed base does
        not scale -- it is about weights, not activations -- and neither does the regional bias, which
        is built as (1, 1, S, S) and broadcast across the batch.

        The linear model holds only while attention runs on a fused kernel. Regional prompting is
        where that stops being a given: it hands the transformer a dense additive ``S x S`` bias,
        which flash attention never accepts, leaving whatever else the build has -- and if that is
        the ``math`` fallback, a materialized ``heads x S x S`` score matrix. Whether a mask forces
        that is build-specific and not worth predicting: CUDA's memory-efficient kernel takes it, and
        so does ROCm's on gfx1100. The device decides too (MPS has no fused SDPA kernel at all), and
        so does the diffusers attention backend this build dispatches through.
        ``sdpa_score_matrix_bytes`` asks all three and adds the score matrix only where it is really
        built: on CUDA with the stock backend the term is zero (verified: peak stays linear with the
        bias attached).
        """
        GB = 1024**3
        MB = 1024**2
        per_token_bytes = int(0.4 * MB * hidden_size / FLUX2_REFERENCE_HIDDEN_SIZE)
        total_seq_len = image_seq_len + ref_image_seq_len + text_seq_len
        estimated = total_seq_len * batch_size * per_token_bytes
        estimated += int(1.0 * GB)
        estimated += regional_attention_bias_bytes
        estimated += sdpa_score_matrix_bytes(
            device=device if device is not None else TorchDevice.choose_torch_device(),
            dtype=dtype,
            num_heads=(hidden_size // FLUX2_ATTENTION_HEAD_DIM) * batch_size,
            head_dim=FLUX2_ATTENTION_HEAD_DIM,
            seq_len=total_seq_len,
            has_attn_mask=has_regional_attention_mask,
            # The FLUX.2 transformer's attention goes through diffusers' `dispatch_attention_fn`,
            # which can route around torch's SDPA entirely -- including to a forced `math` backend.
            via_diffusers_dispatch=True,
        )
        if num_loras > 0:
            # A sidecar branch is an activation, so it scales with the batch like the rest of them.
            estimated += int(0.5 * num_loras * batch_size * GB)
        return estimated

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        cond_fields: list[FluxConditioningField],
        packed_height: int,
        packed_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Flux2TextConditioning]:
        """Load FLUX.2 Klein text conditionings, preprocessing per-conditioning regional masks."""
        out: list[Flux2TextConditioning] = []
        for field in cond_fields:
            cond_data = context.conditioning.load(field.conditioning_name)
            assert len(cond_data.conditionings) == 1
            info = cond_data.conditionings[0]
            assert isinstance(info, FLUXConditioningInfo)
            info = info.to(dtype=dtype, device=device)

            # mask=None marks a global prompt; only preprocess when a mask field is attached
            mask_processed: torch.Tensor | None = None
            if field.mask is not None:
                mask_tensor = context.tensors.load(field.mask.tensor_name).to(device=device)
                mask_processed = Flux2RegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask=mask_tensor,
                    packed_height=packed_height,
                    packed_width=packed_width,
                    dtype=dtype,
                    device=device,
                )
            out.append(Flux2TextConditioning(txt_embeddings=info.t5_embeds, mask=mask_processed))
        return out

    def _lora_iterator(self, context: InvocationContext) -> Iterator[PatchSpec]:
        """Iterate over LoRA models to apply.

        Converts BFL-format LoRA keys to diffusers format if needed, since FLUX.2 Klein
        uses Flux2Transformer2DModel (diffusers naming) but LoRAs may have been loaded
        with BFL naming (e.g. when a Klein 4B LoRA is misidentified as FLUX.1).
        """
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            converted = convert_bfl_lora_patch_to_diffusers(lora_info.model)
            yield (converted, lora.weight, lora_info.model_in_ram())

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        """Build a callback for step progress updates."""

        def step_callback(state: PipelineIntermediateState) -> None:
            latents = state.latents.float()
            state.latents = unpack_flux2(latents, self.height, self.width).squeeze()
            context.util.flux2_step_callback(state)

        return step_callback
