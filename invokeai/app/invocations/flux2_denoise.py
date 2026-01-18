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
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional,
    get_schedule,
)
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux2.denoise import denoise
from invokeai.backend.flux2.sampling_utils import (
    generate_img_ids_flux2,
    get_noise_flux2,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
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
    version="1.0.0",
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
        default=28,
        description="Number of diffusion steps. Recommended: 28 for Klein.",
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

            # Log VAE type for debugging
            context.logger.debug(f"FLUX.2 VAE type: {type(vae).__name__}")

            # Try to find the BN layer - it may be at different locations depending on model format
            bn_layer = None
            if hasattr(vae, "bn"):
                bn_layer = vae.bn
                context.logger.info(f"Found BN layer at vae.bn: {type(bn_layer).__name__}")
            elif hasattr(vae, "batch_norm"):
                bn_layer = vae.batch_norm
                context.logger.info(f"Found BN layer at vae.batch_norm: {type(bn_layer).__name__}")
            elif hasattr(vae, "encoder") and hasattr(vae.encoder, "bn"):
                bn_layer = vae.encoder.bn
                context.logger.info(f"Found BN layer at vae.encoder.bn: {type(bn_layer).__name__}")

            if bn_layer is None:
                context.logger.warning(
                    "FLUX.2 VAE does not have a BatchNorm layer at expected location. "
                    "Skipping BN denormalization - the VAE may handle this internally."
                )
                # Log available attributes for debugging
                vae_attrs = [attr for attr in dir(vae) if not attr.startswith("_")]
                context.logger.info(f"VAE class: {type(vae).__name__}, attributes: {vae_attrs[:30]}")
                return None

            # Verify running statistics are initialized
            if bn_layer.running_mean is None or bn_layer.running_var is None:
                context.logger.warning(
                    "FLUX.2 VAE BN layer has uninitialized running statistics. "
                    "This may indicate the model wasn't properly loaded with pretrained weights."
                )
                return None

            # Get BN running statistics from VAE
            bn_mean = bn_layer.running_mean.clone()  # Shape: (128,)
            bn_var = bn_layer.running_var.clone()  # Shape: (128,)
            bn_eps = bn_layer.eps if hasattr(bn_layer, "eps") else 1e-4  # BFL uses 1e-4
            bn_std = torch.sqrt(bn_var + bn_eps)

            # Validate BN statistics are not corrupted
            if bn_mean.isnan().any() or bn_std.isnan().any():
                context.logger.warning("FLUX.2 BN statistics contain NaN values!")
            if bn_std.min() < 1e-6:
                context.logger.warning(f"FLUX.2 BN std contains very small values: min={bn_std.min().item():.6f}")

            context.logger.debug(
                f"FLUX.2 BN stats: mean=[{bn_mean.min().item():.4f}, {bn_mean.max().item():.4f}], "
                f"std=[{bn_std.min().item():.4f}, {bn_std.max().item():.4f}]"
            )

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

        # Debug: Check for NaN in conditioning
        if txt.isnan().any():
            context.logger.warning(f"FLUX.2: NaN detected in text conditioning! Shape: {txt.shape}")
        else:
            context.logger.info(
                f"FLUX.2 conditioning (raw): shape={txt.shape}, dtype={txt.dtype}, "
                f"min={txt.min().item():.4f}, max={txt.max().item():.4f}, mean={txt.mean().item():.4f}"
            )

        # Generate text position IDs (4D format for FLUX.2: T, H, W, L)
        # FLUX.2 uses 4D position coordinates for its rotary position embeddings
        # IMPORTANT: Position IDs must be int64 (long) like diffusers, not bfloat16
        # For text tokens: T=1, H=1, W=1, L=0..seq_len-1 (L varies per token)
        seq_len = txt.shape[1]
        txt_ids = torch.ones(1, seq_len, 4, device=device, dtype=torch.long)
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
            # For text tokens: T=1, H=1, W=1, L=0..seq_len-1 (L varies per token)
            neg_seq_len = neg_txt.shape[1]
            neg_txt_ids = torch.ones(1, neg_seq_len, 4, device=device, dtype=torch.long)
            neg_txt_ids[..., 3] = torch.arange(neg_seq_len, device=device, dtype=torch.long)

        # Validate transformer config
        transformer_config = context.models.get_config(self.transformer.transformer)
        assert transformer_config.base == BaseModelType.Flux2 and transformer_config.type == ModelType.Main

        # Calculate the timestep schedule
        # Klein uses shifted schedule like Dev
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=packed_h * packed_w,
            shift=True,  # Klein uses shifted schedule
        )

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

        # Debug: Check for NaN in packed inputs before denoising
        if x.isnan().any():
            context.logger.warning(f"FLUX.2: NaN in packed latents BEFORE denoising! Shape: {x.shape}")
        else:
            context.logger.debug(
                f"FLUX.2 packed latents: shape={x.shape}, min={x.min().item():.4f}, "
                f"max={x.max().item():.4f}, mean={x.mean().item():.4f}"
            )

        # Debug: Verify position IDs dtype and values
        context.logger.info(
            f"FLUX.2 position IDs: img_ids dtype={img_ids.dtype}, txt_ids dtype={txt_ids.dtype}, "
            f"img_ids shape={img_ids.shape}, txt_ids shape={txt_ids.shape}"
        )
        context.logger.info(
            f"FLUX.2 txt_ids sample (first 3 tokens): {txt_ids[0, :3, :].tolist()}, "
            f"last token: {txt_ids[0, -1, :].tolist()}"
        )
        context.logger.info(
            f"FLUX.2 img_ids sample (first 3 patches): {img_ids[0, :3, :].tolist()}, "
            f"last patch: {img_ids[0, -1, :].tolist()}"
        )

        # Apply BN normalization BEFORE denoising (as per diffusers Flux2KleinPipeline)
        # BN normalization: y = (x - mean) / std
        # This transforms latents to normalized space for the transformer
        if bn_mean is not None and bn_std is not None:
            context.logger.debug(
                f"FLUX.2 packed latents before BN norm: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )
            x = self._bn_normalize(x, bn_mean, bn_std)
            context.logger.info(
                f"FLUX.2 packed latents after BN norm: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )

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

        # Create scheduler if not using default euler
        scheduler = None
        if self.scheduler in FLUX_SCHEDULER_MAP:
            scheduler_class = FLUX_SCHEDULER_MAP[self.scheduler]
            scheduler = scheduler_class(num_train_timesteps=1000)

        with ExitStack() as exit_stack:
            # Load the transformer model
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            config = transformer_config

            # Debug: Check context_embedder dimensions vs Qwen3 embeddings
            if hasattr(transformer, "context_embedder"):
                ctx_emb = transformer.context_embedder
                if hasattr(ctx_emb, "weight"):
                    ctx_in_dim = ctx_emb.weight.shape[1]
                    ctx_out_dim = ctx_emb.weight.shape[0]
                    txt_dim = txt.shape[-1]
                    ctx_weight_dtype = ctx_emb.weight.dtype
                    context.logger.info(
                        f"FLUX.2 transformer context_embedder: in_dim={ctx_in_dim}, out_dim={ctx_out_dim}, "
                        f"weight_dtype={ctx_weight_dtype}, Qwen3 txt_dim={txt_dim}, txt_dtype={txt.dtype}"
                    )
                    if ctx_in_dim != txt_dim:
                        context.logger.error(
                            f"DIMENSION MISMATCH! Transformer expects {ctx_in_dim}-dim input but got {txt_dim}-dim Qwen3 embeddings!"
                        )

                    # Debug: Check if context_embedder weights contain NaN
                    if ctx_emb.weight.isnan().any():
                        context.logger.error("context_embedder.weight contains NaN!")
                    if hasattr(ctx_emb, "bias") and ctx_emb.bias is not None and ctx_emb.bias.isnan().any():
                        context.logger.error("context_embedder.bias contains NaN!")

                    # Log context_embedder weight statistics
                    context.logger.info(
                        f"context_embedder.weight stats: min={ctx_emb.weight.min().item():.6f}, "
                        f"max={ctx_emb.weight.max().item():.6f}, std={ctx_emb.weight.std().item():.6f}"
                    )

                    # Debug: Test context_embedder forward pass
                    with torch.no_grad():
                        # Ensure matching dtype for the matmul
                        txt_for_ctx = txt.to(dtype=ctx_weight_dtype)
                        test_ctx_out = ctx_emb(txt_for_ctx)
                        if test_ctx_out.isnan().any():
                            context.logger.error(
                                f"context_embedder produces NaN! Input: min={txt_for_ctx.min().item():.4f}, max={txt_for_ctx.max().item():.4f}, "
                                f"Output: nan_count={test_ctx_out.isnan().sum().item()}"
                            )
                        else:
                            context.logger.info(
                                f"context_embedder output: min={test_ctx_out.min().item():.4f}, "
                                f"max={test_ctx_out.max().item():.4f}, mean={test_ctx_out.mean().item():.4f}"
                            )

            # Debug: Check x_embedder
            if hasattr(transformer, "x_embedder"):
                x_emb = transformer.x_embedder
                if hasattr(x_emb, "weight"):
                    if x_emb.weight.isnan().any():
                        context.logger.error("x_embedder.weight contains NaN!")
                    with torch.no_grad():
                        test_x_out = x_emb(x)
                        if test_x_out.isnan().any():
                            context.logger.error(f"x_embedder produces NaN!")
                        else:
                            context.logger.info(
                                f"x_embedder output OK: min={test_x_out.min().item():.4f}, "
                                f"max={test_x_out.max().item():.4f}"
                            )

            # Debug: Check pos_embed (position embeddings)
            # Note: pos_embed expects 2D input (seq_len, 4), not 3D (batch, seq_len, 4)
            if hasattr(transformer, "pos_embed"):
                pos_emb = transformer.pos_embed
                with torch.no_grad():
                    try:
                        # Extract first batch element for testing (pos_embed expects 2D)
                        test_img_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
                        test_pos_out = pos_emb(test_img_ids)
                        if isinstance(test_pos_out, tuple):
                            for i, t in enumerate(test_pos_out):
                                if t.isnan().any():
                                    context.logger.error(f"pos_embed output[{i}] contains NaN!")
                        elif test_pos_out.isnan().any():
                            context.logger.error(f"pos_embed produces NaN!")
                        else:
                            context.logger.info("pos_embed output OK")
                    except Exception as e:
                        context.logger.error(f"pos_embed error: {e}")

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
            )

            # Debug: Check for NaN immediately after denoising
            if x.isnan().any():
                nan_count = x.isnan().sum().item()
                total_elements = x.numel()
                context.logger.error(
                    f"FLUX.2: NaN detected AFTER denoising! "
                    f"{nan_count}/{total_elements} elements are NaN ({100*nan_count/total_elements:.1f}%)"
                )
            else:
                context.logger.debug(
                    f"FLUX.2 after denoise: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                    f"mean={x.mean().item():.4f}"
                )

            # Apply inpainting if enabled
            if inpaint_extension is not None:
                x = inpaint_extension.merge_intermediate_latents_with_init_latents(x, 0.0)

        # Apply BN denormalization if BN stats are available
        # The diffusers Flux2KleinPipeline applies: latents = latents * bn_std + bn_mean
        # This transforms latents from normalized space to VAE's expected input space
        if bn_mean is not None and bn_std is not None:
            context.logger.debug(
                f"FLUX.2 latents before BN denorm: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )
            x = self._bn_denormalize(x, bn_mean, bn_std)
            context.logger.info(
                f"FLUX.2 latents after BN denorm: min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )
        else:
            context.logger.info(
                f"FLUX.2 latents (no BN stats): min={x.min().item():.4f}, max={x.max().item():.4f}, "
                f"mean={x.mean().item():.4f}"
            )

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
        """Iterate over LoRA models to apply."""
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        """Build a callback for step progress updates."""

        def step_callback(state: PipelineIntermediateState) -> None:
            latents = state.latents.float()
            state.latents = unpack_flux2(latents, self.height, self.width).squeeze()
            context.util.flux2_step_callback(state)

        return step_callback
