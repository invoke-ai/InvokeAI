from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple, cast

import einops
import torch
import torchvision.transforms as tv_transforms
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from PIL import Image
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
    ZImageConditioningField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ZImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter
from invokeai.backend.z_image.z_image_control_transformer import ZImageControlTransformer2DModel


@invocation(
    "z_image_denoise",
    title="Denoise - Z-Image",
    tags=["image", "z-image"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class ZImageDenoiseInvocation(BaseInvocation):
    """Run the denoising process with a Z-Image model."""

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
        description=FieldDescriptions.z_image_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: ZImageConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[ZImageConditioningField] = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # Z-Image-Turbo uses guidance_scale=0.0 by default (no CFG)
    guidance_scale: float = InputField(
        default=0.0,
        ge=0.0,
        description="Guidance scale for classifier-free guidance. Use 0.0 for Z-Image-Turbo.",
        title="Guidance Scale",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    # Z-Image-Turbo uses 8 steps by default
    steps: int = InputField(default=8, gt=0, description="Number of denoising steps. 8 recommended for Z-Image-Turbo.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    # Z-Image Control support
    control: Optional[ZImageControlField] = InputField(
        default=None,
        description="Z-Image control conditioning for spatial control (Canny, HED, Depth, Pose, MLSD).",
        input=Input.Connection,
    )
    # VAE for encoding control images (required when using control)
    vae: Optional[VAEField] = InputField(
        default=None,
        description=FieldDescriptions.vae + " Required for control conditioning.",
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        """Prepare the inpaint mask."""
        if self.denoise_mask is None:
            return None
        mask = context.tensors.load(self.denoise_mask.mask_name)

        # Invert mask: 0.0 = regions to denoise, 1.0 = regions to preserve
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
    ) -> torch.Tensor:
        """Load Z-Image text conditioning."""
        cond_data = context.conditioning.load(conditioning_name)
        if len(cond_data.conditionings) != 1:
            raise ValueError(
                f"Expected exactly 1 conditioning entry for Z-Image, got {len(cond_data.conditionings)}. "
                "Ensure you are using the Z-Image text encoder."
            )
        z_image_conditioning = cond_data.conditionings[0]
        if not isinstance(z_image_conditioning, ZImageConditioningInfo):
            raise TypeError(
                f"Expected ZImageConditioningInfo, got {type(z_image_conditioning).__name__}. "
                "Ensure you are using the Z-Image text encoder."
            )
        z_image_conditioning = z_image_conditioning.to(dtype=dtype, device=device)
        return z_image_conditioning.prompt_embeds

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
        """Generate initial noise tensor."""
        # Generate noise as float32 on CPU for maximum compatibility,
        # then cast to target dtype/device
        rand_device = "cpu"
        rand_dtype = torch.float32

        return torch.randn(
            batch_size,
            num_channels_latents,
            int(height) // LATENT_SCALE_FACTOR,
            int(width) // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to(device=device, dtype=dtype)

    def _calculate_shift(
        self,
        image_seq_len: int,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        """Calculate timestep shift based on image sequence length.

        Based on diffusers ZImagePipeline.calculate_shift method.
        """
        m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
        b = base_shift - m * base_image_seq_len
        mu = image_seq_len * m + b
        return mu

    def _get_sigmas(self, mu: float, num_steps: int) -> list[float]:
        """Generate sigma schedule with time shift.

        Based on FlowMatchEulerDiscreteScheduler with shift.
        Generates num_steps + 1 sigma values (including terminal 0.0).
        """
        import math

        def time_shift(mu: float, sigma: float, t: float) -> float:
            """Apply time shift to a single timestep value."""
            if t <= 0:
                return 0.0
            if t >= 1:
                return 1.0
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

        # Generate linearly spaced values from 1 to 0 (excluding endpoints for safety)
        # then apply time shift
        sigmas = []
        for i in range(num_steps + 1):
            t = 1.0 - i / num_steps  # Goes from 1.0 to 0.0
            sigma = time_shift(mu, 1.0, t)
            sigmas.append(sigma)

        return sigmas

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        transformer_info = context.models.load(self.transformer.transformer)

        # Load positive conditioning
        pos_prompt_embeds = self._load_text_conditioning(
            context=context,
            conditioning_name=self.positive_conditioning.conditioning_name,
            dtype=inference_dtype,
            device=device,
        )

        # Load negative conditioning if provided and guidance_scale > 0
        neg_prompt_embeds: torch.Tensor | None = None
        do_classifier_free_guidance = self.guidance_scale > 0.0 and self.negative_conditioning is not None
        if do_classifier_free_guidance:
            if self.negative_conditioning is None:
                raise ValueError("Negative conditioning is required when guidance_scale > 0")
            neg_prompt_embeds = self._load_text_conditioning(
                context=context,
                conditioning_name=self.negative_conditioning.conditioning_name,
                dtype=inference_dtype,
                device=device,
            )

        # Calculate image sequence length for timestep shifting
        patch_size = 2  # Z-Image uses patch_size=2
        image_seq_len = ((self.height // LATENT_SCALE_FACTOR) * (self.width // LATENT_SCALE_FACTOR)) // (patch_size**2)

        # Calculate shift based on image sequence length
        mu = self._calculate_shift(image_seq_len)

        # Generate sigma schedule with time shift
        sigmas = self._get_sigmas(mu, self.steps)

        # Apply denoising_start and denoising_end clipping
        if self.denoising_start > 0 or self.denoising_end < 1:
            # Calculate start and end indices based on denoising range
            total_sigmas = len(sigmas)
            start_idx = int(self.denoising_start * (total_sigmas - 1))
            end_idx = int(self.denoising_end * (total_sigmas - 1)) + 1
            sigmas = sigmas[start_idx:end_idx]

        total_steps = len(sigmas) - 1

        # Load input latents if provided (image-to-image)
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        # Generate initial noise
        num_channels_latents = 16  # Z-Image uses 16 latent channels
        noise = self._get_noise(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=self.height,
            width=self.width,
            dtype=inference_dtype,
            device=device,
            seed=self.seed,
        )

        # Prepare input latent image
        if init_latents is not None:
            s_0 = sigmas[0]
            latents = s_0 * noise + (1.0 - s_0) * init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        # Short-circuit if no denoising steps
        if total_steps <= 0:
            return latents

        # Prepare inpaint extension
        inpaint_mask = self._prep_inpaint_mask(context, latents)
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            if init_latents is None:
                raise ValueError("Initial latents are required when using an inpaint mask (image-to-image inpainting)")
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
                timestep=int(sigmas[0] * 1000),
                latents=latents,
            ),
        )

        with ExitStack() as exit_stack:
            # Get transformer config to determine if it's quantized
            transformer_config = context.models.get_config(self.transformer.transformer)

            # Determine if the model is quantized.
            # If the model is quantized, then we need to apply the LoRA weights as sidecar layers. This results in
            # slower inference than direct patching, but is agnostic to the quantization format.
            if transformer_config.format in [ModelFormat.Diffusers, ModelFormat.Checkpoint]:
                model_is_quantized = False
            elif transformer_config.format in [ModelFormat.GGUFQuantized]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported Z-Image model format: {transformer_config.format}")

            # Load control adapter and prepare combined transformer if control is provided
            control_context: list[torch.Tensor] | None = None
            control_context_scale = 0.75
            begin_step_percent = 0.0
            end_step_percent = 1.0
            cached_weights = None
            control_in_dim = 16  # Default, will be set from adapter config if control is used

            if self.control is not None:
                # Load base transformer config (NOT to GPU yet - just get the model reference)
                base_transformer = cast(ZImageTransformer2DModel, transformer_info.model)
                base_config = base_transformer.config

                # Load control adapter
                control_model_info = context.models.load(self.control.control_model)
                control_adapter = control_model_info.model
                assert isinstance(control_adapter, ZImageControlAdapter)

                # Get control_in_dim from adapter config (16 for V1, 33 for V2.0)
                adapter_config = control_adapter.config
                control_in_dim = adapter_config.get("control_in_dim", 16)
                num_control_blocks = adapter_config.get("num_control_blocks", 6)
                n_refiner_layers = adapter_config.get("n_refiner_layers", 2)

                # Calculate control_layers_places based on num_control_blocks
                control_layers_places = [i * 2 for i in range(num_control_blocks)]

                # Log control configuration for debugging
                version = "V2.0" if control_in_dim > 16 else "V1"
                context.util.signal_progress(
                    f"Using Z-Image ControlNet {version}: control_in_dim={control_in_dim}, "
                    f"num_blocks={num_control_blocks}, scale={self.control.control_context_scale}"
                )

                # Create control transformer structure with empty weights
                import accelerate

                with accelerate.init_empty_weights():
                    control_transformer = ZImageControlTransformer2DModel(
                        control_layers_places=control_layers_places,
                        control_in_dim=control_in_dim,
                        all_patch_size=base_config.all_patch_size,
                        all_f_patch_size=base_config.all_f_patch_size,
                        in_channels=base_config.in_channels,
                        dim=base_config.dim,
                        n_layers=base_config.n_layers,
                        n_refiner_layers=n_refiner_layers,
                        n_heads=base_config.n_heads,
                        n_kv_heads=base_config.n_kv_heads,
                        norm_eps=base_config.norm_eps,
                        qk_norm=base_config.qk_norm,
                        cap_feat_dim=base_config.cap_feat_dim,
                        rope_theta=base_config.rope_theta,
                        t_scale=base_config.t_scale,
                        axes_dims=base_config.axes_dims,
                        axes_lens=base_config.axes_lens,
                    )

                # Load base weights with assign=True (assigns tensors directly, no copy of data)
                control_transformer.load_state_dict(base_transformer.state_dict(), strict=False, assign=True)

                # Load control adapter weights on top
                adapter_weights = {k: v for k, v in control_adapter.state_dict().items() if k.startswith("control_")}
                control_transformer.load_state_dict(adapter_weights, strict=False, assign=True)

                # Move combined model to device
                control_transformer = control_transformer.to(device=device, dtype=inference_dtype)
                active_transformer = control_transformer

                # Clean up to save memory # need to check
                #del control_adapter
                #if torch.cuda.is_available():
                #    torch.cuda.empty_cache()

                # Load and prepare control image - must be VAE-encoded!
                if self.vae is None:
                    raise ValueError("VAE is required when using Z-Image Control. Connect a VAE to the 'vae' input.")

                control_image = context.images.get_pil(self.control.image_name)

                # Resize control image to match output dimensions
                control_image = control_image.convert("RGB")
                control_image = control_image.resize((self.width, self.height), Image.Resampling.LANCZOS)

                # Convert to tensor format for VAE encoding
                from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor

                control_image_tensor = image_resized_to_grid_as_tensor(control_image)
                if control_image_tensor.dim() == 3:
                    control_image_tensor = einops.rearrange(control_image_tensor, "c h w -> 1 c h w")

                # Encode control image through VAE to get latents
                vae_info = context.models.load(self.vae.vae)
                control_latents = ZImageImageToLatentsInvocation.vae_encode(
                    vae_info=vae_info,
                    image_tensor=control_image_tensor,
                )

                # Move to inference device/dtype
                control_latents = control_latents.to(device=device, dtype=inference_dtype)

                # Add frame dimension: [B, C, H, W] -> [B, C, 1, H, W]
                control_latents = control_latents.unsqueeze(2)

                # Prepare control_context based on control_in_dim
                # V1: 16 channels (just control latents)
                # V2.0: 33 channels (control latents + zero padding)
                # Following diffusers approach: simple zero-padding to match control_in_dim
                b, c, f, h, w = control_latents.shape
                if c < control_in_dim:
                    # Pad with zeros to match control_in_dim (diffusers approach)
                    padding_channels = control_in_dim - c
                    zero_padding = torch.zeros(
                        (b, padding_channels, f, h, w),
                        device=device,
                        dtype=inference_dtype,
                    )
                    control_latents = torch.cat([control_latents, zero_padding], dim=1)
                control_context = list(control_latents.unbind(dim=0))

                control_context_scale = self.control.control_context_scale
                begin_step_percent = self.control.begin_step_percent
                end_step_percent = self.control.end_step_percent
            else:
                # No control - load transformer normally
                (cached_weights, transformer) = exit_stack.enter_context(transformer_info.model_on_device())
                active_transformer = transformer

            # Apply LoRA models to the active transformer.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            # cached_weights is None when using control (since we create a new combined model),
            # otherwise it comes from model_on_device() context.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=active_transformer,
                    patches=self._lora_iterator(context),
                    prefix=Z_IMAGE_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Denoising loop
            for step_idx in tqdm(range(total_steps)):
                sigma_curr = sigmas[step_idx]
                sigma_prev = sigmas[step_idx + 1]

                # Timestep tensor for Z-Image model
                # The model expects t=0 at start (noise) and t=1 at end (clean)
                # Sigma goes from 1 (noise) to 0 (clean), so model_t = 1 - sigma
                model_t = 1.0 - sigma_curr
                timestep = torch.tensor([model_t], device=device, dtype=inference_dtype).expand(latents.shape[0])

                # Run transformer for positive prediction
                # Z-Image transformer expects: x as list of [C, 1, H, W] tensors, t, cap_feats as list
                # Prepare latent input: [B, C, H, W] -> [B, C, 1, H, W] -> list of [C, 1, H, W]
                latent_model_input = latents.to(active_transformer.dtype)
                latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                # Determine if control should be applied at this step
                step_percent = step_idx / total_steps
                use_control = self.control is not None
                apply_control = (
                    use_control
                    and step_percent >= begin_step_percent
                    and step_percent <= end_step_percent
                )

                # Transformer returns (List[torch.Tensor], dict) - we only need the tensor list
                # If control is active, pass control_context to the control transformer
                if apply_control and control_context is not None:
                    model_output = active_transformer(
                        x=latent_model_input_list,
                        t=timestep,
                        cap_feats=[pos_prompt_embeds],
                        control_context=control_context,
                        control_context_scale=control_context_scale,
                    )
                else:
                    model_output = active_transformer(
                        x=latent_model_input_list,
                        t=timestep,
                        cap_feats=[pos_prompt_embeds],
                    )
                model_out_list = model_output[0]  # Extract list of tensors from tuple
                noise_pred_cond = torch.stack([t.float() for t in model_out_list], dim=0)
                noise_pred_cond = noise_pred_cond.squeeze(2)  # Remove frame dimension
                noise_pred_cond = -noise_pred_cond  # Z-Image uses v-prediction with negation

                # Apply CFG if enabled
                if do_classifier_free_guidance and neg_prompt_embeds is not None:
                    if apply_control and control_context is not None:
                        model_output_uncond = active_transformer(
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[neg_prompt_embeds],
                            control_context=control_context,
                            control_context_scale=control_context_scale,
                        )
                    else:
                        model_output_uncond = active_transformer(
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[neg_prompt_embeds],
                        )
                    model_out_list_uncond = model_output_uncond[0]  # Extract list of tensors from tuple
                    noise_pred_uncond = torch.stack([t.float() for t in model_out_list_uncond], dim=0)
                    noise_pred_uncond = noise_pred_uncond.squeeze(2)
                    noise_pred_uncond = -noise_pred_uncond
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Euler step
                latents_dtype = latents.dtype
                latents = latents.to(dtype=torch.float32)
                latents = latents + (sigma_prev - sigma_curr) * noise_pred
                latents = latents.to(dtype=latents_dtype)

                if inpaint_extension is not None:
                    latents = inpaint_extension.merge_intermediate_latents_with_init_latents(latents, sigma_prev)

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(sigma_curr * 1000),
                        latents=latents,
                    ),
                )

        return latents

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.ZImage)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the transformer."""
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
