import inspect
import math
from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import einops
import torch
import torchvision.transforms as tv_transforms
from diffusers.schedulers.scheduling_utils import SchedulerMixin
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
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.z_image_control import ZImageControlField
from invokeai.app.invocations.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.schedulers import ZIMAGE_SCHEDULER_LABELS, ZIMAGE_SCHEDULER_MAP, ZIMAGE_SCHEDULER_NAME_VALUES
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ZImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.z_image.extensions.regional_prompting_extension import ZImageRegionalPromptingExtension
from invokeai.backend.z_image.text_conditioning import ZImageTextConditioning
from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter
from invokeai.backend.z_image.z_image_controlnet_extension import (
    ZImageControlNetExtension,
    z_image_forward_with_control,
)
from invokeai.backend.z_image.z_image_transformer_patch import patch_transformer_for_regional_prompting


@invocation(
    "z_image_denoise",
    title="Denoise - Z-Image",
    tags=["image", "z-image"],
    category="image",
    version="1.3.0",
    classification=Classification.Prototype,
)
class ZImageDenoiseInvocation(BaseInvocation):
    """Run the denoising process with a Z-Image model.

    Supports regional prompting by connecting multiple conditioning inputs with masks.
    """

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
    positive_conditioning: ZImageConditioningField | list[ZImageConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ZImageConditioningField | list[ZImageConditioningField] | None = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # Z-Image-Turbo works best without CFG (guidance_scale=1.0)
    guidance_scale: float = InputField(
        default=1.0,
        ge=1.0,
        description="Guidance scale for classifier-free guidance. 1.0 = no CFG (recommended for Z-Image-Turbo). "
        "Values > 1.0 amplify guidance.",
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
    # Scheduler selection for the denoising process
    scheduler: ZIMAGE_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process. Euler is the default and recommended for "
        "Z-Image-Turbo. Heun is 2nd-order (better quality, 2x slower). LCM is optimized for few steps.",
        ui_choice_labels=ZIMAGE_SCHEDULER_LABELS,
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
        cond_field: ZImageConditioningField | list[ZImageConditioningField],
        img_height: int,
        img_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[ZImageTextConditioning]:
        """Load Z-Image text conditioning with optional regional masks.

        Args:
            context: The invocation context.
            cond_field: Single conditioning field or list of fields.
            img_height: Height of the image token grid (H // patch_size).
            img_width: Width of the image token grid (W // patch_size).
            dtype: Target dtype.
            device: Target device.

        Returns:
            List of ZImageTextConditioning objects with embeddings and masks.
        """
        # Normalize to a list
        cond_list = [cond_field] if isinstance(cond_field, ZImageConditioningField) else cond_field

        text_conditionings: list[ZImageTextConditioning] = []
        for cond in cond_list:
            # Load the text embeddings
            cond_data = context.conditioning.load(cond.conditioning_name)
            assert len(cond_data.conditionings) == 1
            z_image_conditioning = cond_data.conditionings[0]
            assert isinstance(z_image_conditioning, ZImageConditioningInfo)
            z_image_conditioning = z_image_conditioning.to(dtype=dtype, device=device)
            prompt_embeds = z_image_conditioning.prompt_embeds

            # Load the mask, if provided
            mask: torch.Tensor | None = None
            if cond.mask is not None:
                mask = context.tensors.load(cond.mask.tensor_name)
                mask = mask.to(device=device)
                mask = ZImageRegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, img_height, img_width, dtype, device
                )

            text_conditionings.append(ZImageTextConditioning(prompt_embeds=prompt_embeds, mask=mask))

        return text_conditionings

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

        # Calculate image token grid dimensions
        patch_size = 2  # Z-Image uses patch_size=2
        latent_height = self.height // LATENT_SCALE_FACTOR
        latent_width = self.width // LATENT_SCALE_FACTOR
        img_token_height = latent_height // patch_size
        img_token_width = latent_width // patch_size
        img_seq_len = img_token_height * img_token_width

        # Load positive conditioning with regional masks
        pos_text_conditionings = self._load_text_conditioning(
            context=context,
            cond_field=self.positive_conditioning,
            img_height=img_token_height,
            img_width=img_token_width,
            dtype=inference_dtype,
            device=device,
        )

        # Create regional prompting extension
        regional_extension = ZImageRegionalPromptingExtension.from_text_conditionings(
            text_conditionings=pos_text_conditionings,
            img_seq_len=img_seq_len,
        )

        # Get the concatenated prompt embeddings for the transformer
        pos_prompt_embeds = regional_extension.regional_text_conditioning.prompt_embeds

        # Load negative conditioning if provided and guidance_scale != 1.0
        # CFG formula: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        # At cfg_scale=1.0: pred = pred_cond (no effect, skip uncond computation)
        # This matches FLUX's convention where 1.0 means "no CFG"
        neg_prompt_embeds: torch.Tensor | None = None
        do_classifier_free_guidance = (
            not math.isclose(self.guidance_scale, 1.0) and self.negative_conditioning is not None
        )
        if do_classifier_free_guidance:
            assert self.negative_conditioning is not None
            # Load all negative conditionings and concatenate embeddings
            # Note: We ignore masks for negative conditioning as regional negative prompting is not fully supported
            neg_text_conditionings = self._load_text_conditioning(
                context=context,
                cond_field=self.negative_conditioning,
                img_height=img_token_height,
                img_width=img_token_width,
                dtype=inference_dtype,
                device=device,
            )
            # Concatenate all negative embeddings
            neg_prompt_embeds = torch.cat([tc.prompt_embeds for tc in neg_text_conditionings], dim=0)

        # Calculate shift based on image sequence length
        mu = self._calculate_shift(img_seq_len)

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

        # Initialize the diffusers scheduler if not using built-in Euler
        scheduler: SchedulerMixin | None = None
        use_scheduler = self.scheduler != "euler"

        if use_scheduler:
            scheduler_class = ZIMAGE_SCHEDULER_MAP[self.scheduler]
            scheduler = scheduler_class(
                num_train_timesteps=1000,
                shift=1.0,
            )
            # Set timesteps using custom sigmas if supported, otherwise use num_inference_steps
            set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
            if "sigmas" in set_timesteps_sig.parameters:
                # Convert sigmas list to tensor for scheduler
                scheduler.set_timesteps(sigmas=sigmas, device=device)
            else:
                # Scheduler doesn't support custom sigmas - use num_inference_steps
                scheduler.set_timesteps(num_inference_steps=total_steps, device=device)

            # For Heun scheduler, the number of actual steps may differ
            num_scheduler_steps = len(scheduler.timesteps)
        else:
            num_scheduler_steps = total_steps

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

            # Load transformer - always use base transformer, control is handled via extension
            (cached_weights, transformer) = exit_stack.enter_context(transformer_info.model_on_device())

            # Prepare control extension if control is provided
            control_extension: ZImageControlNetExtension | None = None

            if self.control is not None:
                # Load control adapter using context manager (proper GPU memory management)
                control_model_info = context.models.load(self.control.control_model)
                (_, control_adapter) = exit_stack.enter_context(control_model_info.model_on_device())
                assert isinstance(control_adapter, ZImageControlAdapter)

                # Get control_in_dim from adapter config (16 for V1, 33 for V2.0)
                adapter_config = control_adapter.config
                control_in_dim = adapter_config.get("control_in_dim", 16)
                num_control_blocks = adapter_config.get("num_control_blocks", 6)

                # Log control configuration for debugging
                version = "V2.0" if control_in_dim > 16 else "V1"
                context.util.signal_progress(
                    f"Using Z-Image ControlNet {version} (Extension): control_in_dim={control_in_dim}, "
                    f"num_blocks={num_control_blocks}, scale={self.control.control_context_scale}"
                )

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

                # Add frame dimension: [B, C, H, W] -> [C, 1, H, W] (single image)
                control_latents = control_latents.squeeze(0).unsqueeze(1)

                # Prepare control_cond based on control_in_dim
                # V1: 16 channels (just control latents)
                # V2.0: 33 channels = 16 control + 16 reference + 1 mask
                #   - Channels 0-15: control image latents (from VAE encoding)
                #   - Channels 16-31: reference/inpaint image latents (zeros for pure control)
                #   - Channel 32: inpaint mask (1.0 = don't inpaint, 0.0 = inpaint region)
                # For pure control (no inpainting), we set mask=1 to tell model "use control, don't inpaint"
                c, f, h, w = control_latents.shape
                if c < control_in_dim:
                    padding_channels = control_in_dim - c
                    if padding_channels == 17:
                        # V2.0: 16 reference channels (zeros) + 1 mask channel (ones)
                        ref_padding = torch.zeros(
                            (16, f, h, w),
                            device=device,
                            dtype=inference_dtype,
                        )
                        # Mask channel = 1.0 means "don't inpaint this region, use control signal"
                        mask_channel = torch.ones(
                            (1, f, h, w),
                            device=device,
                            dtype=inference_dtype,
                        )
                        control_latents = torch.cat([control_latents, ref_padding, mask_channel], dim=0)
                    else:
                        # Generic padding with zeros for other cases
                        zero_padding = torch.zeros(
                            (padding_channels, f, h, w),
                            device=device,
                            dtype=inference_dtype,
                        )
                        control_latents = torch.cat([control_latents, zero_padding], dim=0)

                # Create control extension (adapter is already on device from model_on_device)
                control_extension = ZImageControlNetExtension(
                    control_adapter=control_adapter,
                    control_cond=control_latents,
                    weight=self.control.control_context_scale,
                    begin_step_percent=self.control.begin_step_percent,
                    end_step_percent=self.control.end_step_percent,
                )

            # Apply LoRA models to the transformer.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=Z_IMAGE_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Apply regional prompting patch if we have regional masks
            exit_stack.enter_context(
                patch_transformer_for_regional_prompting(
                    transformer=transformer,
                    regional_attn_mask=regional_extension.regional_attn_mask,
                    img_seq_len=img_seq_len,
                )
            )

            # Denoising loop - supports both built-in Euler and diffusers schedulers
            # Track user-facing step for progress (accounts for Heun's double steps)
            user_step = 0

            if use_scheduler and scheduler is not None:
                # Use diffusers scheduler for stepping
                for step_index in tqdm(range(num_scheduler_steps)):
                    sched_timestep = scheduler.timesteps[step_index]
                    # Convert scheduler timestep (0-1000) to normalized sigma (0-1)
                    sigma_curr = sched_timestep.item() / scheduler.config.num_train_timesteps

                    # For Heun scheduler, track if we're in first or second order step
                    is_heun = hasattr(scheduler, "state_in_first_order")
                    in_first_order = scheduler.state_in_first_order if is_heun else True

                    # Timestep tensor for Z-Image model
                    # The model expects t=0 at start (noise) and t=1 at end (clean)
                    model_t = 1.0 - sigma_curr
                    timestep = torch.tensor([model_t], device=device, dtype=inference_dtype).expand(latents.shape[0])

                    # Run transformer for positive prediction
                    latent_model_input = latents.to(transformer.dtype)
                    latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
                    latent_model_input_list = list(latent_model_input.unbind(dim=0))

                    # Determine if control should be applied at this step
                    apply_control = control_extension is not None and control_extension.should_apply(
                        user_step, total_steps
                    )

                    # Run forward pass
                    if apply_control:
                        model_out_list, _ = z_image_forward_with_control(
                            transformer=transformer,
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[pos_prompt_embeds],
                            control_extension=control_extension,
                        )
                    else:
                        model_output = transformer(
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[pos_prompt_embeds],
                        )
                        model_out_list = model_output[0]

                    noise_pred_cond = torch.stack([t.float() for t in model_out_list], dim=0)
                    noise_pred_cond = noise_pred_cond.squeeze(2)
                    noise_pred_cond = -noise_pred_cond  # Z-Image uses v-prediction with negation

                    # Apply CFG if enabled
                    if do_classifier_free_guidance and neg_prompt_embeds is not None:
                        if apply_control:
                            model_out_list_uncond, _ = z_image_forward_with_control(
                                transformer=transformer,
                                x=latent_model_input_list,
                                t=timestep,
                                cap_feats=[neg_prompt_embeds],
                                control_extension=control_extension,
                            )
                        else:
                            model_output_uncond = transformer(
                                x=latent_model_input_list,
                                t=timestep,
                                cap_feats=[neg_prompt_embeds],
                            )
                            model_out_list_uncond = model_output_uncond[0]

                        noise_pred_uncond = torch.stack([t.float() for t in model_out_list_uncond], dim=0)
                        noise_pred_uncond = noise_pred_uncond.squeeze(2)
                        noise_pred_uncond = -noise_pred_uncond
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    # Use scheduler.step() for the update
                    step_output = scheduler.step(model_output=noise_pred, timestep=sched_timestep, sample=latents)
                    latents = step_output.prev_sample

                    # Get sigma_prev for inpainting (next sigma value)
                    if step_index + 1 < len(scheduler.sigmas):
                        sigma_prev = scheduler.sigmas[step_index + 1].item()
                    else:
                        sigma_prev = 0.0

                    if inpaint_extension is not None:
                        latents = inpaint_extension.merge_intermediate_latents_with_init_latents(latents, sigma_prev)

                    # For Heun, only increment user step after second-order step completes
                    if is_heun:
                        if not in_first_order:
                            user_step += 1
                            # Only call step_callback if we haven't exceeded total_steps
                            if user_step <= total_steps:
                                step_callback(
                                    PipelineIntermediateState(
                                        step=user_step,
                                        order=2,
                                        total_steps=total_steps,
                                        timestep=int(sigma_curr * 1000),
                                        latents=latents,
                                    ),
                                )
                    else:
                        # For Euler, LCM and other first-order schedulers
                        user_step += 1
                        # Only call step_callback if we haven't exceeded total_steps
                        # (LCM scheduler may have more internal steps than user-facing steps)
                        if user_step <= total_steps:
                            step_callback(
                                PipelineIntermediateState(
                                    step=user_step,
                                    order=1,
                                    total_steps=total_steps,
                                    timestep=int(sigma_curr * 1000),
                                    latents=latents,
                                ),
                            )
            else:
                # Original Euler implementation (default, optimized for Z-Image)
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
                    latent_model_input = latents.to(transformer.dtype)
                    latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
                    latent_model_input_list = list(latent_model_input.unbind(dim=0))

                    # Determine if control should be applied at this step
                    apply_control = control_extension is not None and control_extension.should_apply(
                        step_idx, total_steps
                    )

                    # Run forward pass - use custom forward with control if extension is active
                    if apply_control:
                        model_out_list, _ = z_image_forward_with_control(
                            transformer=transformer,
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[pos_prompt_embeds],
                            control_extension=control_extension,
                        )
                    else:
                        model_output = transformer(
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
                        if apply_control:
                            model_out_list_uncond, _ = z_image_forward_with_control(
                                transformer=transformer,
                                x=latent_model_input_list,
                                t=timestep,
                                cap_feats=[neg_prompt_embeds],
                                control_extension=control_extension,
                            )
                        else:
                            model_output_uncond = transformer(
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
