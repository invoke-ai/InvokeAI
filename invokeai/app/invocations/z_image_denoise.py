from contextlib import ExitStack
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
    LatentsField,
    WithBoard,
    WithMetadata,
    ZImageConditioningField,
)
from invokeai.app.invocations.model import LoRAField, TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ZImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "z_image_denoise",
    title="Denoise - Z-Image",
    tags=["image", "z-image"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ZImageDenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
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
        assert len(cond_data.conditionings) == 1
        z_image_conditioning = cond_data.conditionings[0]
        assert isinstance(z_image_conditioning, ZImageConditioningInfo)
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
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

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
            assert self.negative_conditioning is not None
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
                timestep=int(sigmas[0] * 1000),
                latents=latents,
            ),
        )

        with ExitStack() as exit_stack:
            # Load transformer and apply LoRA patches
            (_, transformer) = exit_stack.enter_context(transformer_info.model_on_device())

            # Apply LoRA models to the transformer
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=Z_IMAGE_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
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
                latent_model_input = latents.to(transformer.dtype)
                latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                # Transformer returns (List[torch.Tensor], dict) - we only need the tensor list
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
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
