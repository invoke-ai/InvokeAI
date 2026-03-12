"""Anima denoising invocation.

Implements the rectified flow denoising loop for Anima models:
- CONST model type: denoised = input - output * sigma
- Fixed shift=3.0 via time_snr_shift (same formula as Flux)
- Timestep convention: timestep = sigma * 1.0 (raw sigma, NOT 1-sigma like Z-Image)
- NO v-prediction negation (unlike Z-Image)
- 3D latent space: [B, C, T, H, W] with T=1 for images
- 16 latent channels, 8x spatial compression

Key differences from Z-Image denoise:
- Anima uses fixed shift=3.0, Z-Image uses dynamic shift based on resolution
- Anima: timestep = sigma (raw), Z-Image: model_t = 1.0 - sigma
- Anima: noise_pred = model_output (CONST), Z-Image: noise_pred = -model_output (v-pred)
- Anima transformer takes (x, timesteps, context, t5xxl_ids, t5xxl_weights)
- Anima uses 3D latents directly, Z-Image converts 4D -> list of 5D
"""

import inspect
import math
from contextlib import ExitStack
from typing import Callable, Optional

import torch
import torchvision.transforms as tv_transforms
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    AnimaConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.schedulers import ANIMA_SCHEDULER_LABELS, ANIMA_SCHEDULER_MAP, ANIMA_SCHEDULER_NAME_VALUES
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import (
    RectifiedFlowInpaintExtension,
    assert_broadcastable,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice

# Anima uses 8x spatial compression (VAE downsamples by 2^3)
ANIMA_LATENT_SCALE_FACTOR = 8
# Anima uses 16 latent channels
ANIMA_LATENT_CHANNELS = 16
# Anima uses fixed shift=3.0 for the rectified flow schedule
ANIMA_SHIFT = 3.0
# Anima uses multiplier=1.0 (raw sigma values as timesteps, per ComfyUI config)
ANIMA_MULTIPLIER = 1.0


def time_snr_shift(alpha: float, t: float) -> float:
    """Apply time-SNR shift to a timestep value.

    This is the same formula used by Flux and ComfyUI's ModelSamplingDiscreteFlow.
    With alpha=3.0, this shifts the noise schedule to spend more time at higher noise levels.

    Args:
        alpha: Shift factor (3.0 for Anima).
        t: Timestep value in [0, 1].

    Returns:
        Shifted timestep value.
    """
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def inverse_time_snr_shift(alpha: float, sigma: float) -> float:
    """Recover linear t from a shifted sigma value.

    Inverse of time_snr_shift: given sigma = alpha * t / (1 + (alpha-1) * t),
    solve for t = sigma / (alpha - (alpha-1) * sigma).

    This is needed for the inpainting extension, which expects linear t values
    for gradient mask thresholding. With Anima's shift=3.0, the difference
    between shifted sigma and linear t is large (e.g. at t=0.5, sigma=0.75),
    causing overly aggressive mask thresholding if sigma is used directly.

    Args:
        alpha: Shift factor (3.0 for Anima).
        sigma: Shifted sigma value in [0, 1].

    Returns:
        Linear t value in [0, 1].
    """
    if alpha == 1.0:
        return sigma
    denominator = alpha - (alpha - 1) * sigma
    if abs(denominator) < 1e-8:
        return 1.0
    return sigma / denominator


class AnimaInpaintExtension(RectifiedFlowInpaintExtension):
    """Inpaint extension for Anima that accounts for the time-SNR shift.

    Anima uses a fixed shift=3.0 which makes sigma values significantly larger
    than the corresponding linear t values. The base RectifiedFlowInpaintExtension
    uses t_prev for both gradient mask thresholding and noise mixing, which assumes
    linear t values.

    This subclass:
    - Uses the LINEAR t for gradient mask thresholding (correct progressive reveal)
    - Uses the SHIFTED sigma for noise mixing (matches the denoiser's noise level)
    """

    def __init__(
        self,
        init_latents: torch.Tensor,
        inpaint_mask: torch.Tensor,
        noise: torch.Tensor,
        shift: float = ANIMA_SHIFT,
    ):
        assert_broadcastable(init_latents.shape, inpaint_mask.shape, noise.shape)
        self._init_latents = init_latents
        self._inpaint_mask = inpaint_mask
        self._noise = noise
        self._shift = shift

    def merge_intermediate_latents_with_init_latents(
        self, intermediate_latents: torch.Tensor, sigma_prev: float
    ) -> torch.Tensor:
        """Merge intermediate latents with init latents, correcting for Anima's shift.

        Args:
            intermediate_latents: The denoised latents at the current step.
            sigma_prev: The SHIFTED sigma value for the next step.
        """
        # Recover linear t from shifted sigma for gradient mask thresholding.
        # This ensures the gradient mask is revealed at the correct pace.
        t_prev = inverse_time_snr_shift(self._shift, sigma_prev)
        mask = self._apply_mask_gradient_adjustment(t_prev)

        # Use shifted sigma for noise mixing to match the denoiser's noise level.
        # The Euler step produces latents at noise level sigma_prev, so the
        # preserved regions must also be at sigma_prev noise level.
        noised_init_latents = self._noise * sigma_prev + (1.0 - sigma_prev) * self._init_latents

        return intermediate_latents * mask + noised_init_latents * (1.0 - mask)


@invocation(
    "anima_denoise",
    title="Denoise - Anima",
    tags=["image", "anima"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class AnimaDenoiseInvocation(BaseInvocation):
    """Run the denoising process with an Anima model.

    Uses rectified flow sampling with shift=3.0 and the Cosmos Predict2 DiT
    backbone with integrated LLM Adapter for text conditioning.

    Supports txt2img, img2img (via latents input), and inpainting (via denoise_mask).
    """

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    # denoise_mask is used for inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description="Anima transformer model.", input=Input.Connection, title="Transformer"
    )
    positive_conditioning: AnimaConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: AnimaConditioningField | None = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    guidance_scale: float = InputField(
        default=4.5,
        ge=1.0,
        description="Guidance scale for classifier-free guidance. Recommended: 4.0-5.0 for Anima.",
        title="Guidance Scale",
    )
    width: int = InputField(default=1024, multiple_of=8, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=8, description="Height of the generated image.")
    steps: int = InputField(default=30, gt=0, description="Number of denoising steps. 30 recommended for Anima.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    scheduler: ANIMA_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process.",
        ui_choice_labels=ANIMA_SCHEDULER_LABELS,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        """Prepare the inpaint mask for Anima.

        Anima uses 3D latents [B, C, T, H, W] internally but the mask operates
        on the spatial dimensions [B, C, H, W] which match the squeezed output.
        """
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

    def _get_noise(
        self,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> torch.Tensor:
        """Generate initial noise tensor in 3D latent space [B, C, T, H, W]."""
        rand_device = "cpu"
        return torch.randn(
            1,
            ANIMA_LATENT_CHANNELS,
            1,  # T=1 for single image
            height // ANIMA_LATENT_SCALE_FACTOR,
            width // ANIMA_LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=torch.float32,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to(device=device, dtype=dtype)

    def _get_sigmas(self, num_steps: int) -> list[float]:
        """Generate sigma schedule with fixed shift=3.0.

        Uses the same time_snr_shift formula as Flux/ComfyUI but with
        a fixed shift factor of 3.0 (no dynamic resolution-based shift).

        Returns:
            List of num_steps + 1 sigma values from ~1.0 (noise) to 0.0 (clean).
        """
        sigmas = []
        for i in range(num_steps + 1):
            t = 1.0 - i / num_steps
            sigma = time_snr_shift(ANIMA_SHIFT, t)
            sigmas.append(sigma)
        return sigmas

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: AnimaConditioningField,
        dtype: torch.dtype,
        device: torch.device,
    ) -> AnimaConditioningInfo:
        """Load Anima conditioning data from storage."""
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        assert len(cond_data.conditionings) == 1
        cond_info = cond_data.conditionings[0]
        assert isinstance(cond_info, AnimaConditioningInfo)
        return cond_info.to(dtype=dtype, device=device)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        transformer_info = context.models.load(self.transformer.transformer)

        # Load positive conditioning
        pos_cond = self._load_conditioning(context, self.positive_conditioning, inference_dtype, device)
        pos_qwen3_embeds = pos_cond.qwen3_embeds.unsqueeze(0)  # Add batch dim: (1, seq_len, 1024)
        pos_t5xxl_ids = pos_cond.t5xxl_ids.unsqueeze(0)  # Add batch dim: (1, seq_len)
        pos_t5xxl_weights = None
        if pos_cond.t5xxl_weights is not None:
            pos_t5xxl_weights = pos_cond.t5xxl_weights.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        # Load negative conditioning if CFG is enabled
        do_cfg = not math.isclose(self.guidance_scale, 1.0) and self.negative_conditioning is not None
        neg_qwen3_embeds = None
        neg_t5xxl_ids = None
        neg_t5xxl_weights = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_cond = self._load_conditioning(context, self.negative_conditioning, inference_dtype, device)
            neg_qwen3_embeds = neg_cond.qwen3_embeds.unsqueeze(0)
            neg_t5xxl_ids = neg_cond.t5xxl_ids.unsqueeze(0)
            if neg_cond.t5xxl_weights is not None:
                neg_t5xxl_weights = neg_cond.t5xxl_weights.unsqueeze(0).unsqueeze(-1)

        # Generate sigma schedule
        sigmas = self._get_sigmas(self.steps)

        # Apply denoising_start and denoising_end clipping (for img2img/inpaint)
        if self.denoising_start > 0 or self.denoising_end < 1:
            total_sigmas = len(sigmas)
            start_idx = int(self.denoising_start * (total_sigmas - 1))
            end_idx = int(self.denoising_end * (total_sigmas - 1)) + 1
            sigmas = sigmas[start_idx:end_idx]

        total_steps = len(sigmas) - 1

        # Load input latents if provided (image-to-image)
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)
            # Anima denoiser works in 3D: add temporal dim if needed
            if init_latents.ndim == 4:
                init_latents = init_latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        # Generate initial noise (3D latent: [B, C, T, H, W])
        noise = self._get_noise(self.height, self.width, inference_dtype, device, self.seed)

        # Prepare input latents
        if init_latents is not None:
            if self.add_noise:
                # Noise the init_latents for img2img: latents = s_0 * noise + (1 - s_0) * init_latents
                s_0 = sigmas[0]
                latents = s_0 * noise + (1.0 - s_0) * init_latents
            else:
                latents = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        if total_steps <= 0:
            return latents.squeeze(2)  # Remove temporal dim for output

        # Prepare inpaint extension (operates on squeezed 4D latents)
        # Uses AnimaInpaintExtension which corrects for the time-SNR shift:
        # - Linear t for gradient mask thresholding (correct progressive reveal)
        # - Shifted sigma for noise mixing (matches denoiser's noise level)
        inpaint_mask = self._prep_inpaint_mask(context, latents.squeeze(2))
        inpaint_extension: AnimaInpaintExtension | None = None
        if inpaint_mask is not None:
            if init_latents is None:
                raise ValueError("Initial latents are required when using an inpaint mask (image-to-image inpainting)")
            inpaint_extension = AnimaInpaintExtension(
                init_latents=init_latents.squeeze(2),
                inpaint_mask=inpaint_mask,
                noise=noise.squeeze(2),
                shift=ANIMA_SHIFT,
            )

        step_callback = self._build_step_callback(context)

        # Initialize diffusers scheduler if not using built-in Euler
        scheduler: SchedulerMixin | None = None
        use_scheduler = self.scheduler != "euler"

        if use_scheduler:
            scheduler_class = ANIMA_SCHEDULER_MAP[self.scheduler]
            scheduler = scheduler_class(num_train_timesteps=1000, shift=1.0)
            is_lcm = self.scheduler == "lcm"
            set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
            if not is_lcm and "sigmas" in set_timesteps_sig.parameters:
                scheduler.set_timesteps(sigmas=sigmas, device=device)
            else:
                scheduler.set_timesteps(num_inference_steps=total_steps, device=device)
            num_scheduler_steps = len(scheduler.timesteps)
        else:
            num_scheduler_steps = total_steps

        with ExitStack() as exit_stack:
            (cached_weights, transformer) = exit_stack.enter_context(transformer_info.model_on_device())

            if use_scheduler and scheduler is not None:
                # Scheduler-based denoising
                user_step = 0
                pbar = tqdm(total=total_steps, desc="Denoising (Anima)")
                for step_index in range(num_scheduler_steps):
                    sched_timestep = scheduler.timesteps[step_index]
                    sigma_curr = sched_timestep.item() / scheduler.config.num_train_timesteps

                    is_heun = hasattr(scheduler, "state_in_first_order")
                    in_first_order = scheduler.state_in_first_order if is_heun else True

                    # Anima timestep convention: timestep = sigma * multiplier (1.0)
                    timestep = torch.tensor(
                        [sigma_curr * ANIMA_MULTIPLIER], device=device, dtype=inference_dtype
                    ).expand(latents.shape[0])

                    # Run transformer (positive)
                    model_output = transformer(
                        x=latents.to(transformer.dtype if hasattr(transformer, 'dtype') else inference_dtype),
                        timesteps=timestep,
                        context=pos_qwen3_embeds,
                        t5xxl_ids=pos_t5xxl_ids,
                        t5xxl_weights=pos_t5xxl_weights,
                    )
                    noise_pred_cond = model_output.float()

                    # Apply CFG
                    if do_cfg and neg_qwen3_embeds is not None:
                        model_output_uncond = transformer(
                            x=latents.to(transformer.dtype if hasattr(transformer, 'dtype') else inference_dtype),
                            timesteps=timestep,
                            context=neg_qwen3_embeds,
                            t5xxl_ids=neg_t5xxl_ids,
                            t5xxl_weights=neg_t5xxl_weights,
                        )
                        noise_pred_uncond = model_output_uncond.float()
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    step_output = scheduler.step(model_output=noise_pred, timestep=sched_timestep, sample=latents)
                    latents = step_output.prev_sample

                    # Get sigma_prev for inpainting
                    if step_index + 1 < len(scheduler.sigmas):
                        sigma_prev = scheduler.sigmas[step_index + 1].item()
                    else:
                        sigma_prev = 0.0

                    if inpaint_extension is not None:
                        latents_4d = latents.squeeze(2)
                        latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(
                            latents_4d, sigma_prev
                        )
                        latents = latents_4d.unsqueeze(2)

                    if is_heun:
                        if not in_first_order:
                            user_step += 1
                            if user_step <= total_steps:
                                pbar.update(1)
                                step_callback(PipelineIntermediateState(
                                    step=user_step, order=2, total_steps=total_steps,
                                    timestep=int(sigma_curr * 1000), latents=latents.squeeze(2),
                                ))
                    else:
                        user_step += 1
                        if user_step <= total_steps:
                            pbar.update(1)
                            step_callback(PipelineIntermediateState(
                                step=user_step, order=1, total_steps=total_steps,
                                timestep=int(sigma_curr * 1000), latents=latents.squeeze(2),
                            ))
                pbar.close()
            else:
                # Built-in Euler implementation (default for Anima)
                for step_idx in tqdm(range(total_steps), desc="Denoising (Anima)"):
                    sigma_curr = sigmas[step_idx]
                    sigma_prev = sigmas[step_idx + 1]

                    # Anima timestep: sigma * multiplier (1.0 = raw sigma)
                    timestep = torch.tensor(
                        [sigma_curr * ANIMA_MULTIPLIER], device=device, dtype=inference_dtype
                    ).expand(latents.shape[0])

                    # Run transformer (positive)
                    model_output = transformer(
                        x=latents.to(transformer.dtype if hasattr(transformer, 'dtype') else inference_dtype),
                        timesteps=timestep,
                        context=pos_qwen3_embeds,
                        t5xxl_ids=pos_t5xxl_ids,
                        t5xxl_weights=pos_t5xxl_weights,
                    )

                    # CONST model: noise_pred = model_output (NO negation, unlike Z-Image v-pred)
                    noise_pred_cond = model_output.float()

                    # Apply CFG
                    if do_cfg and neg_qwen3_embeds is not None:
                        model_output_uncond = transformer(
                            x=latents.to(transformer.dtype if hasattr(transformer, 'dtype') else inference_dtype),
                            timesteps=timestep,
                            context=neg_qwen3_embeds,
                            t5xxl_ids=neg_t5xxl_ids,
                            t5xxl_weights=neg_t5xxl_weights,
                        )
                        noise_pred_uncond = model_output_uncond.float()
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    # Euler step: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * model_output
                    latents_dtype = latents.dtype
                    latents = latents.to(dtype=torch.float32)
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                    latents = latents.to(dtype=latents_dtype)

                    if inpaint_extension is not None:
                        latents_4d = latents.squeeze(2)
                        latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(
                            latents_4d, sigma_prev
                        )
                        latents = latents_4d.unsqueeze(2)

                    step_callback(
                        PipelineIntermediateState(
                            step=step_idx + 1,
                            order=1,
                            total_steps=total_steps,
                            timestep=int(sigma_curr * 1000),
                            latents=latents.squeeze(2),  # Remove temporal dim for preview
                        ),
                    )

        # Remove temporal dimension for output: [B, C, 1, H, W] -> [B, C, H, W]
        return latents.squeeze(2)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Anima)

        return step_callback
