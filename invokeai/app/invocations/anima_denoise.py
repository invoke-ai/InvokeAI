"""Anima denoising invocation.

Implements the rectified flow denoising loop for Anima models:
- Direct prediction: denoised = input - output * sigma
- Fixed shift=3.0 via loglinear_timestep_shift (Flux paper by Black Forest Labs)
- Timestep convention: timestep = sigma * 1.0 (raw sigma, NOT 1-sigma like Z-Image)
- NO v-prediction negation (unlike Z-Image)
- 3D latent space: [B, C, T, H, W] with T=1 for images
- 16 latent channels, 8x spatial compression

Key differences from Z-Image denoise:
- Anima uses fixed shift=3.0, Z-Image uses dynamic shift based on resolution
- Anima: timestep = sigma (raw), Z-Image: model_t = 1.0 - sigma
- Anima: noise_pred = model_output (direct), Z-Image: noise_pred = -model_output (v-pred)
- Anima transformer takes (x, timesteps, context, t5xxl_ids, t5xxl_weights)
- Anima uses 3D latents directly, Z-Image converts 4D -> list of 5D
"""

import math
import sys
from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
import torchvision.transforms as tv_transforms
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from invokeai.app.invocations.anima_lllite import AnimaLLLiteField
from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    AnimaConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.latent_noise import validate_noise_tensor_shape
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.anima.anima_transformer_patch import patch_anima_for_regional_prompting
from invokeai.backend.anima.conditioning_data import AnimaRegionalTextConditioning, AnimaTextConditioning
from invokeai.backend.anima.control_net_lllite import (
    AnimaControlNetLLLite,
    build_inpaint_cond_image,
    prepare_cond_image,
    prepare_mask,
)
from invokeai.backend.anima.regional_prompting import AnimaRegionalPromptingExtension
from invokeai.backend.anima.scheduler_driver import AnimaSchedulerDriver
from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_LABELS,
    ANIMA_SCHEDULER_NAME_VALUES,
    ANIMA_SHIFT,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.anima_lora_constants import ANIMA_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import (
    RectifiedFlowInpaintExtension,
    assert_broadcastable,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo, Range
from invokeai.backend.util.devices import TorchDevice

# Anima uses 8x spatial compression (VAE downsamples by 2^3)
ANIMA_LATENT_SCALE_FACTOR = 8
# Anima uses 16 latent channels
ANIMA_LATENT_CHANNELS = 16
# Anima uses raw sigma values as timesteps (no rescaling)
ANIMA_MULTIPLIER = 1.0


def loglinear_timestep_shift(alpha: float, t: float) -> float:
    """Apply log-linear timestep shift to a noise schedule value.

    This shift biases the noise schedule toward higher noise levels, as described
    in the Flux model (Black Forest Labs, 2024). With alpha > 1, the model spends
    proportionally more denoising steps at higher noise levels.

    Formula: sigma = alpha * t / (1 + (alpha - 1) * t)

    Args:
        alpha: Shift factor (3.0 for Anima, resolution-dependent for Flux).
        t: Timestep value in [0, 1].

    Returns:
        Shifted timestep value.
    """
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def inverse_loglinear_timestep_shift(alpha: float, sigma: float) -> float:
    """Recover linear t from a shifted sigma value.

    Inverse of loglinear_timestep_shift: given sigma = alpha * t / (1 + (alpha-1) * t),
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
        t_prev = inverse_loglinear_timestep_shift(self._shift, sigma_prev)
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
    version="1.8.0",
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
    noise: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.noise, input=Input.Connection
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
    positive_conditioning: AnimaConditioningField | list[AnimaConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: AnimaConditioningField | list[AnimaConditioningField] | None = InputField(
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
    # ControlNet-LLLite support (e.g. model-level inpaint conditioning, control layers)
    control_lllite: AnimaLLLiteField | list[AnimaLLLiteField] | None = InputField(
        default=None,
        description="Anima ControlNet-LLLite conditioning (e.g. inpaint adapter, control layers). Adapters are "
        "applied in a deterministic order (sorted by model key); each model may be used at most once.",
        input=Input.Connection,
    )
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

    @staticmethod
    def _normalize_control_lllite(
        control_lllite: AnimaLLLiteField | list[AnimaLLLiteField] | None,
    ) -> list[AnimaLLLiteField]:
        """Normalize the control_lllite input to a sorted list and reject duplicate models.

        The model cache returns ONE shared AnimaControlNetLLLite instance per
        model key, so two adapters using the same model in one run would share
        cond/multiplier state and clobber each other's bindings.

        The list is sorted by model key: the frontend fans adapters into a
        `collect` node whose output order follows graph node ids (random
        UUIDs), not user intent, and composition is weakly order-sensitive
        (each adapter's delta sees the perturbations of adapters applied after
        it). Sorting makes the cascade deterministic and reproducible.
        """
        if control_lllite is None:
            lllite_fields: list[AnimaLLLiteField] = []
        elif isinstance(control_lllite, AnimaLLLiteField):
            lllite_fields = [control_lllite]
        elif isinstance(control_lllite, list):
            lllite_fields = control_lllite
        else:
            raise ValueError(f"Unsupported control_lllite type: {type(control_lllite)}")

        seen_keys: set[str] = set()
        for lllite_field in lllite_fields:
            key = lllite_field.control_model.key
            if key in seen_keys:
                raise ValueError(
                    f"The Anima ControlNet-LLLite model '{lllite_field.control_model.name}' is used by more than "
                    "one control input. Each LLLite model can only be applied once per generation — remove the "
                    "duplicate, or select a different model for it."
                )
            seen_keys.add(key)
        return sorted(lllite_fields, key=lambda f: f.control_model.key)

    def _build_lllite_cond_image(
        self,
        context: InvocationContext,
        lllite_field: AnimaLLLiteField,
        lllite_model: AnimaControlNetLLLite,
        latents: torch.Tensor,
        patch_spatial: int = 2,
    ) -> torch.Tensor:
        """Build one adapter's LLLite conditioning image tensor (once per generation).

        The cond image is sized from the ACTUAL latent H/W (mirroring the DiT's
        patch padding) — see target_cond_hw in the backend module.
        """
        latent_h, latent_w = latents.shape[-2], latents.shape[-1]

        image_pil = context.images.get_pil(lllite_field.image_name, "RGB")
        rgb_01 = to_tensor(image_pil).unsqueeze(0)  # (1, 3, H, W) in [0, 1]
        rgb_pm1 = prepare_cond_image(rgb_01, latent_h, latent_w, patch_spatial)

        if lllite_model.cond_in_channels == 4:
            if lllite_field.mask_name is None:
                raise ValueError(
                    "This Anima ControlNet-LLLite adapter is an inpainting adapter (4-channel conditioning) and "
                    "requires a mask. Connect a mask (white = inpaint area) to the Anima ControlNet-LLLite node."
                )
            mask_pil = context.images.get_pil(lllite_field.mask_name, "L")
            mask_01 = to_tensor(mask_pil).unsqueeze(0)  # (1, 1, H, W) in [0, 1]
            mask_01 = prepare_mask(mask_01, latent_h, latent_w, patch_spatial)
            return build_inpaint_cond_image(rgb_pm1, mask_01, lllite_model.inpaint_masked_input)

        if lllite_model.cond_in_channels != 3:
            raise ValueError(
                f"Unsupported Anima ControlNet-LLLite adapter: expected 3 or 4 conditioning channels, got "
                f"{lllite_model.cond_in_channels}."
            )
        if lllite_field.mask_name is not None:
            context.logger.warning(
                "The selected Anima ControlNet-LLLite adapter does not use a mask (3-channel conditioning); the "
                "connected mask will be ignored."
            )
        return rgb_pm1

    @staticmethod
    def _get_lllite_multiplier(lllite_field: AnimaLLLiteField, step_index: int, total_steps: int) -> float:
        """Step-range gate for one LLLite adapter's multiplier.

        Uses the same user-facing step-index/percent convention as
        BaseControlNetExtension._get_weight.
        """
        first_step = math.floor(lllite_field.begin_step_percent * total_steps)
        last_step = math.ceil(lllite_field.end_step_percent * total_steps)

        if step_index < first_step or step_index > last_step:
            return 0.0

        return lllite_field.weight

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

    @staticmethod
    def _compute_img_token_grid(height: int, width: int, patch_size: int = 2) -> tuple[int, int]:
        """Compute the (height, width) of the image token grid for regional prompting.

        Anima uses 8x VAE compression and a 2x DiT patch size (16x total). The mask's
        token grid must match the token grid the transformer actually produces.

        Ceiling division mirrors the transformer's MiniTrainDIT._pad_to_patch_size, which
        pads the latent H/W up to a multiple of patch_size BEFORE patchifying. An odd latent
        dimension (e.g. 1080 // 8 = 135) is padded up (135 -> 136 -> 68 tokens); floor
        division here would size the mask for one fewer token row/column than the transformer
        produces, causing a cross-attention mask shape mismatch during regional prompting.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.
            patch_size: DiT spatial patch size (patch_spatial).

        Returns:
            Tuple of (img_token_height, img_token_width).
        """
        latent_height = height // ANIMA_LATENT_SCALE_FACTOR
        latent_width = width // ANIMA_LATENT_SCALE_FACTOR
        img_token_height = math.ceil(latent_height / patch_size)
        img_token_width = math.ceil(latent_width / patch_size)
        return img_token_height, img_token_width

    def _get_sigmas(self, num_steps: int) -> list[float]:
        """Generate sigma schedule with fixed shift=3.0.

        Uses the log-linear timestep shift from the Flux model (Black Forest Labs)
        with a fixed shift factor of 3.0 (no dynamic resolution-based shift).

        Returns:
            List of num_steps + 1 sigma values from ~1.0 (noise) to 0.0 (clean).
        """
        sigmas = []
        for i in range(num_steps + 1):
            t = 1.0 - i / num_steps
            sigma = loglinear_timestep_shift(ANIMA_SHIFT, t)
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

    def _load_text_conditionings(
        self,
        context: InvocationContext,
        cond_field: AnimaConditioningField | list[AnimaConditioningField],
        img_token_height: int,
        img_token_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[AnimaTextConditioning]:
        """Load Anima text conditioning with optional regional masks.

        Args:
            context: The invocation context.
            cond_field: Single conditioning field or list of fields.
            img_token_height: Height of the image token grid (H // patch_size).
            img_token_width: Width of the image token grid (W // patch_size).
            dtype: Target dtype.
            device: Target device.

        Returns:
            List of AnimaTextConditioning objects with optional masks.
        """
        cond_list = cond_field if isinstance(cond_field, list) else [cond_field]

        text_conditionings: list[AnimaTextConditioning] = []
        for cond in cond_list:
            cond_info = self._load_conditioning(context, cond, dtype, device)

            # Load the mask, if provided
            mask: torch.Tensor | None = None
            if cond.mask is not None:
                mask = context.tensors.load(cond.mask.tensor_name)
                mask = mask.to(device=device)
                mask = AnimaRegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, img_token_height, img_token_width, dtype, device
                )

            text_conditionings.append(
                AnimaTextConditioning(
                    qwen3_embeds=cond_info.qwen3_embeds,
                    t5xxl_ids=cond_info.t5xxl_ids,
                    t5xxl_weights=cond_info.t5xxl_weights,
                    mask=mask,
                )
            )

        return text_conditionings

    def _run_llm_adapter_for_regions(
        self,
        transformer,
        text_conditionings: list[AnimaTextConditioning],
        dtype: torch.dtype,
    ) -> AnimaRegionalTextConditioning:
        """Run the LLM Adapter separately for each regional conditioning and concatenate.

        Args:
            transformer: The AnimaTransformer instance (must be on device).
            text_conditionings: List of per-region conditioning data.
            dtype: Inference dtype.

        Returns:
            AnimaRegionalTextConditioning with concatenated context and masks.
        """
        context_embeds_list: list[torch.Tensor] = []
        context_ranges: list[Range] = []
        image_masks: list[torch.Tensor | None] = []
        cur_len = 0

        for tc in text_conditionings:
            qwen3_embeds = tc.qwen3_embeds.unsqueeze(0)  # (1, seq_len, 1024)
            t5xxl_ids = tc.t5xxl_ids.unsqueeze(0)  # (1, seq_len)
            t5xxl_weights = None
            if tc.t5xxl_weights is not None:
                t5xxl_weights = tc.t5xxl_weights.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

            # Run the LLM Adapter to produce context for this region
            context = transformer.preprocess_text_embeds(
                qwen3_embeds.to(dtype=dtype),
                t5xxl_ids,
                t5xxl_weights=t5xxl_weights.to(dtype=dtype) if t5xxl_weights is not None else None,
            )
            # context shape: (1, 512, 1024) — squeeze batch dim
            context_2d = context.squeeze(0)  # (512, 1024)

            context_embeds_list.append(context_2d)
            context_ranges.append(Range(start=cur_len, end=cur_len + context_2d.shape[0]))
            image_masks.append(tc.mask)
            cur_len += context_2d.shape[0]

        concatenated_context = torch.cat(context_embeds_list, dim=0)

        return AnimaRegionalTextConditioning(
            context_embeds=concatenated_context,
            image_masks=image_masks,
            context_ranges=context_ranges,
        )

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_anima_inference_dtype(device)

        if self.denoising_start >= self.denoising_end:
            raise ValueError(
                f"denoising_start ({self.denoising_start}) must be less than denoising_end ({self.denoising_end})."
            )

        lllite_fields = self._normalize_control_lllite(self.control_lllite)

        transformer_info = context.models.load(self.transformer.transformer)

        # Compute image token grid dimensions for regional prompting
        img_token_height, img_token_width = self._compute_img_token_grid(self.height, self.width)
        img_seq_len = img_token_height * img_token_width

        # Load positive conditioning with optional regional masks
        pos_text_conditionings = self._load_text_conditionings(
            context=context,
            cond_field=self.positive_conditioning,
            img_token_height=img_token_height,
            img_token_width=img_token_width,
            dtype=inference_dtype,
            device=device,
        )
        has_regional = len(pos_text_conditionings) > 1 or any(tc.mask is not None for tc in pos_text_conditionings)

        # Load negative conditioning if CFG is enabled
        do_cfg = not math.isclose(self.guidance_scale, 1.0) and self.negative_conditioning is not None
        neg_text_conditionings: list[AnimaTextConditioning] | None = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_text_conditionings = self._load_text_conditionings(
                context=context,
                cond_field=self.negative_conditioning,
                img_token_height=img_token_height,
                img_token_width=img_token_width,
                dtype=inference_dtype,
                device=device,
            )

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

        # Generate initial noise (3D latent: [B, C, T, H, W]).
        # If noise will never be consumed, avoid validating/loading it.
        should_ignore_noise = init_latents is not None and not self.add_noise and self.denoise_mask is None
        noise: torch.Tensor | None
        if should_ignore_noise:
            noise = None
        else:
            noise = self._prepare_noise_tensor(context, inference_dtype, device)

        # Prepare input latents
        if init_latents is not None:
            if self.add_noise:
                assert noise is not None
                # Noise the init latents using the first sigma from the clipped
                # InvokeAI schedule.
                #
                # Known limitation: if the selected scheduler later starts from a
                # different first effective sigma/timestep than sigmas[0], the
                # img2img preblend below may not match that scheduler exactly.
                # This is an existing pipeline limitation and affects both
                # internally generated noise and externally supplied noise.
                s_0 = sigmas[0]
                latents = s_0 * noise + (1.0 - s_0) * init_latents
            else:
                latents = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            assert noise is not None
            latents = noise

        if total_steps <= 0:
            return latents.squeeze(2)

        # Prepare inpaint extension
        inpaint_mask = self._prep_inpaint_mask(context, latents.squeeze(2))
        inpaint_extension: AnimaInpaintExtension | None = None
        if inpaint_mask is not None:
            if init_latents is None:
                raise ValueError("Initial latents are required when using an inpaint mask (image-to-image inpainting)")
            assert noise is not None
            inpaint_extension = AnimaInpaintExtension(
                init_latents=init_latents.squeeze(2),
                inpaint_mask=inpaint_mask,
                noise=noise.squeeze(2),
                shift=ANIMA_SHIFT,
            )

        step_callback = self._build_step_callback(context)

        # Initialize scheduler driver if not using built-in Euler.
        use_scheduler = self.scheduler != "euler"
        driver: AnimaSchedulerDriver | None = None
        if use_scheduler:
            driver = AnimaSchedulerDriver(
                scheduler_name=self.scheduler,
                sigmas=sigmas,
                steps=self.steps,
                denoising_start=self.denoising_start,
                denoising_end=self.denoising_end,
                device=device,
                seed=self.seed,
            )

        with ExitStack() as exit_stack:
            # On Windows + CUDA, opt into cuDNN's flash-attention kernel. torch on
            # Windows does not ship the native flash-attention backend, so SDPA falls
            # back to the slower memory-efficient kernel; cuDNN's kernel is ~1.7x faster
            # for Anima's attention shapes (~6% per denoising step at 1024x1024). The
            # cuDNN SDP backend sits low in torch's default priority and is never
            # selected, so we opt in explicitly.
            #
            # This is a priority *list*, not a forced kernel: torch picks the first
            # eligible backend per call, and MATH is always eligible, so it degrades
            # safely on any GPU/dtype where cuDNN is unavailable. Scoped to Windows +
            # CUDA — elsewhere torch's default backend is already fast.
            if device.type == "cuda" and sys.platform == "win32":
                exit_stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.CUDNN_ATTENTION,
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.MATH,
                        ],
                        set_priority=True,
                    )
                )
            (cached_weights, transformer) = exit_stack.enter_context(transformer_info.model_on_device())

            # Prepare the ControlNet-LLLite adapters if provided. Each adapter's
            # conditioning image is built ONCE per generation (not per step).
            lllite_adapters: list[tuple[AnimaLLLiteField, AnimaControlNetLLLite, torch.Tensor]] = []
            for lllite_field in lllite_fields:
                lllite_info = context.models.load(lllite_field.control_model)
                (_, lllite_model) = exit_stack.enter_context(lllite_info.model_on_device())
                assert isinstance(lllite_model, AnimaControlNetLLLite)
                lllite_cond = self._build_lllite_cond_image(
                    context,
                    lllite_field,
                    lllite_model,
                    latents,
                    patch_spatial=int(getattr(transformer, "patch_spatial", 2)),
                )
                lllite_adapters.append((lllite_field, lllite_model, lllite_cond))

            # Apply LoRA models to the transformer.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=ANIMA_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                )
            )

            # Run LLM Adapter for each regional conditioning to produce context vectors.
            # This must happen with the transformer on device since it uses the adapter weights.
            if has_regional:
                pos_regional = self._run_llm_adapter_for_regions(transformer, pos_text_conditionings, inference_dtype)
                pos_context = pos_regional.context_embeds.unsqueeze(0)  # (1, total_ctx_len, 1024)

                # Build regional prompting extension with cross-attention mask
                regional_extension = AnimaRegionalPromptingExtension.from_regional_conditioning(
                    pos_regional, img_seq_len
                )

                # For negative, concatenate all regions without masking (matches Z-Image behavior)
                neg_context = None
                if do_cfg and neg_text_conditionings is not None:
                    neg_regional = self._run_llm_adapter_for_regions(
                        transformer, neg_text_conditionings, inference_dtype
                    )
                    neg_context = neg_regional.context_embeds.unsqueeze(0)
            else:
                # Single conditioning — run LLM Adapter via normal forward path
                tc = pos_text_conditionings[0]
                pos_qwen3_embeds = tc.qwen3_embeds.unsqueeze(0)
                pos_t5xxl_ids = tc.t5xxl_ids.unsqueeze(0)
                pos_t5xxl_weights = None
                if tc.t5xxl_weights is not None:
                    pos_t5xxl_weights = tc.t5xxl_weights.unsqueeze(0).unsqueeze(-1)

                # Pre-compute context via LLM Adapter
                pos_context = transformer.preprocess_text_embeds(
                    pos_qwen3_embeds.to(dtype=inference_dtype),
                    pos_t5xxl_ids,
                    t5xxl_weights=pos_t5xxl_weights.to(dtype=inference_dtype)
                    if pos_t5xxl_weights is not None
                    else None,
                )

                neg_context = None
                if do_cfg and neg_text_conditionings is not None:
                    ntc = neg_text_conditionings[0]
                    neg_qwen3 = ntc.qwen3_embeds.unsqueeze(0)
                    neg_ids = ntc.t5xxl_ids.unsqueeze(0)
                    neg_weights = None
                    if ntc.t5xxl_weights is not None:
                        neg_weights = ntc.t5xxl_weights.unsqueeze(0).unsqueeze(-1)
                    neg_context = transformer.preprocess_text_embeds(
                        neg_qwen3.to(dtype=inference_dtype),
                        neg_ids,
                        t5xxl_weights=neg_weights.to(dtype=inference_dtype) if neg_weights is not None else None,
                    )

                regional_extension = None

            # Apply regional prompting patch if we have regional masks
            exit_stack.enter_context(patch_anima_for_regional_prompting(transformer, regional_extension))

            # Helper to run transformer with pre-computed context (bypasses LLM Adapter)
            def _run_transformer(ctx: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return transformer(
                    x=x.to(transformer.dtype if hasattr(transformer, "dtype") else inference_dtype),
                    timesteps=t,
                    context=ctx,
                    # t5xxl_ids=None skips the LLM Adapter — context is already pre-computed
                )

            try:
                # Bind AFTER LoRA patching so the LLLite modules wrap the patched
                # forwards. List order = apply order; restore must be the reverse.
                for _, lllite_model, lllite_cond in lllite_adapters:
                    lllite_model.apply_to(transformer)
                    lllite_model.set_cond_image(lllite_cond)

                if driver is not None:
                    user_step = 0
                    pbar = tqdm(total=total_steps, desc="Denoising (Anima)")
                    for it in driver.iterations():
                        # Gate on the user-facing step index so both halves of a
                        # multi-pass step (e.g. Heun pairs) share one gate value.
                        for lllite_field, lllite_model, _ in lllite_adapters:
                            lllite_model.set_multiplier(
                                self._get_lllite_multiplier(lllite_field, user_step, total_steps)
                            )

                        timestep = torch.tensor(
                            [it.sigma_curr * ANIMA_MULTIPLIER], device=device, dtype=inference_dtype
                        ).expand(latents.shape[0])

                        noise_pred_cond = _run_transformer(pos_context, latents, timestep).float()

                        if do_cfg and neg_context is not None:
                            noise_pred_uncond = _run_transformer(neg_context, latents, timestep).float()
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_cond

                        latents_preview = self._estimate_preview_latents(
                            latents=latents,
                            sigma=it.sigma_curr,
                            noise_pred=noise_pred,
                        )

                        latents = driver.step(model_output=noise_pred, timestep=it.sched_timestep, sample=latents)

                        if it.completes_user_step:
                            # RectifiedFlowInpaintExtension expects this once per user step (its
                            # docstring), so for Heun we skip the FO half of each pair to avoid
                            # corrupting the second-order corrector's input.
                            if inpaint_extension is not None:
                                latents_4d = latents.squeeze(2)
                                latents_4d = inpaint_extension.merge_intermediate_latents_with_init_latents(
                                    latents_4d, it.sigma_prev
                                )
                                latents = latents_4d.unsqueeze(2)

                            user_step += 1
                            pbar.update(1)
                            step_callback(
                                PipelineIntermediateState(
                                    step=user_step,
                                    order=it.order,
                                    total_steps=total_steps,
                                    timestep=int(it.sigma_curr * 1000),
                                    latents=latents_preview.squeeze(2),
                                )
                            )
                    pbar.close()
                else:
                    # Built-in Euler implementation (default for Anima)
                    for step_idx in tqdm(range(total_steps), desc="Denoising (Anima)"):
                        for lllite_field, lllite_model, _ in lllite_adapters:
                            lllite_model.set_multiplier(
                                self._get_lllite_multiplier(lllite_field, step_idx, total_steps)
                            )

                        sigma_curr = sigmas[step_idx]
                        sigma_prev = sigmas[step_idx + 1]

                        timestep = torch.tensor(
                            [sigma_curr * ANIMA_MULTIPLIER], device=device, dtype=inference_dtype
                        ).expand(latents.shape[0])

                        noise_pred_cond = _run_transformer(pos_context, latents, timestep).float()

                        if do_cfg and neg_context is not None:
                            noise_pred_uncond = _run_transformer(neg_context, latents, timestep).float()
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_cond

                        latents_dtype = latents.dtype
                        latents = latents.to(dtype=torch.float32)
                        latents = latents + (sigma_prev - sigma_curr) * noise_pred
                        latents = latents.to(dtype=latents_dtype)
                        latents_preview = self._estimate_preview_latents(
                            latents=latents, sigma=sigma_prev, noise_pred=noise_pred
                        )

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
                                latents=latents_preview.squeeze(2),
                            ),
                        )
            finally:
                # The adapter models are shared via the model cache — always undo
                # the forward swaps and drop the per-run cond state. unbind() is
                # only correct LIFO, so restore in REVERSE apply order (see
                # AnimaControlNetLLLite.restore). Each restore is isolated so one
                # failure cannot leave the remaining adapters bound to the
                # cache-shared transformer.
                for lllite_field, lllite_model, _ in reversed(lllite_adapters):
                    try:
                        lllite_model.restore()
                        lllite_model.clear_cond_image()
                    except Exception as e:
                        context.logger.error(
                            f"Failed to restore Anima ControlNet-LLLite adapter "
                            f"'{lllite_field.control_model.name}': {e}"
                        )

        # Remove temporal dimension for output: [B, C, 1, H, W] -> [B, C, H, W]
        return latents.squeeze(2)

    def _prepare_noise_tensor(
        self, context: InvocationContext, inference_dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self.noise is not None:
            noise = context.tensors.load(self.noise.latents_name).to(device=device, dtype=inference_dtype)
            validate_noise_tensor_shape(noise, "Anima", self.width, self.height)
            return noise

        return self._get_noise(self.height, self.width, inference_dtype, device, self.seed)

    def _estimate_preview_latents(self, latents: torch.Tensor, sigma: float, noise_pred: torch.Tensor) -> torch.Tensor:
        latents_dtype = latents.dtype
        latents_fp32 = latents.to(dtype=torch.float32)
        preview = latents_fp32 - sigma * noise_pred.to(dtype=torch.float32)
        return preview.to(dtype=latents_dtype)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.Anima)

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
