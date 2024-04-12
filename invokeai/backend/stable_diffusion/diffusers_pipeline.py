from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import einops
import PIL.Image
import psutil
import torch
import torchvision.transforms as T
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.import_utils import is_xformers_available
from pydantic import Field
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    IPAdapterData,
    TextConditioningData,
)
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher
from invokeai.backend.util.attention import auto_detect_slice_size
from invokeai.backend.util.devices import normalize_device


@dataclass
class PipelineIntermediateState:
    step: int
    order: int
    total_steps: int
    timestep: int
    latents: torch.Tensor
    predicted_original: Optional[torch.Tensor] = None


@dataclass
class AddsMaskLatents:
    """Add the channels required for inpainting model input.

    The inpainting model takes the normal latent channels as input, _plus_ a one-channel mask
    and the latent encoding of the base image.

    This class assumes the same mask and base image should apply to all items in the batch.
    """

    forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    mask: torch.Tensor
    initial_image_latents: torch.Tensor

    def __call__(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        model_input = self.add_mask_channels(latents)
        return self.forward(model_input, t, text_embeddings, **kwargs)

    def add_mask_channels(self, latents):
        batch_size = latents.size(0)
        # duplicate mask and latents for each batch
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        image_latents = einops.repeat(self.initial_image_latents, "b c h w -> (repeat b) c h w", repeat=batch_size)
        # add mask and image as additional channels
        model_input, _ = einops.pack([latents, mask, image_latents], "b * h w")
        return model_input


def are_like_tensors(a: torch.Tensor, b: object) -> bool:
    return isinstance(b, torch.Tensor) and (a.size() == b.size())


@dataclass
class AddsMaskGuidance:
    orig_latents: torch.Tensor
    mask: torch.Tensor # 0 is masked, 1 is unmasked
    masked_latents: torch.Tensor | None
    scheduler: SchedulerMixin
    noise: torch.Tensor | None
    gradient_mask: bool
    unet_type: str
    inpaint_model: bool
    seed: int

    def __post_init__(self):
        """Align internal data and create noise if necessary"""
        self.mask = tv_resize(self.mask, self.orig_latents.shape[-2:])
        self.mask = self.mask.to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)
        if self.noise is None:
            self.noise = torch.randn(
                self.orig_latents.shape,
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(self.seed),
            ).to(device=self.orig_latents.device, dtype=self.orig_latents.dtype)

    def mask_from_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Create a mask based on the current timestep"""
        if self.inpaint_model:
            mask_bool = self.mask < 1
            floored_mask = torch.where(mask_bool, 0, 1)
            return floored_mask
        elif self.gradient_mask:
            threshhold = (t.item()) / self.scheduler.config.num_train_timesteps
            mask_bool = self.mask < 1 - threshhold
            timestep_mask = torch.where(mask_bool, 0, 1)
            return timestep_mask.to(device=self.mask.device)
        else:
            print("normal mask used")
            return self.mask.clone()

    def modify_latents_before_scaling(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Replace unmasked region with original latents. Called before the scheduler scales the latent values."""
        if self.inpaint_model:
            return latents # skip this stage

        #expand to match batch size if necessary
        batch_size = latents.size(0)
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            t = einops.repeat(t, "-> batch", batch=batch_size)

        # create noised version of the original latents
        noised_latents = self.scheduler.add_noise(self.orig_latents, self.noise, t)
        noised_latents = einops.repeat(noised_latents, "b c h w -> (repeat b) c h w", repeat=batch_size).to(device=latents.device, dtype=latents.dtype)
        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        masked_input = torch.lerp(latents, noised_latents, mask)
        return masked_input

    def shrink_mask(self, mask: torch.Tensor, n_operations: int) -> torch.Tensor:
        kernel = torch.ones(1, 1, 3, 3).to(device=mask.device, dtype=mask.dtype)
        for _ in range(n_operations):
            mask = torch.nn.functional.conv2d(mask, kernel, padding=1).clamp(0, 1)
        return mask

    def modify_latents_before_noise_prediction(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Expand latents with information needed by inpaint model"""
        if not self.inpaint_model:
            return latents # skip this stage

        mask = self.mask_from_timestep(t).to(device=latents.device, dtype=latents.dtype)
        if self.masked_latents is None:
            #latent values for a black region after VAE encode
            if self.unet_type == "sd-1":
                latent_zeros = [0.78857421875, -0.638671875, 0.576171875, 0.12213134765625]
            elif self.unet_type == "sd-2":
                latent_zeros = [0.7890625, -0.638671875, 0.576171875, 0.12213134765625]
                print("WARNING: SD-2 Inpaint Models are not yet supported")
            elif self.unet_type == "sdxl":
                latent_zeros = [-0.578125, 0.501953125, 0.59326171875, -0.393798828125]
            else:
                raise ValueError(f"Unet type {self.unet_type} not supported as an inpaint model. Where did you get this?")

            # replace masked region with specified values
            mask_values = torch.tensor(latent_zeros).view(1, 4, 1, 1).expand_as(latents).to(device=latents.device, dtype=latents.dtype)
            small_mask = self.shrink_mask(mask, 1) #make the synthetic mask fill in the masked_latents smaller than the mask channel
            masked_latents = self.scheduler.scale_model_input(torch.where(small_mask == 0, mask_values, self.orig_latents), t)
        else:
            masked_latents = self.scheduler.scale_model_input(self.masked_latents,t)


        masked_latents = einops.repeat(masked_latents, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        model_input = torch.cat([latents, 1 - mask, masked_latents], dim=1).to(dtype=latents.dtype, device=latents.device)
        return model_input

    def modify_result_before_callback(self, step_output, t) -> torch.Tensor:
        """Fix preview images to show the original image in the unmasked region"""
        if hasattr(step_output, "denoised"): #LCM Sampler
            prediction = step_output.denoised
        elif hasattr(step_output, "pred_original_sample"): #Samplers with final predictions
            prediction = step_output.pred_original_sample
        else: #all other samplers (no prediction available)
            prediction = step_output.prev_sample

        mask = self.mask_from_timestep(t)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=prediction.size(0))
        step_output.pred_original_sample = torch.lerp(prediction, self.orig_latents.to(dtype=prediction.dtype), mask.to(dtype=prediction.dtype))

        return step_output

    def modify_latents_after_denoising(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply original unmasked to denoised latents"""
        if self.inpaint_model:
            if self.masked_latents is None:
                mask = self.shrink_mask(self.mask, 1)
            else:
                return latents
        else:
            mask = self.mask_from_timestep(torch.Tensor([0]))
        mask = einops.repeat(mask, "b c h w -> (repeat b) c h w", repeat=latents.size(0))
        latents = torch.lerp(latents, self.orig_latents.to(dtype=latents.dtype), mask.to(dtype=latents.dtype)).to(device=latents.device)
        return latents


def trim_to_multiple_of(*args, multiple_of=8):
    return tuple((x - x % multiple_of) for x in args)


def image_resized_to_grid_as_tensor(image: PIL.Image.Image, normalize: bool = True, multiple_of=8) -> torch.FloatTensor:
    """

    :param image: input image
    :param normalize: scale the range to [-1, 1] instead of [0, 1]
    :param multiple_of: resize the input so both dimensions are a multiple of this
    """
    w, h = trim_to_multiple_of(*image.size, multiple_of=multiple_of)
    transformation = T.Compose(
        [
            T.Resize((h, w), T.InterpolationMode.LANCZOS, antialias=True),
            T.ToTensor(),
        ]
    )
    tensor = transformation(image)
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def is_inpainting_model(unet: UNet2DConditionModel):
    return unet.conv_in.in_channels == 9


@dataclass
class ControlNetData:
    model: ControlNetModel = Field(default=None)
    image_tensor: torch.Tensor = Field(default=None)
    weight: Union[float, List[float]] = Field(default=1.0)
    begin_step_percent: float = Field(default=0.0)
    end_step_percent: float = Field(default=1.0)
    control_mode: str = Field(default="balanced")
    resize_mode: str = Field(default="just_resize")


@dataclass
class T2IAdapterData:
    """A structure containing the information required to apply conditioning from a single T2I-Adapter model."""

    adapter_state: dict[torch.Tensor] = Field()
    weight: Union[float, list[float]] = Field(default=1.0)
    begin_step_percent: float = Field(default=0.0)
    end_step_percent: float = Field(default=1.0)


class StableDiffusionGeneratorPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Implementation note: This class started as a refactored copy of diffusers.StableDiffusionPipeline.
    Hopefully future versions of diffusers provide access to more of these functions so that we don't
    need to duplicate them here: https://github.com/huggingface/diffusers/issues/551#issuecomment-1281508384

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: Optional[CLIPFeatureExtractor],
        requires_safety_checker: bool = False,
        control_model: ControlNetModel = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        self.invokeai_diffuser = InvokeAIDiffuserComponent(self.unet, self._unet_forward)
        self.control_model = control_model
        self.use_ip_adapter = False

    def _adjust_memory_efficient_attention(self, latents: torch.Tensor):
        """
        if xformers is available, use it, otherwise use sliced attention.
        """
        config = get_config()
        if config.attention_type == "xformers":
            self.enable_xformers_memory_efficient_attention()
            return
        elif config.attention_type == "sliced":
            slice_size = config.attention_slice_size
            if slice_size == "auto":
                slice_size = auto_detect_slice_size(latents)
            elif slice_size == "balanced":
                slice_size = "auto"
            self.enable_attention_slicing(slice_size=slice_size)
            return
        elif config.attention_type == "normal":
            self.disable_attention_slicing()
            return
        elif config.attention_type == "torch-sdp":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # diffusers enables sdp automatically
                return
            else:
                raise Exception("torch-sdp attention slicing not available")

        # the remainder if this code is called when attention_type=='auto'
        if self.unet.device.type == "cuda":
            if is_xformers_available():
                self.enable_xformers_memory_efficient_attention()
                return
            elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # diffusers enables sdp automatically
                return

        if self.unet.device.type == "cpu" or self.unet.device.type == "mps":
            mem_free = psutil.virtual_memory().free
        elif self.unet.device.type == "cuda":
            mem_free, _ = torch.cuda.mem_get_info(normalize_device(self.unet.device))
        else:
            raise ValueError(f"unrecognized device {self.unet.device}")
        # input tensor of [1, 4, h/8, w/8]
        # output tensor of [16, (h/8 * w/8), (h/8 * w/8)]
        bytes_per_element_needed_for_baddbmm_duplication = latents.element_size() + 4
        max_size_required_for_baddbmm = (
            16
            * latents.size(dim=2)
            * latents.size(dim=3)
            * latents.size(dim=2)
            * latents.size(dim=3)
            * bytes_per_element_needed_for_baddbmm_duplication
        )
        if max_size_required_for_baddbmm > (mem_free * 3.0 / 4.0):  # 3.3 / 4.0 is from old Invoke code
            self.enable_attention_slicing(slice_size="max")
        elif torch.backends.mps.is_available():
            # diffusers recommends always enabling for mps
            self.enable_attention_slicing(slice_size="max")
        else:
            self.disable_attention_slicing()

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings=False):
        raise Exception("Should not be called")

    def latents_from_embeddings(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        scheduler_step_kwargs: dict[str, Any],
        conditioning_data: TextConditioningData,
        *,
        noise: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        additional_guidance: List[Callable] = None,
        callback: Callable[[PipelineIntermediateState], None] = None,
        control_data: List[ControlNetData] = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        mask: Optional[torch.Tensor] = None,
        masked_latents: Optional[torch.Tensor] = None,
        gradient_mask: Optional[bool] = False,
        seed: int,
    ) -> torch.Tensor:
        if init_timestep.shape[0] == 0:
            return latents

        if additional_guidance is None:
            additional_guidance = []

        orig_latents = latents.clone()

        batch_size = latents.shape[0]
        batched_t = init_timestep.expand(batch_size)

        if noise is not None:
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            latents = self.scheduler.add_noise(latents, noise, batched_t)

        try:
            latents = self.generate_latents_from_embeddings(
                latents,
                timesteps,
                conditioning_data,
                scheduler_step_kwargs=scheduler_step_kwargs,
                additional_guidance=additional_guidance,
                control_data=control_data,
                ip_adapter_data=ip_adapter_data,
                t2i_adapter_data=t2i_adapter_data,
                callback=callback,
            )
        finally:
            self.invokeai_diffuser.model_forward_callback = self._unet_forward

        # restore unmasked part after the last step is completed
        # in-process masking happens before each step
        for guidance in additional_guidance:
            latents = guidance.modify_latents_after_denoising(latents)

        return latents

    def generate_latents_from_embeddings(
        self,
        latents: torch.Tensor,
        timesteps,
        conditioning_data: TextConditioningData,
        scheduler_step_kwargs: dict[str, Any],
        *,
        additional_guidance: List[Callable] = None,
        control_data: List[ControlNetData] = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        callback: Callable[[PipelineIntermediateState], None] = None,
    ) -> torch.Tensor:
        self._adjust_memory_efficient_attention(latents)
        if additional_guidance is None:
            additional_guidance = []

        batch_size = latents.shape[0]

        if timesteps.shape[0] == 0:
            return latents

        use_ip_adapter = ip_adapter_data is not None
        use_regional_prompting = (
            conditioning_data.cond_regions is not None or conditioning_data.uncond_regions is not None
        )
        unet_attention_patcher = None
        self.use_ip_adapter = use_ip_adapter
        attn_ctx = nullcontext()
        if use_ip_adapter or use_regional_prompting:
            ip_adapters = [ipa.ip_adapter_model for ipa in ip_adapter_data] if use_ip_adapter else None
            unet_attention_patcher = UNetAttentionPatcher(ip_adapters)
            attn_ctx = unet_attention_patcher.apply_ip_adapter_attention(self.invokeai_diffuser.model)

        with attn_ctx:
            if callback is not None:
                callback(
                    PipelineIntermediateState(
                        step=-1,
                        order=self.scheduler.order,
                        total_steps=len(timesteps),
                        timestep=self.scheduler.config.num_train_timesteps,
                        latents=latents,
                    )
                )

            # print("timesteps:", timesteps)
            for i, t in enumerate(self.progress_bar(timesteps)):
                batched_t = t.expand(batch_size)
                step_output = self.step(
                    batched_t,
                    latents,
                    conditioning_data,
                    step_index=i,
                    total_step_count=len(timesteps),
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    additional_guidance=additional_guidance,
                    control_data=control_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                )

                for guidance in additional_guidance: #fix preview images to show original image in unmasked region
                    step_output = guidance.modify_result_before_callback(step_output, t)

                latents = step_output.prev_sample
                predicted_original = getattr(step_output, "pred_original_sample", None)

                if callback is not None:
                    callback(
                        PipelineIntermediateState(
                            step=i,
                            order=self.scheduler.order,
                            total_steps=len(timesteps),
                            timestep=int(t),
                            latents=latents,
                            predicted_original=predicted_original,
                        )
                    )

            return latents

    @torch.inference_mode()
    def step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: TextConditioningData,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
        additional_guidance: List[Callable] = None,
        control_data: List[ControlNetData] = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]
        if additional_guidance is None:
            additional_guidance = []

        for guidance in additional_guidance: #apply denoise mask based on unscaled input latents
            latents = guidance.modify_latents_before_scaling(latents, timestep)

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )

        # Handle T2I-Adapter(s)
        down_intrablock_additional_residuals = None
        if t2i_adapter_data is not None:
            accum_adapter_state = None
            for single_t2i_adapter_data in t2i_adapter_data:
                # Determine the T2I-Adapter weights for the current denoising step.
                first_t2i_adapter_step = math.floor(single_t2i_adapter_data.begin_step_percent * total_step_count)
                last_t2i_adapter_step = math.ceil(single_t2i_adapter_data.end_step_percent * total_step_count)
                t2i_adapter_weight = (
                    single_t2i_adapter_data.weight[step_index]
                    if isinstance(single_t2i_adapter_data.weight, list)
                    else single_t2i_adapter_data.weight
                )
                if step_index < first_t2i_adapter_step or step_index > last_t2i_adapter_step:
                    # If the current step is outside of the T2I-Adapter's begin/end step range, then set its weight to 0
                    # so it has no effect.
                    t2i_adapter_weight = 0.0

                # Apply the t2i_adapter_weight, and accumulate.
                if accum_adapter_state is None:
                    # Handle the first T2I-Adapter.
                    accum_adapter_state = [val * t2i_adapter_weight for val in single_t2i_adapter_data.adapter_state]
                else:
                    # Add to the previous adapter states.
                    for idx, value in enumerate(single_t2i_adapter_data.adapter_state):
                        accum_adapter_state[idx] += value * t2i_adapter_weight

            down_intrablock_additional_residuals = accum_adapter_state

        for guidance in additional_guidance: #add mask channels for inpaint models
            latent_model_input = guidance.modify_latents_before_noise_prediction(latent_model_input, timestep)

        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,  # TODO: debug how handled batched and non batched timesteps
            step_index=step_index,
            total_step_count=total_step_count,
            conditioning_data=conditioning_data,
            ip_adapter_data=ip_adapter_data,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        noise_pred = self.invokeai_diffuser._combine(uc_noise_pred, c_noise_pred, guidance_scale)
        guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier
        if guidance_rescale_multiplier > 0:
            noise_pred = self._rescale_cfg(
                noise_pred,
                c_noise_pred,
                guidance_rescale_multiplier,
            )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **conditioning_data.scheduler_args)

        return step_output

    @staticmethod
    def _rescale_cfg(total_noise_pred, pos_noise_pred, multiplier=0.7):
        """Implementation of Algorithm 2 from https://arxiv.org/pdf/2305.08891.pdf."""
        ro_pos = torch.std(pos_noise_pred, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(total_noise_pred, dim=(1, 2, 3), keepdim=True)

        x_rescaled = total_noise_pred * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * total_noise_pred
        return x_final

    def _unet_forward(
        self,
        latents,
        t,
        text_embeddings,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """predict the noise residual"""
        # First three args should be positional, not keywords, so torch hooks can see them.
        return self.unet(
            latents,
            t,
            text_embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs,
        ).sample
