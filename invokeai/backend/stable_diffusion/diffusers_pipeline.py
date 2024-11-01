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
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.utils.import_utils import is_xformers_available
from pydantic import Field
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterData, TextConditioningData
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher, UNetIPAdapterData
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState
from invokeai.backend.util.attention import auto_detect_slice_size
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.hotfixes import ControlNetModel


@dataclass
class AddsMaskGuidance:
    mask: torch.Tensor
    mask_latents: torch.Tensor
    scheduler: SchedulerMixin
    noise: torch.Tensor
    is_gradient_mask: bool

    def __call__(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.apply_mask(latents, t)

    def apply_mask(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = self.scheduler.add_noise(self.mask_latents, self.noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        # mask_latents = self.scheduler.scale_model_input(mask_latents, t)
        mask_latents = einops.repeat(mask_latents, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if self.is_gradient_mask:
            threshhold = (t.item()) / self.scheduler.config.num_train_timesteps
            mask_bool = mask > threshhold  # I don't know when mask got inverted, but it did
            masked_input = torch.where(mask_bool, latents, mask_latents)
        else:
            masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        return masked_input


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

    def _adjust_memory_efficient_attention(self, latents: torch.Tensor):
        """
        if xformers is available, use it, otherwise use sliced attention.
        """

        # On 30xx and 40xx series GPUs, `torch-sdp` is faster than `xformers`. This corresponds to a CUDA major
        # version of 8 or higher. So, for major version 7 or below, we prefer `xformers`.
        # See:
        # - https://developer.nvidia.com/cuda-gpus
        # - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        try:
            prefer_xformers = torch.cuda.is_available() and torch.cuda.get_device_properties("cuda").major <= 7  # type: ignore # Type of "get_device_properties" is partially unknown
        except Exception:
            prefer_xformers = False

        config = get_config()
        if config.attention_type == "xformers" and is_xformers_available() and prefer_xformers:
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
            # torch-sdp is the default in diffusers.
            return

        # See https://github.com/invoke-ai/InvokeAI/issues/7049 for context.
        # Bumping torch from 2.2.2 to 2.4.1 caused the sliced attention implementation to produce incorrect results.
        # For now, if a user is on an MPS device and has not explicitly set the attention_type, then we select the
        # non-sliced torch-sdp implementation. This keeps things working on MPS at the cost of increased peak memory
        # utilization.
        if torch.backends.mps.is_available():
            return

        # The remainder if this code is called when attention_type=='auto'.
        if self.unet.device.type == "cuda":
            if is_xformers_available() and prefer_xformers:
                self.enable_xformers_memory_efficient_attention()
                return
            # torch-sdp is the default in diffusers.
            return

        if self.unet.device.type == "cpu" or self.unet.device.type == "mps":
            mem_free = psutil.virtual_memory().free
        elif self.unet.device.type == "cuda":
            mem_free, _ = torch.cuda.mem_get_info(TorchDevice.normalize(self.unet.device))
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

    def add_inpainting_channels_to_latents(
        self, latents: torch.Tensor, masked_ref_image_latents: torch.Tensor, inpainting_mask: torch.Tensor
    ):
        """Given a `latents` tensor, adds the mask and image latents channels required for inpainting.

        Standard (non-inpainting) SD UNet models expect an input with shape (N, 4, H, W). Inpainting models expect an
        input of shape (N, 9, H, W). The 9 channels are defined as follows:
        - Channel 0-3: The latents being denoised.
        - Channel 4: The mask indicating which parts of the image are being inpainted.
        - Channel 5-8: The latent representation of the masked reference image being inpainted.

        This function assumes that the same mask and base image should apply to all items in the batch.
        """
        # Validate assumptions about input tensor shapes.
        batch_size, latent_channels, latent_height, latent_width = latents.shape
        assert latent_channels == 4
        assert list(masked_ref_image_latents.shape) == [1, 4, latent_height, latent_width]
        assert list(inpainting_mask.shape) == [1, 1, latent_height, latent_width]

        # Repeat original_image_latents and inpainting_mask to match the latents batch size.
        original_image_latents = masked_ref_image_latents.expand(batch_size, -1, -1, -1)
        inpainting_mask = inpainting_mask.expand(batch_size, -1, -1, -1)

        # Concatenate along the channel dimension.
        return torch.cat([latents, inpainting_mask, original_image_latents], dim=1)

    def latents_from_embeddings(
        self,
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        conditioning_data: TextConditioningData,
        noise: Optional[torch.Tensor],
        seed: int,
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        mask: Optional[torch.Tensor] = None,
        masked_latents: Optional[torch.Tensor] = None,
        is_gradient_mask: bool = False,
    ) -> torch.Tensor:
        """Denoise the latents.

        Args:
            latents: The latent-space image to denoise.
                - If we are inpainting, this is the initial latent image before noise has been added.
                - If we are generating a new image, this should be initialized to zeros.
                - In some cases, this may be a partially-noised latent image (e.g. when running the SDXL refiner).
            scheduler_step_kwargs: kwargs forwarded to the scheduler.step() method.
            conditioning_data: Text conditionging data.
            noise: Noise used for two purposes:
                1. Used by the scheduler to noise the initial `latents` before denoising.
                2. Used to noise the `masked_latents` when inpainting.
                `noise` should be None if the `latents` tensor has already been noised.
            seed: The seed used to generate the noise for the denoising process.
                HACK(ryand): seed is only used in a particular case when `noise` is None, but we need to re-generate the
                same noise used earlier in the pipeline. This should really be handled in a clearer way.
            timesteps: The timestep schedule for the denoising process.
            init_timestep: The first timestep in the schedule. This is used to determine the initial noise level, so
                should be populated if you want noise applied *even* if timesteps is empty.
            callback: A callback function that is called to report progress during the denoising process.
            control_data: ControlNet data.
            ip_adapter_data: IP-Adapter data.
            t2i_adapter_data: T2I-Adapter data.
            mask: A mask indicating which parts of the image are being inpainted. The presence of mask is used to
                determine whether we are inpainting or not. `mask` should have the same spatial dimensions as the
                `latents` tensor.
                TODO(ryand): Check and document the expected dtype, range, and values used to represent
                foreground/background.
            masked_latents: A latent-space representation of a masked inpainting reference image. This tensor is only
                used if an *inpainting* model is being used i.e. this tensor is not used when inpainting with a standard
                SD UNet model.
            is_gradient_mask: A flag indicating whether `mask` is a gradient mask or not.
        """
        if init_timestep.shape[0] == 0:
            return latents

        orig_latents = latents.clone()

        batch_size = latents.shape[0]
        batched_init_timestep = init_timestep.expand(batch_size)

        # noise can be None if the latents have already been noised (e.g. when running the SDXL refiner).
        if noise is not None:
            # TODO(ryand): I'm pretty sure we should be applying init_noise_sigma in cases where we are starting with
            # full noise. Investigate the history of why this got commented out.
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            latents = self.scheduler.add_noise(latents, noise, batched_init_timestep)

        self._adjust_memory_efficient_attention(latents)

        # Handle mask guidance (a.k.a. inpainting).
        mask_guidance: AddsMaskGuidance | None = None
        if mask is not None and not is_inpainting_model(self.unet):
            # We are doing inpainting, since a mask is provided, but we are not using an inpainting model, so we will
            # apply mask guidance to the latents.

            # 'noise' might be None if the latents have already been noised (e.g. when running the SDXL refiner).
            # We still need noise for inpainting, so we generate it from the seed here.
            if noise is None:
                noise = torch.randn(
                    orig_latents.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).to(device=orig_latents.device, dtype=orig_latents.dtype)

            mask_guidance = AddsMaskGuidance(
                mask=mask,
                mask_latents=orig_latents,
                scheduler=self.scheduler,
                noise=noise,
                is_gradient_mask=is_gradient_mask,
            )

        use_ip_adapter = ip_adapter_data is not None
        use_regional_prompting = (
            conditioning_data.cond_regions is not None or conditioning_data.uncond_regions is not None
        )
        unet_attention_patcher = None
        attn_ctx = nullcontext()

        if use_ip_adapter or use_regional_prompting:
            ip_adapters: Optional[List[UNetIPAdapterData]] = (
                [{"ip_adapter": ipa.ip_adapter_model, "target_blocks": ipa.target_blocks} for ipa in ip_adapter_data]
                if use_ip_adapter
                else None
            )
            unet_attention_patcher = UNetAttentionPatcher(ip_adapters)
            attn_ctx = unet_attention_patcher.apply_ip_adapter_attention(self.invokeai_diffuser.model)

        with attn_ctx:
            callback(
                PipelineIntermediateState(
                    step=0,  # initial latents
                    order=self.scheduler.order,
                    total_steps=len(timesteps),
                    timestep=self.scheduler.config.num_train_timesteps,
                    latents=latents,
                )
            )

            for i, t in enumerate(self.progress_bar(timesteps)):
                batched_t = t.expand(batch_size)
                step_output = self.step(
                    t=batched_t,
                    latents=latents,
                    conditioning_data=conditioning_data,
                    step_index=i,
                    total_step_count=len(timesteps),
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    mask_guidance=mask_guidance,
                    mask=mask,
                    masked_latents=masked_latents,
                    control_data=control_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                )
                latents = step_output.prev_sample
                predicted_original = getattr(step_output, "pred_original_sample", None)

                callback(
                    PipelineIntermediateState(
                        step=i + 1,  # final latents
                        order=self.scheduler.order,
                        total_steps=len(timesteps),
                        timestep=int(t),
                        latents=latents,
                        predicted_original=predicted_original,
                    )
                )

        # restore unmasked part after the last step is completed
        # in-process masking happens before each step
        if mask is not None:
            if is_gradient_mask:
                latents = torch.where(mask > 0, latents, orig_latents)
            else:
                latents = torch.lerp(
                    orig_latents, latents.to(dtype=orig_latents.dtype), mask.to(dtype=orig_latents.dtype)
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
        mask_guidance: AddsMaskGuidance | None,
        mask: torch.Tensor | None,
        masked_latents: torch.Tensor | None,
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]

        # Handle masked image-to-image (a.k.a inpainting).
        if mask_guidance is not None:
            # NOTE: This is intentionally done *before* self.scheduler.scale_model_input(...).
            latents = mask_guidance(latents, timestep)

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

            # Hack: force compatibility with irregular resolutions by padding the feature map with zeros
            for idx, tensor in enumerate(accum_adapter_state):
                # The tensor size is supposed to be some integer downscale factor of the latents size.
                # Internally, the unet will pad the latents before downscaling between levels when it is no longer divisible by its downscale factor.
                # If the latent size does not scale down evenly, we need to pad the tensor so that it matches the the downscaled padded latents later on.
                scale_factor = latents.size()[-1] // tensor.size()[-1]
                required_padding_width = math.ceil(latents.size()[-1] / scale_factor) - tensor.size()[-1]
                required_padding_height = math.ceil(latents.size()[-2] / scale_factor) - tensor.size()[-2]
                tensor = torch.nn.functional.pad(
                    tensor,
                    (0, required_padding_width, 0, required_padding_height, 0, 0, 0, 0),
                    mode="constant",
                    value=0,
                )
                accum_adapter_state[idx] = tensor

            down_intrablock_additional_residuals = accum_adapter_state

        # Handle inpainting models.
        if is_inpainting_model(self.unet):
            # NOTE: These calls to add_inpainting_channels_to_latents(...) are intentionally done *after*
            # self.scheduler.scale_model_input(...) so that the scaling is not applied to the mask or reference image
            # latents.
            if mask is not None:
                if masked_latents is None:
                    raise ValueError("Source image required for inpaint mask when inpaint model used!")
                latent_model_input = self.add_inpainting_channels_to_latents(
                    latents=latent_model_input, masked_ref_image_latents=masked_latents, inpainting_mask=mask
                )
            else:
                # We are using an inpainting model, but no mask was provided, so we are not really "inpainting".
                # We generate a global mask and empty original image so that we can still generate in this
                # configuration.
                # TODO(ryand): Should we just raise an exception here instead? I can't think of a use case for wanting
                # to do this.
                # TODO(ryand): If we decide that there is a good reason to keep this, then we should generate the 'fake'
                # mask and original image once rather than on every denoising step.
                latent_model_input = self.add_inpainting_channels_to_latents(
                    latents=latent_model_input,
                    masked_ref_image_latents=torch.zeros_like(latent_model_input[:1]),
                    inpainting_mask=torch.ones_like(latent_model_input[:1, :1]),
                )

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
        step_output = self.scheduler.step(noise_pred, timestep, latents, **scheduler_step_kwargs)

        # TODO: discuss injection point options. For now this is a patch to get progress images working with inpainting
        # again.
        if mask_guidance is not None:
            # Apply the mask to any "denoised" or "pred_original_sample" fields.
            if hasattr(step_output, "denoised"):
                step_output.pred_original_sample = mask_guidance(step_output.denoised, self.scheduler.timesteps[-1])
            elif hasattr(step_output, "pred_original_sample"):
                step_output.pred_original_sample = mask_guidance(
                    step_output.pred_original_sample, self.scheduler.timesteps[-1]
                )
            else:
                step_output.pred_original_sample = mask_guidance(latents, self.scheduler.timesteps[-1])

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
