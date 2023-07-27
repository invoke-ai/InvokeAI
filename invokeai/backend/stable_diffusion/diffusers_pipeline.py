from __future__ import annotations

import dataclasses
import inspect
import math
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, List, Optional, Type, TypeVar, Union
from pydantic import Field

import einops
import PIL.Image
import numpy as np
from accelerate.utils import set_seed
import psutil
import torch
import torchvision.transforms as T
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils import PIL_INTERPOLATION
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.outputs import BaseOutput
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from typing_extensions import ParamSpec

from invokeai.app.services.config import InvokeAIAppConfig
from ..util import CPU_DEVICE, normalize_device
from .diffusion import (
    AttentionMapSaver,
    InvokeAIDiffuserComponent,
    PostprocessingSettings,
)
from .offloading import FullyLoadedModelGroup, ModelGroup


@dataclass
class PipelineIntermediateState:
    run_id: str
    step: int
    timestep: int
    latents: torch.Tensor
    predicted_original: Optional[torch.Tensor] = None
    attention_map_saver: Optional[AttentionMapSaver] = None


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
    mask: torch.FloatTensor
    mask_latents: torch.FloatTensor
    scheduler: SchedulerMixin
    noise: torch.Tensor
    _debug: Optional[Callable] = None

    def __call__(self, step_output: Union[BaseOutput, SchedulerOutput], t: torch.Tensor, conditioning) -> BaseOutput:
        output_class = step_output.__class__  # We'll create a new one with masked data.

        # The problem with taking SchedulerOutput instead of the model output is that we're less certain what's in it.
        # It's reasonable to assume the first thing is prev_sample, but then does it have other things
        # like pred_original_sample? Should we apply the mask to them too?
        # But what if there's just some other random field?
        prev_sample = step_output[0]
        # Mask anything that has the same shape as prev_sample, return others as-is.
        return output_class(
            {
                k: (self.apply_mask(v, self._t_for_field(k, t)) if are_like_tensors(prev_sample, v) else v)
                for k, v in step_output.items()
            }
        )

    def _t_for_field(self, field_name: str, t):
        if field_name == "pred_original_sample":
            return self.scheduler.timesteps[-1]
        return t

    def apply_mask(self, latents: torch.Tensor, t) -> torch.Tensor:
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
        masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        if self._debug:
            self._debug(masked_input, f"t={t} lerped")
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
            T.Resize((h, w), T.InterpolationMode.LANCZOS),
            T.ToTensor(),
        ]
    )
    tensor = transformation(image)
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def is_inpainting_model(unet: UNet2DConditionModel):
    return unet.conv_in.in_channels == 9


CallbackType = TypeVar("CallbackType")
ReturnType = TypeVar("ReturnType")
ParamType = ParamSpec("ParamType")


@dataclass(frozen=True)
class GeneratorToCallbackinator(Generic[ParamType, ReturnType, CallbackType]):
    """Convert a generator to a function with a callback and a return value."""

    generator_method: Callable[ParamType, ReturnType]
    callback_arg_type: Type[CallbackType]

    def __call__(
        self,
        *args: ParamType.args,
        callback: Callable[[CallbackType], Any] = None,
        **kwargs: ParamType.kwargs,
    ) -> ReturnType:
        result = None
        for result in self.generator_method(*args, **kwargs):
            if callback is not None and isinstance(result, self.callback_arg_type):
                callback(result)
        if result is None:
            raise AssertionError("why was that an empty generator?")
        return result


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
class ConditioningData:
    unconditioned_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    guidance_scale: Union[float, List[float]]
    """
    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
    `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
    Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
    images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
    """
    extra: Optional[InvokeAIDiffuserComponent.ExtraConditioningInfo] = None
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    """
    Additional arguments to pass to invokeai_diffuser.do_latent_postprocessing().
    """
    postprocessing_settings: Optional[PostprocessingSettings] = None

    @property
    def dtype(self):
        return self.text_embeddings.dtype

    def add_scheduler_args_if_applicable(self, scheduler, **kwargs):
        scheduler_args = dict(self.scheduler_args)
        step_method = inspect.signature(scheduler.step)
        for name, value in kwargs.items():
            try:
                step_method.bind_partial(**{name: value})
            except TypeError:
                # FIXME: don't silently discard arguments
                pass  # debug("%s does not accept argument named %r", scheduler, name)
            else:
                scheduler_args[name] = value
        return dataclasses.replace(self, scheduler_args=scheduler_args)


@dataclass
class InvokeAIStableDiffusionPipelineOutput(StableDiffusionPipelineOutput):
    r"""
    Output class for InvokeAI's Stable Diffusion pipeline.

    Args:
        attention_map_saver (`AttentionMapSaver`): Object containing attention maps that can be displayed to the user
         after generation completes. Optional.
    """
    attention_map_saver: Optional[AttentionMapSaver]


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
    _model_group: ModelGroup

    ID_LENGTH = 8

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
        precision: str = "float32",
        control_model: ControlNetModel = None,
        execution_device: Optional[torch.device] = None,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            # FIXME: can't currently register control module
            # control_model=control_model,
        )
        self.invokeai_diffuser = InvokeAIDiffuserComponent(self.unet, self._unet_forward)

        self._model_group = FullyLoadedModelGroup(execution_device or self.unet.device)
        self._model_group.install(*self._submodels)
        self.control_model = control_model

    def _adjust_memory_efficient_attention(self, latents: torch.Tensor):
        """
        if xformers is available, use it, otherwise use sliced attention.
        """
        config = InvokeAIAppConfig.get_config()
        if torch.cuda.is_available() and is_xformers_available() and not config.disable_xformers:
            self.enable_xformers_memory_efficient_attention()
        else:
            if self.device.type == "cpu" or self.device.type == "mps":
                mem_free = psutil.virtual_memory().free
            elif self.device.type == "cuda":
                mem_free, _ = torch.cuda.mem_get_info(normalize_device(self.device))
            else:
                raise ValueError(f"unrecognized device {self.device}")
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
        # overridden method; types match the superclass.
        if torch_device is None:
            return self
        self._model_group.set_device(torch.device(torch_device))
        self._model_group.ready()

    @property
    def device(self) -> torch.device:
        return self._model_group.execution_device

    @property
    def _submodels(self) -> Sequence[torch.nn.Module]:
        module_names, _, _ = self.extract_init_dict(dict(self.config))
        submodels = []
        for name in module_names.keys():
            if hasattr(self, name):
                value = getattr(self, name)
            else:
                value = getattr(self.config, name)
            if isinstance(value, torch.nn.Module):
                submodels.append(value)
        return submodels

    def image_from_embeddings(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        conditioning_data: ConditioningData,
        *,
        noise: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None] = None,
        run_id=None,
    ) -> InvokeAIStableDiffusionPipelineOutput:
        r"""
        Function invoked when calling the pipeline for generation.

        :param conditioning_data:
        :param latents: Pre-generated un-noised latents, to be used as inputs for
            image generation. Can be used to tweak the same generation with different prompts.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
            image at the expense of slower inference.
        :param noise: Noise to add to the latents, sampled from a Gaussian distribution.
        :param callback:
        :param run_id:
        """
        result_latents, result_attention_map_saver = self.latents_from_embeddings(
            latents,
            num_inference_steps,
            conditioning_data,
            noise=noise,
            run_id=run_id,
            callback=callback,
        )
        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(result_latents)
            output = InvokeAIStableDiffusionPipelineOutput(
                images=image,
                nsfw_content_detected=[],
                attention_map_saver=result_attention_map_saver,
            )
            return self.check_for_safety(output, dtype=conditioning_data.dtype)

    def latents_from_embeddings(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        conditioning_data: ConditioningData,
        *,
        noise: torch.Tensor,
        timesteps=None,
        additional_guidance: List[Callable] = None,
        run_id=None,
        callback: Callable[[PipelineIntermediateState], None] = None,
        control_data: List[ControlNetData] = None,
    ) -> tuple[torch.Tensor, Optional[AttentionMapSaver]]:
        if self.scheduler.config.get("cpu_only", False):
            scheduler_device = torch.device("cpu")
        else:
            scheduler_device = self._model_group.device_for(self.unet)

        if timesteps is None:
            self.scheduler.set_timesteps(num_inference_steps, device=scheduler_device)
            timesteps = self.scheduler.timesteps
        infer_latents_from_embeddings = GeneratorToCallbackinator(
            self.generate_latents_from_embeddings, PipelineIntermediateState
        )
        result: PipelineIntermediateState = infer_latents_from_embeddings(
            latents,
            timesteps,
            conditioning_data,
            noise=noise,
            run_id=run_id,
            additional_guidance=additional_guidance,
            control_data=control_data,
            callback=callback,
        )
        return result.latents, result.attention_map_saver

    def generate_latents_from_embeddings(
        self,
        latents: torch.Tensor,
        timesteps,
        conditioning_data: ConditioningData,
        *,
        noise: torch.Tensor,
        run_id: str = None,
        additional_guidance: List[Callable] = None,
        control_data: List[ControlNetData] = None,
    ):
        self._adjust_memory_efficient_attention(latents)
        if run_id is None:
            run_id = secrets.token_urlsafe(self.ID_LENGTH)
        if additional_guidance is None:
            additional_guidance = []
        extra_conditioning_info = conditioning_data.extra
        with self.invokeai_diffuser.custom_attention_context(
            self.invokeai_diffuser.model,
            extra_conditioning_info=extra_conditioning_info,
            step_count=len(self.scheduler.timesteps),
        ):
            yield PipelineIntermediateState(
                run_id=run_id,
                step=-1,
                timestep=self.scheduler.config.num_train_timesteps,
                latents=latents,
            )

            batch_size = latents.shape[0]
            batched_t = torch.full(
                (batch_size,),
                timesteps[0],
                dtype=timesteps.dtype,
                device=self._model_group.device_for(self.unet),
            )
            latents = self.scheduler.add_noise(latents, noise, batched_t)

            attention_map_saver: Optional[AttentionMapSaver] = None
            # print("timesteps:", timesteps)
            for i, t in enumerate(self.progress_bar(timesteps)):
                batched_t.fill_(t)
                step_output = self.step(
                    batched_t,
                    latents,
                    conditioning_data,
                    step_index=i,
                    total_step_count=len(timesteps),
                    additional_guidance=additional_guidance,
                    control_data=control_data,
                )
                latents = step_output.prev_sample

                latents = self.invokeai_diffuser.do_latent_postprocessing(
                    postprocessing_settings=conditioning_data.postprocessing_settings,
                    latents=latents,
                    sigma=batched_t,
                    step_index=i,
                    total_step_count=len(timesteps),
                )

                predicted_original = getattr(step_output, "pred_original_sample", None)

                # TODO resuscitate attention map saving
                # if i == len(timesteps)-1 and extra_conditioning_info is not None:
                #    eos_token_index = extra_conditioning_info.tokens_count_including_eos_bos - 1
                #    attention_map_token_ids = range(1, eos_token_index)
                #    attention_map_saver = AttentionMapSaver(token_ids=attention_map_token_ids, latents_shape=latents.shape[-2:])
                #    self.invokeai_diffuser.setup_attention_map_saving(attention_map_saver)

                yield PipelineIntermediateState(
                    run_id=run_id,
                    step=i,
                    timestep=int(t),
                    latents=latents,
                    predicted_original=predicted_original,
                    attention_map_saver=attention_map_saver,
                )

            return latents, attention_map_saver

    @torch.inference_mode()
    def step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: ConditioningData,
        step_index: int,
        total_step_count: int,
        additional_guidance: List[Callable] = None,
        control_data: List[ControlNetData] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]
        if additional_guidance is None:
            additional_guidance = []

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        unet_latent_input = self.scheduler.scale_model_input(latents, timestep)

        # default is no controlnet, so set controlnet processing output to None
        down_block_res_samples, mid_block_res_sample = None, None

        if control_data is not None:
            # control_data should be type List[ControlNetData]
            # this loop covers both ControlNet (one ControlNetData in list)
            #      and MultiControlNet (multiple ControlNetData in list)
            for i, control_datum in enumerate(control_data):
                control_mode = control_datum.control_mode
                # soft_injection and cfg_injection are the two ControlNet control_mode booleans
                #     that are combined at higher level to make control_mode enum
                #  soft_injection determines whether to do per-layer re-weighting adjustment (if True)
                #     or default weighting (if False)
                soft_injection = control_mode == "more_prompt" or control_mode == "more_control"
                #  cfg_injection = determines whether to apply ControlNet to only the conditional (if True)
                #      or the default both conditional and unconditional (if False)
                cfg_injection = control_mode == "more_control" or control_mode == "unbalanced"

                first_control_step = math.floor(control_datum.begin_step_percent * total_step_count)
                last_control_step = math.ceil(control_datum.end_step_percent * total_step_count)
                # only apply controlnet if current step is within the controlnet's begin/end step range
                if step_index >= first_control_step and step_index <= last_control_step:
                    if cfg_injection:
                        control_latent_input = unet_latent_input
                    else:
                        # expand the latents input to control model if doing classifier free guidance
                        #    (which I think for now is always true, there is conditional elsewhere that stops execution if
                        #     classifier_free_guidance is <= 1.0 ?)
                        control_latent_input = torch.cat([unet_latent_input] * 2)

                    if cfg_injection:  # only applying ControlNet to conditional instead of in unconditioned
                        encoder_hidden_states = conditioning_data.text_embeddings
                        encoder_attention_mask = None
                    else:
                        (
                            encoder_hidden_states,
                            encoder_attention_mask,
                        ) = self.invokeai_diffuser._concat_conditionings_for_batch(
                            conditioning_data.unconditioned_embeddings,
                            conditioning_data.text_embeddings,
                        )
                    if isinstance(control_datum.weight, list):
                        # if controlnet has multiple weights, use the weight for the current step
                        controlnet_weight = control_datum.weight[step_index]
                    else:
                        # if controlnet has a single weight, use it for all steps
                        controlnet_weight = control_datum.weight

                    # controlnet(s) inference
                    down_samples, mid_sample = control_datum.model(
                        sample=control_latent_input,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=control_datum.image_tensor,
                        conditioning_scale=controlnet_weight,  # controlnet specific, NOT the guidance scale
                        encoder_attention_mask=encoder_attention_mask,
                        guess_mode=soft_injection,  # this is still called guess_mode in diffusers ControlNetModel
                        return_dict=False,
                    )
                    if cfg_injection:
                        # Inferred ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        #    prepend zeros for unconditional batch
                        down_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_samples]
                        mid_sample = torch.cat([torch.zeros_like(mid_sample), mid_sample])

                    if down_block_res_samples is None and mid_block_res_sample is None:
                        down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                    else:
                        # add controlnet outputs together if have multiple controlnets
                        down_block_res_samples = [
                            samples_prev + samples_curr
                            for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                        ]
                        mid_block_res_sample += mid_sample

        # predict the noise residual
        noise_pred = self.invokeai_diffuser.do_diffusion_step(
            x=unet_latent_input,
            sigma=t,
            unconditioning=conditioning_data.unconditioned_embeddings,
            conditioning=conditioning_data.text_embeddings,
            unconditional_guidance_scale=conditioning_data.guidance_scale,
            step_index=step_index,
            total_step_count=total_step_count,
            down_block_additional_residuals=down_block_res_samples,  # from controlnet(s)
            mid_block_additional_residual=mid_block_res_sample,  # from controlnet(s)
        )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **conditioning_data.scheduler_args)

        # TODO: this additional_guidance extension point feels redundant with InvokeAIDiffusionComponent.
        #    But the way things are now, scheduler runs _after_ that, so there was
        #    no way to use it to apply an operation that happens after the last scheduler.step.
        for guidance in additional_guidance:
            step_output = guidance(step_output, timestep, conditioning_data)

        return step_output

    def _unet_forward(
        self,
        latents,
        t,
        text_embeddings,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """predict the noise residual"""
        if is_inpainting_model(self.unet) and latents.size(1) == 4:
            # Pad out normal non-inpainting inputs for an inpainting model.
            # FIXME: There are too many layers of functions and we have too many different ways of
            #     overriding things! This should get handled in a way more consistent with the other
            #     use of AddsMaskLatents.
            latents = AddsMaskLatents(
                self._unet_forward,
                mask=torch.ones_like(latents[:1, :1], device=latents.device, dtype=latents.dtype),
                initial_image_latents=torch.zeros_like(latents[:1], device=latents.device, dtype=latents.dtype),
            ).add_mask_channels(latents)

        # First three args should be positional, not keywords, so torch hooks can see them.
        return self.unet(
            latents,
            t,
            text_embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs,
        ).sample

    def img2img_from_embeddings(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        strength: float,
        num_inference_steps: int,
        conditioning_data: ConditioningData,
        *,
        callback: Callable[[PipelineIntermediateState], None] = None,
        run_id=None,
        noise_func=None,
        seed=None,
    ) -> InvokeAIStableDiffusionPipelineOutput:
        if isinstance(init_image, PIL.Image.Image):
            init_image = image_resized_to_grid_as_tensor(init_image.convert("RGB"))

        if init_image.dim() == 3:
            init_image = einops.rearrange(init_image, "c h w -> 1 c h w")

        # 6. Prepare latent variables
        initial_latents = self.non_noised_latents_from_image(
            init_image,
            device=self._model_group.device_for(self.unet),
            dtype=self.unet.dtype,
        )
        if seed is not None:
            set_seed(seed)
        noise = noise_func(initial_latents)

        return self.img2img_from_latents_and_embeddings(
            initial_latents,
            num_inference_steps,
            conditioning_data,
            strength,
            noise,
            run_id,
            callback,
        )

    def img2img_from_latents_and_embeddings(
        self,
        initial_latents,
        num_inference_steps,
        conditioning_data: ConditioningData,
        strength,
        noise: torch.Tensor,
        run_id=None,
        callback=None,
    ) -> InvokeAIStableDiffusionPipelineOutput:
        timesteps, _ = self.get_img2img_timesteps(num_inference_steps, strength)
        result_latents, result_attention_maps = self.latents_from_embeddings(
            latents=initial_latents
            if strength < 1.0
            else torch.zeros_like(initial_latents, device=initial_latents.device, dtype=initial_latents.dtype),
            num_inference_steps=num_inference_steps,
            conditioning_data=conditioning_data,
            timesteps=timesteps,
            noise=noise,
            run_id=run_id,
            callback=callback,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(result_latents)
            output = InvokeAIStableDiffusionPipelineOutput(
                images=image,
                nsfw_content_detected=[],
                attention_map_saver=result_attention_maps,
            )
            return self.check_for_safety(output, dtype=conditioning_data.dtype)

    def get_img2img_timesteps(self, num_inference_steps: int, strength: float, device=None) -> (torch.Tensor, int):
        img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.components)
        assert img2img_pipeline.scheduler is self.scheduler

        if self.scheduler.config.get("cpu_only", False):
            scheduler_device = torch.device("cpu")
        else:
            scheduler_device = self._model_group.device_for(self.unet)

        img2img_pipeline.scheduler.set_timesteps(num_inference_steps, device=scheduler_device)
        timesteps, adjusted_steps = img2img_pipeline.get_timesteps(
            num_inference_steps, strength, device=scheduler_device
        )
        # Workaround for low strength resulting in zero timesteps.
        # TODO: submit upstream fix for zero-step img2img
        if timesteps.numel() == 0:
            timesteps = self.scheduler.timesteps[-1:]
            adjusted_steps = timesteps.numel()
        return timesteps, adjusted_steps

    def inpaint_from_embeddings(
        self,
        init_image: torch.FloatTensor,
        mask: torch.FloatTensor,
        strength: float,
        num_inference_steps: int,
        conditioning_data: ConditioningData,
        *,
        callback: Callable[[PipelineIntermediateState], None] = None,
        run_id=None,
        noise_func=None,
        seed=None,
    ) -> InvokeAIStableDiffusionPipelineOutput:
        device = self._model_group.device_for(self.unet)
        latents_dtype = self.unet.dtype

        if isinstance(init_image, PIL.Image.Image):
            init_image = image_resized_to_grid_as_tensor(init_image.convert("RGB"))

        init_image = init_image.to(device=device, dtype=latents_dtype)
        mask = mask.to(device=device, dtype=latents_dtype)

        if init_image.dim() == 3:
            init_image = init_image.unsqueeze(0)

        timesteps, _ = self.get_img2img_timesteps(num_inference_steps, strength)

        # 6. Prepare latent variables
        # can't quite use upstream StableDiffusionImg2ImgPipeline.prepare_latents
        # because we have our own noise function
        init_image_latents = self.non_noised_latents_from_image(init_image, device=device, dtype=latents_dtype)
        if seed is not None:
            set_seed(seed)
        noise = noise_func(init_image_latents)

        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        latent_mask = tv_resize(mask, init_image_latents.shape[-2:], T.InterpolationMode.BILINEAR).to(
            device=device, dtype=latents_dtype
        )

        guidance: List[Callable] = []

        if is_inpainting_model(self.unet):
            # You'd think the inpainting model wouldn't be paying attention to the area it is going to repaint
            # (that's why there's a mask!) but it seems to really want that blanked out.
            masked_init_image = init_image * torch.where(mask < 0.5, 1, 0)
            masked_latents = self.non_noised_latents_from_image(masked_init_image, device=device, dtype=latents_dtype)

            # TODO: we should probably pass this in so we don't have to try/finally around setting it.
            self.invokeai_diffuser.model_forward_callback = AddsMaskLatents(
                self._unet_forward, latent_mask, masked_latents
            )
        else:
            guidance.append(AddsMaskGuidance(latent_mask, init_image_latents, self.scheduler, noise))

        try:
            result_latents, result_attention_maps = self.latents_from_embeddings(
                latents=init_image_latents
                if strength < 1.0
                else torch.zeros_like(
                    init_image_latents, device=init_image_latents.device, dtype=init_image_latents.dtype
                ),
                num_inference_steps=num_inference_steps,
                conditioning_data=conditioning_data,
                noise=noise,
                timesteps=timesteps,
                additional_guidance=guidance,
                run_id=run_id,
                callback=callback,
            )
        finally:
            self.invokeai_diffuser.model_forward_callback = self._unet_forward

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(result_latents)
            output = InvokeAIStableDiffusionPipelineOutput(
                images=image,
                nsfw_content_detected=[],
                attention_map_saver=result_attention_maps,
            )
            return self.check_for_safety(output, dtype=conditioning_data.dtype)

    def non_noised_latents_from_image(self, init_image, *, device: torch.device, dtype):
        init_image = init_image.to(device=device, dtype=dtype)
        with torch.inference_mode():
            self._model_group.load(self.vae)
            init_latent_dist = self.vae.encode(init_image).latent_dist
            init_latents = init_latent_dist.sample().to(dtype=dtype)  # FIXME: uses torch.randn. make reproducible!

        init_latents = 0.18215 * init_latents
        return init_latents

    def check_for_safety(self, output, dtype):
        with torch.inference_mode():
            screened_images, has_nsfw_concept = self.run_safety_checker(output.images, dtype=dtype)
        screened_attention_map_saver = None
        if has_nsfw_concept is None or not has_nsfw_concept:
            screened_attention_map_saver = output.attention_map_saver
        return InvokeAIStableDiffusionPipelineOutput(
            screened_images,
            has_nsfw_concept,
            # block the attention maps if NSFW content is detected
            attention_map_saver=screened_attention_map_saver,
        )

    def run_safety_checker(self, image, device=None, dtype=None):
        # overriding to use the model group for device info instead of requiring the caller to know.
        if self.safety_checker is not None:
            device = self._model_group.device_for(self.safety_checker)
        return super().run_safety_checker(image, device, dtype)

    def decode_latents(self, latents):
        # Explicit call to get the vae loaded, since `decode` isn't the forward method.
        self._model_group.load(self.vae)
        return super().decode_latents(latents)

    def debug_latents(self, latents, msg):
        from invokeai.backend.image_util import debug_image

        with torch.inference_mode():
            decoded = self.numpy_to_pil(self.decode_latents(latents))
        for i, img in enumerate(decoded):
            debug_image(img, f"latents {msg} {i+1}/{len(decoded)}", debug_status=True)
