from __future__ import annotations

import dataclasses
import inspect
import secrets
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union, Callable, Type, TypeVar, Generic, Any

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

import PIL.Image
import einops
import torch
import torchvision.transforms as T
from diffusers.models import attention
from diffusers.utils.import_utils import is_xformers_available

from ...models.diffusion import cross_attention_control
from ...models.diffusion.cross_attention_map_saving import AttentionMapSaver
from ...modules.prompt_to_embeddings_converter import WeightedPromptFragmentsToEmbeddingsConverter

# monkeypatch diffusers CrossAttention 🙈
# this is to make prompt2prompt and (future) attention maps work
attention.CrossAttention = cross_attention_control.InvokeAIDiffusersCrossAttention

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils.outputs import BaseOutput
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ldm.invoke.globals import Globals
from ldm.models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent, ThresholdSettings
from ldm.modules.textual_inversion_manager import TextualInversionManager


@dataclass
class PipelineIntermediateState:
    run_id: str
    step: int
    timestep: int
    latents: torch.Tensor
    predicted_original: Optional[torch.Tensor] = None
    attention_map_saver: Optional[AttentionMapSaver] = None


# copied from configs/stable-diffusion/v1-inference.yaml
_default_personalization_config_params = dict(
    placeholder_strings=["*"],
    initializer_wods=["sculpture"],
    per_image_tokens=False,
    num_vectors_per_token=1,
    progressive_words=False
)


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

    def __call__(self, latents: torch.Tensor, t: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        model_input = self.add_mask_channels(latents)
        return self.forward(model_input, t, text_embeddings)

    def add_mask_channels(self, latents):
        batch_size = latents.size(0)
        # duplicate mask and latents for each batch
        mask = einops.repeat(self.mask, 'b c h w -> (repeat b) c h w', repeat=batch_size)
        image_latents = einops.repeat(self.initial_image_latents, 'b c h w -> (repeat b) c h w', repeat=batch_size)
        # add mask and image as additional channels
        model_input, _ = einops.pack([latents, mask, image_latents], 'b * h w')
        return model_input


def are_like_tensors(a: torch.Tensor, b: object) -> bool:
    return (
        isinstance(b, torch.Tensor)
        and (a.size() == b.size())
    )

@dataclass
class AddsMaskGuidance:
    mask: torch.FloatTensor
    mask_latents: torch.FloatTensor
    scheduler: SchedulerMixin
    noise: torch.Tensor
    _debug: Optional[Callable] = None

    def __call__(self, step_output: BaseOutput | SchedulerOutput, t: torch.Tensor, conditioning) -> BaseOutput:
        output_class = step_output.__class__  # We'll create a new one with masked data.

        # The problem with taking SchedulerOutput instead of the model output is that we're less certain what's in it.
        # It's reasonable to assume the first thing is prev_sample, but then does it have other things
        # like pred_original_sample? Should we apply the mask to them too?
        # But what if there's just some other random field?
        prev_sample = step_output[0]
        # Mask anything that has the same shape as prev_sample, return others as-is.
        return output_class(
            {k: (self.apply_mask(v, self._t_for_field(k, t))
                 if are_like_tensors(prev_sample, v) else v)
            for k, v in step_output.items()}
        )

    def _t_for_field(self, field_name:str, t):
        if field_name == "pred_original_sample":
            return torch.zeros_like(t, dtype=t.dtype)  # it represents t=0
        return t

    def apply_mask(self, latents: torch.Tensor, t) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self.mask, 'b c h w -> (repeat b) c h w', repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, '-> batch', batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = self.scheduler.add_noise(self.mask_latents, self.noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        # mask_latents = self.scheduler.scale_model_input(mask_latents, t)
        mask_latents = einops.repeat(mask_latents, 'b c h w -> (repeat b) c h w', repeat=batch_size)
        masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        if self._debug:
            self._debug(masked_input, f"t={t} lerped")
        return masked_input


def trim_to_multiple_of(*args, multiple_of=8):
    return tuple((x - x % multiple_of) for x in args)


def image_resized_to_grid_as_tensor(image: PIL.Image.Image, normalize: bool=True, multiple_of=8) -> torch.FloatTensor:
    """

    :param image: input image
    :param normalize: scale the range to [-1, 1] instead of [0, 1]
    :param multiple_of: resize the input so both dimensions are a multiple of this
    """
    w, h = trim_to_multiple_of(*image.size)
    transformation = T.Compose([
        T.Resize((h, w), T.InterpolationMode.LANCZOS),
        T.ToTensor(),
    ])
    tensor = transformation(image)
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def is_inpainting_model(unet: UNet2DConditionModel):
    return unet.conv_in.in_channels == 9

CallbackType = TypeVar('CallbackType')
ReturnType = TypeVar('ReturnType')
ParamType = ParamSpec('ParamType')

@dataclass(frozen=True)
class GeneratorToCallbackinator(Generic[ParamType, ReturnType, CallbackType]):
    """Convert a generator to a function with a callback and a return value."""

    generator_method: Callable[ParamType, ReturnType]
    callback_arg_type: Type[CallbackType]

    def __call__(self, *args: ParamType.args,
                 callback:Callable[[CallbackType], Any]=None,
                 **kwargs: ParamType.kwargs) -> ReturnType:
        result = None
        for result in self.generator_method(*args, **kwargs):
            if callback is not None and isinstance(result, self.callback_arg_type):
                callback(result)
        if result is None:
            raise AssertionError("why was that an empty generator?")
        return result


@dataclass(frozen=True)
class ConditioningData:
    unconditioned_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    guidance_scale: float
    """
    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
    `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
    Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
    images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
    """
    extra: Optional[InvokeAIDiffuserComponent.ExtraConditioningInfo] = None
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    """Additional arguments to pass to scheduler.step."""
    threshold: Optional[ThresholdSettings] = None

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
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    ID_LENGTH = 8

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: Optional[CLIPFeatureExtractor],
        requires_safety_checker: bool = False,
        precision: str = 'float32',
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         safety_checker, feature_extractor, requires_safety_checker)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.invokeai_diffuser = InvokeAIDiffuserComponent(self.unet, self._unet_forward)
        use_full_precision = (precision == 'float32' or precision == 'autocast')
        self.textual_inversion_manager = TextualInversionManager(tokenizer=self.tokenizer,
                                                                 text_encoder=self.text_encoder,
                                                                 full_precision=use_full_precision)
        # InvokeAI's interface for text embeddings and whatnot
        self.prompt_fragments_to_embeddings_converter = WeightedPromptFragmentsToEmbeddingsConverter(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            textual_inversion_manager=self.textual_inversion_manager
        )

        if is_xformers_available() and not Globals.disable_xformers:
            self.enable_xformers_memory_efficient_attention()

    def image_from_embeddings(self, latents: torch.Tensor, num_inference_steps: int,
                              conditioning_data: ConditioningData,
                              *,
                              noise: torch.Tensor,
                              callback: Callable[[PipelineIntermediateState], None]=None,
                              run_id=None) -> InvokeAIStableDiffusionPipelineOutput:
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
            latents, num_inference_steps,
            conditioning_data,
            noise=noise,
            run_id=run_id,
            callback=callback)
        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(result_latents)
            output = InvokeAIStableDiffusionPipelineOutput(images=image, nsfw_content_detected=[], attention_map_saver=result_attention_map_saver)
            return self.check_for_safety(output, dtype=conditioning_data.dtype)

    def latents_from_embeddings(self, latents: torch.Tensor, num_inference_steps: int,
                                conditioning_data: ConditioningData,
                                *,
                                noise: torch.Tensor,
                                timesteps=None,
                                additional_guidance: List[Callable] = None, run_id=None,
                                callback: Callable[[PipelineIntermediateState], None] = None
                                ) -> tuple[torch.Tensor, Optional[AttentionMapSaver]]:
        if timesteps is None:
            self.scheduler.set_timesteps(num_inference_steps, device=self.unet.device)
            timesteps = self.scheduler.timesteps
        infer_latents_from_embeddings = GeneratorToCallbackinator(self.generate_latents_from_embeddings, PipelineIntermediateState)
        result: PipelineIntermediateState = infer_latents_from_embeddings(
            latents, timesteps, conditioning_data,
            noise=noise,
            additional_guidance=additional_guidance,
            run_id=run_id,
            callback=callback)
        return result.latents, result.attention_map_saver

    def generate_latents_from_embeddings(self, latents: torch.Tensor, timesteps,
                                         conditioning_data: ConditioningData,
                                         *,
                                         noise: torch.Tensor,
                                         run_id: str = None,
                                         additional_guidance: List[Callable] = None):
        if run_id is None:
            run_id = secrets.token_urlsafe(self.ID_LENGTH)
        if additional_guidance is None:
            additional_guidance = []
        extra_conditioning_info = conditioning_data.extra
        if extra_conditioning_info is not None and extra_conditioning_info.wants_cross_attention_control:
            self.invokeai_diffuser.setup_cross_attention_control(extra_conditioning_info,
                                                                 step_count=len(self.scheduler.timesteps))
        else:
            self.invokeai_diffuser.remove_cross_attention_control()

        yield PipelineIntermediateState(run_id=run_id, step=-1, timestep=self.scheduler.num_train_timesteps,
                                        latents=latents)

        batch_size = latents.shape[0]
        batched_t = torch.full((batch_size,), timesteps[0],
                               dtype=timesteps.dtype, device=self.unet.device)
        latents = self.scheduler.add_noise(latents, noise, batched_t)

        attention_map_saver: Optional[AttentionMapSaver] = None
        self.invokeai_diffuser.remove_attention_map_saving()
        for i, t in enumerate(self.progress_bar(timesteps)):
            batched_t.fill_(t)
            step_output = self.step(batched_t, latents, conditioning_data,
                                    step_index=i,
                                    total_step_count=len(timesteps),
                                    additional_guidance=additional_guidance)
            latents = step_output.prev_sample
            predicted_original = getattr(step_output, 'pred_original_sample', None)

            if i == len(timesteps)-1 and extra_conditioning_info is not None:
                eos_token_index = extra_conditioning_info.tokens_count_including_eos_bos - 1
                attention_map_token_ids = range(1, eos_token_index)
                attention_map_saver = AttentionMapSaver(token_ids=attention_map_token_ids, latents_shape=latents.shape[-2:])
                self.invokeai_diffuser.setup_attention_map_saving(attention_map_saver)

            yield PipelineIntermediateState(run_id=run_id, step=i, timestep=int(t), latents=latents,
                                            predicted_original=predicted_original, attention_map_saver=attention_map_saver)

        self.invokeai_diffuser.remove_attention_map_saving()
        return latents, attention_map_saver

    @torch.inference_mode()
    def step(self, t: torch.Tensor, latents: torch.Tensor,
             conditioning_data: ConditioningData,
             step_index:int, total_step_count:int,
             additional_guidance: List[Callable] = None):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]

        if additional_guidance is None:
            additional_guidance = []

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # predict the noise residual
        noise_pred = self.invokeai_diffuser.do_diffusion_step(
            latent_model_input, t,
            conditioning_data.unconditioned_embeddings, conditioning_data.text_embeddings,
            conditioning_data.guidance_scale,
            step_index=step_index,
            total_step_count=total_step_count,
            threshold=conditioning_data.threshold
        )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents,
                                          **conditioning_data.scheduler_args)

        # TODO: this additional_guidance extension point feels redundant with InvokeAIDiffusionComponent.
        #    But the way things are now, scheduler runs _after_ that, so there was
        #    no way to use it to apply an operation that happens after the last scheduler.step.
        for guidance in additional_guidance:
            step_output = guidance(step_output, timestep, conditioning_data)

        return step_output

    def _unet_forward(self, latents, t, text_embeddings):
        """predict the noise residual"""
        if is_inpainting_model(self.unet) and latents.size(1) == 4:
            # Pad out normal non-inpainting inputs for an inpainting model.
            # FIXME: There are too many layers of functions and we have too many different ways of
            #     overriding things! This should get handled in a way more consistent with the other
            #     use of AddsMaskLatents.
            latents = AddsMaskLatents(
                self._unet_forward,
                mask=torch.ones_like(latents[:1, :1], device=latents.device, dtype=latents.dtype),
                initial_image_latents=torch.zeros_like(latents[:1], device=latents.device, dtype=latents.dtype)
            ).add_mask_channels(latents)

        return self.unet(latents, t, encoder_hidden_states=text_embeddings).sample

    def img2img_from_embeddings(self,
                                init_image: Union[torch.FloatTensor, PIL.Image.Image],
                                strength: float,
                                num_inference_steps: int,
                                conditioning_data: ConditioningData,
                                *, callback: Callable[[PipelineIntermediateState], None] = None,
                                run_id=None,
                                noise_func=None
                                ) -> InvokeAIStableDiffusionPipelineOutput:
        if isinstance(init_image, PIL.Image.Image):
            init_image = image_resized_to_grid_as_tensor(init_image.convert('RGB'))

        if init_image.dim() == 3:
            init_image = einops.rearrange(init_image, 'c h w -> 1 c h w')

        # 6. Prepare latent variables
        device = self.unet.device
        latents_dtype = self.unet.dtype
        initial_latents = self.non_noised_latents_from_image(init_image, device=device, dtype=latents_dtype)
        noise = noise_func(initial_latents)

        return self.img2img_from_latents_and_embeddings(initial_latents, num_inference_steps,
                                                        conditioning_data,
                                                        strength,
                                                        noise, run_id, callback)

    def img2img_from_latents_and_embeddings(self, initial_latents, num_inference_steps,
                                            conditioning_data: ConditioningData,
                                            strength,
                                            noise: torch.Tensor, run_id=None, callback=None
                                            ) -> InvokeAIStableDiffusionPipelineOutput:
        device = self.unet.device
        img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.components)
        img2img_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = img2img_pipeline.get_timesteps(num_inference_steps, strength, device=device)

        result_latents, result_attention_maps = self.latents_from_embeddings(
            initial_latents, num_inference_steps, conditioning_data,
            timesteps=timesteps,
            noise=noise,
            run_id=run_id,
            callback=callback)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(result_latents)
            output = InvokeAIStableDiffusionPipelineOutput(images=image, nsfw_content_detected=[], attention_map_saver=result_attention_maps)
            return self.check_for_safety(output, dtype=conditioning_data.dtype)

    def inpaint_from_embeddings(
            self,
            init_image: torch.FloatTensor,
            mask: torch.FloatTensor,
            strength: float,
            num_inference_steps: int,
            conditioning_data: ConditioningData,
            *, callback: Callable[[PipelineIntermediateState], None] = None,
            run_id=None,
            noise_func=None,
            ) -> InvokeAIStableDiffusionPipelineOutput:
        device = self.unet.device
        latents_dtype = self.unet.dtype

        if isinstance(init_image, PIL.Image.Image):
            init_image = image_resized_to_grid_as_tensor(init_image.convert('RGB'))

        init_image = init_image.to(device=device, dtype=latents_dtype)

        if init_image.dim() == 3:
            init_image = init_image.unsqueeze(0)

        img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.components)
        img2img_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = img2img_pipeline.get_timesteps(num_inference_steps, strength, device=device)

        assert img2img_pipeline.scheduler is self.scheduler

        # 6. Prepare latent variables
        # can't quite use upstream StableDiffusionImg2ImgPipeline.prepare_latents
        # because we have our own noise function
        init_image_latents = self.non_noised_latents_from_image(init_image, device=device, dtype=latents_dtype)
        noise = noise_func(init_image_latents)

        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = tv_resize(mask, init_image_latents.shape[-2:], T.InterpolationMode.BILINEAR) \
            .to(device=device, dtype=latents_dtype)

        guidance: List[Callable] = []

        if is_inpainting_model(self.unet):
            # TODO: we should probably pass this in so we don't have to try/finally around setting it.
            self.invokeai_diffuser.model_forward_callback = \
                AddsMaskLatents(self._unet_forward, mask, init_image_latents)
        else:
            guidance.append(AddsMaskGuidance(mask, init_image_latents, self.scheduler, noise))

        try:
            result_latents, result_attention_maps = self.latents_from_embeddings(
                init_image_latents, num_inference_steps,
                conditioning_data, noise=noise, timesteps=timesteps,
                additional_guidance=guidance,
                run_id=run_id, callback=callback)
        finally:
            self.invokeai_diffuser.model_forward_callback = self._unet_forward

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(result_latents)
            output = InvokeAIStableDiffusionPipelineOutput(images=image, nsfw_content_detected=[], attention_map_saver=result_attention_maps)
            return self.check_for_safety(output, dtype=conditioning_data.dtype)

    def non_noised_latents_from_image(self, init_image, *, device, dtype):
        init_image = init_image.to(device=device, dtype=dtype)
        with torch.inference_mode():
            init_latent_dist = self.vae.encode(init_image).latent_dist
            init_latents = init_latent_dist.sample().to(dtype=dtype)  # FIXME: uses torch.randn. make reproducible!
        init_latents = 0.18215 * init_latents
        return init_latents

    def check_for_safety(self, output, dtype):
        with torch.inference_mode():
            screened_images, has_nsfw_concept = self.run_safety_checker(
                output.images, device=self._execution_device, dtype=dtype)
        screened_attention_map_saver = None
        if has_nsfw_concept is None or not has_nsfw_concept:
            screened_attention_map_saver = output.attention_map_saver
        return InvokeAIStableDiffusionPipelineOutput(screened_images,
                                                     has_nsfw_concept,
                                                     # block the attention maps if NSFW content is detected
                                                     attention_map_saver=screened_attention_map_saver)

    @torch.inference_mode()
    def get_learned_conditioning(self, c: List[List[str]], *, return_tokens=True, fragment_weights=None):
        """
        Compatibility function for ldm.models.diffusion.ddpm.LatentDiffusion.
        """
        return self.prompt_fragments_to_embeddings_converter.get_embeddings_for_weighted_prompt_fragments(
            text=c,
            fragment_weights=fragment_weights,
            should_return_tokens=return_tokens,
            device=self.device)

    @property
    def cond_stage_model(self):
        warnings.warn("legacy compatibility layer", DeprecationWarning)
        return self.prompt_fragments_to_embeddings_converter

    @torch.inference_mode()
    def _tokenize(self, prompt: Union[str, List[str]]):
        return self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    @property
    def channels(self) -> int:
        """Compatible with DiffusionWrapper"""
        return self.unet.in_channels

    def debug_latents(self, latents, msg):
        with torch.inference_mode():
            from ldm.util import debug_image
            decoded = self.numpy_to_pil(self.decode_latents(latents))
            for i, img in enumerate(decoded):
                debug_image(img, f"latents {msg} {i+1}/{len(decoded)}", debug_status=True)
