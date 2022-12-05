from __future__ import annotations

import secrets
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union, Callable

import PIL.Image
import einops
import torch
import torchvision.transforms as T
from diffusers.models import attention

from ldm.models.diffusion.cross_attention_control import InvokeAIDiffusersCrossAttention

# monkeypatch diffusers CrossAttention 🙈
# this is to make prompt2prompt and (future) attention maps work
attention.CrossAttention = InvokeAIDiffusersCrossAttention

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

from ldm.models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from ldm.modules.embedding_manager import EmbeddingManager
from ldm.modules.encoders.modules import WeightedFrozenCLIPEmbedder


@dataclass
class PipelineIntermediateState:
    run_id: str
    step: int
    timestep: int
    latents: torch.Tensor
    predicted_original: Optional[torch.Tensor] = None


# copied from configs/stable-diffusion/v1-inference.yaml
_default_personalization_config_params = dict(
    placeholder_strings=["*"],
    initializer_wods=["sculpture"],
    per_image_tokens=False,
    num_vectors_per_token=8,
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
    mask: torch.FloatTensor
    initial_image_latents: torch.FloatTensor

    def __call__(self, latents: torch.FloatTensor, t: torch.Tensor, text_embeddings: torch.FloatTensor) -> torch.Tensor:
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
    _scheduler: SchedulerMixin
    _noise_func: Callable
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
        noise = self._noise_func(self.mask_latents)
        mask_latents = self._scheduler.add_noise(self.mask_latents, noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        mask_latents = einops.repeat(mask_latents, 'b c h w -> (repeat b) c h w', repeat=batch_size)
        masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        if self._debug:
            self._debug(masked_input, f"t={t} lerped")
        return masked_input


def image_resized_to_grid_as_tensor(image: PIL.Image.Image, normalize: bool=True, multiple_of=8) -> torch.FloatTensor:
    """

    :param image: input image
    :param normalize: scale the range to [-1, 1] instead of [0, 1]
    :param multiple_of: resize the input so both dimensions are a multiple of this
    """
    w, h = image.size
    w, h = map(lambda x: x - x % multiple_of, (w, h))  # resize to integer multiple of 8
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
        requires_safety_checker: bool = False
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
        # InvokeAI's interface for text embeddings and whatnot
        self.clip_embedder = WeightedFrozenCLIPEmbedder(
            tokenizer=self.tokenizer,
            transformer=self.text_encoder
        )
        self.invokeai_diffuser = InvokeAIDiffuserComponent(self.unet, self._unet_forward)
        self.embedding_manager = EmbeddingManager(self.clip_embedder, **_default_personalization_config_params)

    def image_from_embeddings(self, latents: torch.Tensor, num_inference_steps: int,
                              text_embeddings: torch.Tensor, unconditioned_embeddings: torch.Tensor,
                              guidance_scale: float,
                              *, callback: Callable[[PipelineIntermediateState], None]=None,
                              extra_conditioning_info: InvokeAIDiffuserComponent.ExtraConditioningInfo=None,
                              run_id=None,
                              **extra_step_kwargs) -> StableDiffusionPipelineOutput:
        r"""
        Function invoked when calling the pipeline for generation.

        :param latents: Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for
            image generation. Can be used to tweak the same generation with different prompts.
        :param num_inference_steps: The number of denoising steps. More denoising steps usually lead to a higher quality
            image at the expense of slower inference.
        :param text_embeddings:
        :param guidance_scale: Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
             Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
             images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
        :param callback:
        :param extra_conditioning_info:
        :param run_id:
        :param extra_step_kwargs:
        """
        self.scheduler.set_timesteps(num_inference_steps, device=self.unet.device)
        result = None
        for result in self.generate_from_embeddings(
                latents, text_embeddings, unconditioned_embeddings, guidance_scale,
                extra_conditioning_info=extra_conditioning_info,
                run_id=run_id, **extra_step_kwargs):
            if callback is not None and isinstance(result, PipelineIntermediateState):
                callback(result)
        if result is None:
            raise AssertionError("why was that an empty generator?")
        return result

    def generate(
        self,
        prompt: Union[str, List[str]],
        *,
        opposing_prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        run_id: str = None,
        **extra_step_kwargs,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        combined_embeddings = self._encode_prompt(prompt, device=self._execution_device, num_images_per_prompt=1,
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            negative_prompt=opposing_prompt)
        text_embeddings, unconditioned_embeddings = combined_embeddings.chunk(2)
        self.scheduler.set_timesteps(num_inference_steps)
        latents = self.prepare_latents(batch_size=batch_size, num_channels_latents=self.unet.in_channels,
                                       height=height, width=width,
                                       dtype=self.unet.dtype, device=self._execution_device,
                                       generator=generator,
                                       latents=latents)

        yield from self.generate_from_embeddings(latents, text_embeddings, unconditioned_embeddings,
                                                 guidance_scale, run_id=run_id, **extra_step_kwargs)

    def generate_from_embeddings(
            self,
            latents: torch.Tensor,
            text_embeddings: torch.Tensor,
            unconditioned_embeddings: torch.Tensor,
            guidance_scale: float,
            *,
            run_id: str = None,
            extra_conditioning_info: InvokeAIDiffuserComponent.ExtraConditioningInfo = None,
            timesteps = None,
            additional_guidance: List[Callable] = None,
            **extra_step_kwargs):
        if run_id is None:
            run_id = secrets.token_urlsafe(self.ID_LENGTH)

        if additional_guidance is None:
            additional_guidance = []

        if extra_conditioning_info is not None and extra_conditioning_info.wants_cross_attention_control:
            self.invokeai_diffuser.setup_cross_attention_control(extra_conditioning_info,
                                                                 step_count=len(self.scheduler.timesteps))
        else:
            self.invokeai_diffuser.remove_cross_attention_control()

        if timesteps is None:
            timesteps = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        latents *= self.scheduler.init_noise_sigma
        yield PipelineIntermediateState(run_id=run_id, step=-1, timestep=self.scheduler.num_train_timesteps,
                                        latents=latents)

        batch_size = latents.shape[0]
        batched_t = torch.full((batch_size,), timesteps[0],
                               dtype=timesteps.dtype, device=self.unet.device)
        # NOTE: Depends on scheduler being already initialized!
        for i, t in enumerate(self.progress_bar(timesteps)):
            batched_t.fill_(t)
            step_output = self.step(batched_t, latents, guidance_scale,
                                    text_embeddings, unconditioned_embeddings,
                                    i, additional_guidance=additional_guidance,
                                    **extra_step_kwargs)
            latents = step_output.prev_sample
            predicted_original = getattr(step_output, 'pred_original_sample', None)
            yield PipelineIntermediateState(run_id=run_id, step=i, timestep=int(t), latents=latents,
                                            predicted_original=predicted_original)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        with torch.inference_mode():
            image = self.decode_latents(latents)
            output = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=[])
            yield self.check_for_safety(output, dtype=text_embeddings.dtype)

    @torch.inference_mode()
    def step(self, t: torch.Tensor, latents: torch.Tensor, guidance_scale: float,
             text_embeddings: torch.Tensor, unconditioned_embeddings: torch.Tensor,
             step_index:int | None = None, additional_guidance: List[Callable] = None,
             **extra_step_kwargs):
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
            unconditioned_embeddings, text_embeddings,
            guidance_scale,
            step_index=step_index)

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs)

        # TODO: this additional_guidance extension point feels redundant with InvokeAIDiffusionComponent.
        #    But the way things are now, scheduler runs _after_ that, so there was
        #    no way to use it to apply an operation that happens after the last scheduler.step.
        for guidance in additional_guidance:
            step_output = guidance(step_output, timestep, (unconditioned_embeddings, text_embeddings))

        return step_output

    def _unet_forward(self, latents, t, text_embeddings):
        # predict the noise residual
        return self.unet(latents, t, encoder_hidden_states=text_embeddings).sample

    def img2img_from_embeddings(self,
                                init_image: Union[torch.FloatTensor, PIL.Image.Image],
                                strength: float,
                                num_inference_steps: int,
                                text_embeddings: torch.Tensor, unconditioned_embeddings: torch.Tensor,
                                guidance_scale: float,
                                *, callback: Callable[[PipelineIntermediateState], None] = None,
                                extra_conditioning_info: InvokeAIDiffuserComponent.ExtraConditioningInfo = None,
                                run_id=None,
                                noise_func=None,
                                **extra_step_kwargs) -> StableDiffusionPipelineOutput:
        device = self.unet.device
        latents_dtype = self.unet.dtype
        batch_size = 1
        num_images_per_prompt = 1

        if isinstance(init_image, PIL.Image.Image):
            init_image = image_resized_to_grid_as_tensor(init_image.convert('RGB'))

        if init_image.dim() == 3:
            init_image = einops.rearrange(init_image, 'c h w -> 1 c h w')

        img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.components)
        img2img_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = img2img_pipeline.get_timesteps(num_inference_steps, strength, device=device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents, _ = self.prepare_latents_from_image(init_image, latent_timestep, latents_dtype, device, noise_func)

        result = None
        for result in self.generate_from_embeddings(
                latents, text_embeddings, unconditioned_embeddings, guidance_scale,
                extra_conditioning_info=extra_conditioning_info,
                timesteps=timesteps,
                run_id=run_id, **extra_step_kwargs):
            if callback is not None and isinstance(result, PipelineIntermediateState):
                callback(result)
        if result is None:
            raise AssertionError("why was that an empty generator?")
        return result

    def inpaint_from_embeddings(
            self,
            init_image: torch.FloatTensor,
            mask: torch.FloatTensor,
            strength: float,
            num_inference_steps: int,
            text_embeddings: torch.Tensor, unconditioned_embeddings: torch.Tensor,
            guidance_scale: float,
            *, callback: Callable[[PipelineIntermediateState], None] = None,
            extra_conditioning_info: InvokeAIDiffuserComponent.ExtraConditioningInfo = None,
            run_id=None,
            noise_func=None,
            **extra_step_kwargs) -> StableDiffusionPipelineOutput:
        device = self.unet.device
        latents_dtype = self.unet.dtype
        batch_size = 1
        num_images_per_prompt = 1

        if isinstance(init_image, PIL.Image.Image):
            init_image = image_resized_to_grid_as_tensor(init_image.convert('RGB'))

        init_image = init_image.to(device=device, dtype=latents_dtype)

        if init_image.dim() == 3:
            init_image = init_image.unsqueeze(0)

        img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.components)
        img2img_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = img2img_pipeline.get_timesteps(num_inference_steps, strength, device=device)

        assert img2img_pipeline.scheduler is self.scheduler

        # 6. Prepare latent variables
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents, init_image_latents = self.prepare_latents_from_image(init_image, latent_timestep, latents_dtype, device, noise_func)

        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = tv_resize(mask, latents.shape[-2:], T.InterpolationMode.BILINEAR) \
            .to(device=device, dtype=latents_dtype)

        guidance: List[Callable] = []

        if is_inpainting_model(self.unet):
            # TODO: we should probably pass this in so we don't have to try/finally around setting it.
            self.invokeai_diffuser.model_forward_callback = \
                AddsMaskLatents(self._unet_forward, mask, init_image_latents)
        else:
            guidance.append(AddsMaskGuidance(mask, init_image_latents, self.scheduler, noise_func))

        result = None

        try:
            for result in self.generate_from_embeddings(
                    latents, text_embeddings, unconditioned_embeddings, guidance_scale,
                    extra_conditioning_info=extra_conditioning_info,
                    timesteps=timesteps,
                    run_id=run_id, additional_guidance=guidance, **extra_step_kwargs):
                if callback is not None and isinstance(result, PipelineIntermediateState):
                    callback(result)
            if result is None:
                raise AssertionError("why was that an empty generator?")
            return result
        finally:
            self.invokeai_diffuser.model_forward_callback = self._unet_forward


    def prepare_latents_from_image(self, init_image, timestep, dtype, device, noise_func) -> (torch.FloatTensor, torch.FloatTensor):
        # can't quite use upstream StableDiffusionImg2ImgPipeline.prepare_latents
        # because we have our own noise function
        init_image = init_image.to(device=device, dtype=dtype)
        with torch.inference_mode():
            init_latent_dist = self.vae.encode(init_image).latent_dist
            init_latents = init_latent_dist.sample().to(dtype=dtype)  # FIXME: uses torch.randn. make reproducible!
        init_latents = 0.18215 * init_latents

        noise = noise_func(init_latents)
        noised_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        return noised_latents, init_latents

    def check_for_safety(self, output, dtype):
        with torch.inference_mode():
            screened_images, has_nsfw_concept = self.run_safety_checker(
                output.images, device=self._execution_device, dtype=dtype)
        return StableDiffusionPipelineOutput(screened_images, has_nsfw_concept)

    @torch.inference_mode()
    def get_learned_conditioning(self, c: List[List[str]], *, return_tokens=True, fragment_weights=None):
        """
        Compatibility function for ldm.models.diffusion.ddpm.LatentDiffusion.
        """
        return self.clip_embedder.encode(c, return_tokens=return_tokens, fragment_weights=fragment_weights)

    @property
    def cond_stage_model(self):
        warnings.warn("legacy compatibility layer", DeprecationWarning)
        return self.clip_embedder

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
