from __future__ import annotations

import secrets
import warnings
from dataclasses import dataclass
from typing import List, Optional, Union, Callable

import PIL.Image
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess, \
    StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
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
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor)

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
            **extra_step_kwargs):
        if run_id is None:
            run_id = secrets.token_urlsafe(self.ID_LENGTH)

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
                                    i, **extra_step_kwargs)
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
             step_index:int | None = None,
             **extra_step_kwargs):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]

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
        return self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs)

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
        latents_dtype = text_embeddings.dtype
        batch_size = 1
        num_images_per_prompt = 1

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image.convert('RGB'))

        img2img_pipeline = StableDiffusionImg2ImgPipeline(**self.components)
        img2img_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = img2img_pipeline.get_timesteps(num_inference_steps, strength, device=device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents_from_image(init_image, latent_timestep, latents_dtype, device, noise_func)

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

    def prepare_latents_from_image(self, init_image, timestep, dtype, device, noise_func) -> torch.FloatTensor:
        # can't quite use upstream StableDiffusionImg2ImgPipeline.prepare_latents
        # because we have our own noise function
        init_image = init_image.to(device=device, dtype=dtype)
        with torch.inference_mode():
            init_latent_dist = self.vae.encode(init_image).latent_dist
            init_latents = init_latent_dist.sample()  # FIXME: uses torch.randn. make reproducible!
        init_latents = 0.18215 * init_latents

        noise = noise_func(init_latents)

        return self.scheduler.add_noise(init_latents, noise, timestep)

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
