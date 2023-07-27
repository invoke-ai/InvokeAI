import torch
import inspect
from tqdm import tqdm
from typing import List, Literal, Optional, Union

from pydantic import Field, validator

from ...backend.model_management import ModelType, SubModelType
from invokeai.app.util.step_callback import stable_diffusion_xl_step_callback
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext

from .model import UNetField, ClipField, VaeField, MainModelField, ModelInfo
from .compel import ConditioningField
from .latent import LatentsField, SAMPLER_NAME_VALUES, LatentsOutput, get_scheduler, build_latents_output


class SDXLModelLoaderOutput(BaseInvocationOutput):
    """SDXL base model loader output"""

    # fmt: off
    type: Literal["sdxl_model_loader_output"] = "sdxl_model_loader_output"

    unet: UNetField = Field(default=None, description="UNet submodel")
    clip: ClipField = Field(default=None, description="Tokenizer and text_encoder submodels")
    clip2: ClipField = Field(default=None, description="Tokenizer and text_encoder submodels")
    vae: VaeField = Field(default=None, description="Vae submodel")
    # fmt: on


class SDXLRefinerModelLoaderOutput(BaseInvocationOutput):
    """SDXL refiner model loader output"""

    # fmt: off
    type: Literal["sdxl_refiner_model_loader_output"] = "sdxl_refiner_model_loader_output"
    unet: UNetField = Field(default=None, description="UNet submodel")
    clip2: ClipField = Field(default=None, description="Tokenizer and text_encoder submodels")
    vae: VaeField = Field(default=None, description="Vae submodel")
    # fmt: on
    # fmt: on


class SDXLModelLoaderInvocation(BaseInvocation):
    """Loads an sdxl base model, outputting its submodels."""

    type: Literal["sdxl_model_loader"] = "sdxl_model_loader"

    model: MainModelField = Field(description="The model to load")
    # TODO: precision?

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "SDXL Model Loader",
                "tags": ["model", "loader", "sdxl"],
                "type_hints": {"model": "model"},
            },
        }

    def invoke(self, context: InvocationContext) -> SDXLModelLoaderOutput:
        base_model = self.model.base_model
        model_name = self.model.model_name
        model_type = ModelType.Main

        # TODO: not found exceptions
        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        return SDXLModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder,
                ),
                loras=[],
                skipped_layers=0,
            ),
            clip2=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer2,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder2,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )


class SDXLRefinerModelLoaderInvocation(BaseInvocation):
    """Loads an sdxl refiner model, outputting its submodels."""

    type: Literal["sdxl_refiner_model_loader"] = "sdxl_refiner_model_loader"

    model: MainModelField = Field(description="The model to load")
    # TODO: precision?

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "SDXL Refiner Model Loader",
                "tags": ["model", "loader", "sdxl_refiner"],
                "type_hints": {"model": "refiner_model"},
            },
        }

    def invoke(self, context: InvocationContext) -> SDXLRefinerModelLoaderOutput:
        base_model = self.model.base_model
        model_name = self.model.model_name
        model_type = ModelType.Main

        # TODO: not found exceptions
        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        return SDXLRefinerModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip2=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer2,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder2,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )


# Text to image
class SDXLTextToLatentsInvocation(BaseInvocation):
    """Generates latents from conditionings."""

    type: Literal["t2l_sdxl"] = "t2l_sdxl"

    # Inputs
    # fmt: off
    positive_conditioning: Optional[ConditioningField] = Field(description="Positive conditioning for generation")
    negative_conditioning: Optional[ConditioningField] = Field(description="Negative conditioning for generation")
    noise: Optional[LatentsField] = Field(description="The noise to use")
    steps:       int = Field(default=10, gt=0, description="The number of steps to use to generate the image")
    cfg_scale: Union[float, List[float]] = Field(default=7.5, ge=1, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )
    scheduler: SAMPLER_NAME_VALUES = Field(default="euler", description="The scheduler to use" )
    unet: UNetField = Field(default=None, description="UNet submodel")
    denoising_end: float = Field(default=1.0, gt=0, le=1, description="")
    # control: Union[ControlField, list[ControlField]] = Field(default=None, description="The control to use")
    # seamless:   bool = Field(default=False, description="Whether or not to generate an image that can tile without seams", )
    # seamless_axes: str = Field(default="", description="The axes to tile the image on, 'x' and/or 'y'")
    # fmt: on

    @validator("cfg_scale")
    def ge_one(cls, v):
        """validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError("cfg_scale must be greater than 1")
        else:
            if v < 1:
                raise ValueError("cfg_scale must be greater than 1")
        return v

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "SDXL Text To Latents",
                "tags": ["latents"],
                "type_hints": {
                    "model": "model",
                    # "cfg_scale": "float",
                    "cfg_scale": "number",
                },
            },
        }

    def dispatch_progress(
        self,
        context: InvocationContext,
        source_node_id: str,
        sample,
        step,
        total_steps,
    ) -> None:
        stable_diffusion_xl_step_callback(
            context=context,
            node=self.dict(),
            source_node_id=source_node_id,
            sample=sample,
            step=step,
            total_steps=total_steps,
        )

    # based on
    # https://github.com/huggingface/diffusers/blob/3ebbaf7c96801271f9e6c21400033b6aa5ffcf29/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L375
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
        source_node_id = graph_execution_state.prepared_source_mapping[self.id]
        latents = context.services.latents.get(self.noise.latents_name)

        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        prompt_embeds = positive_cond_data.conditionings[0].embeds
        pooled_prompt_embeds = positive_cond_data.conditionings[0].pooled_embeds
        add_time_ids = positive_cond_data.conditionings[0].add_time_ids

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        negative_prompt_embeds = negative_cond_data.conditionings[0].embeds
        negative_pooled_prompt_embeds = negative_cond_data.conditionings[0].pooled_embeds
        add_neg_time_ids = negative_cond_data.conditionings[0].add_time_ids

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
        )

        num_inference_steps = self.steps
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps

        latents = latents * scheduler.init_noise_sigma

        unet_info = context.services.model_manager.get_model(**self.unet.unet.dict(), context=context)
        do_classifier_free_guidance = True
        cross_attention_kwargs = None
        with unet_info as unet:
            extra_step_kwargs = dict()
            if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
                extra_step_kwargs.update(
                    eta=0.0,
                )
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                extra_step_kwargs.update(
                    generator=torch.Generator(device=unet.device).manual_seed(0),
                )

            num_warmup_steps = len(timesteps) - self.steps * scheduler.order

            # apply denoising_end
            skipped_final_steps = int(round((1 - self.denoising_end) * self.steps))
            num_inference_steps = num_inference_steps - skipped_final_steps
            timesteps = timesteps[: num_warmup_steps + scheduler.order * num_inference_steps]

            if not context.services.configuration.sequential_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_text_embeds = add_text_embeds.to(device=unet.device, dtype=unet.dtype)
                add_time_ids = add_time_ids.to(device=unet.device, dtype=unet.dtype)
                latents = latents.to(device=unet.device, dtype=unet.dtype)

                with tqdm(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                            # del noise_pred_uncond
                            # del noise_pred_text

                        # if do_classifier_free_guidance and guidance_rescale > 0.0:
                        #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                            progress_bar.update()
                            self.dispatch_progress(context, source_node_id, latents, i, num_inference_steps)
                            # if callback is not None and i % callback_steps == 0:
                            #    callback(i, t, latents)
            else:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_neg_time_ids = add_neg_time_ids.to(device=unet.device, dtype=unet.dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                prompt_embeds = prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_time_ids = add_time_ids.to(device=unet.device, dtype=unet.dtype)
                latents = latents.to(device=unet.device, dtype=unet.dtype)

                with tqdm(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        latent_model_input = scheduler.scale_model_input(latents, t)

                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()

                        # predict the noise residual

                        added_cond_kwargs = {"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids}
                        noise_pred_uncond = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=negative_prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                        noise_pred_text = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

                        # del noise_pred_text
                        # del noise_pred_uncond
                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()

                        # if do_classifier_free_guidance and guidance_rescale > 0.0:
                        #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        # del noise_pred
                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                            progress_bar.update()
                            self.dispatch_progress(context, source_node_id, latents, i, num_inference_steps)
                            # if callback is not None and i % callback_steps == 0:
                            #    callback(i, t, latents)

        #################

        latents = latents.to("cpu")
        torch.cuda.empty_cache()

        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, latents)
        return build_latents_output(latents_name=name, latents=latents)


class SDXLLatentsToLatentsInvocation(BaseInvocation):
    """Generates latents from conditionings."""

    type: Literal["l2l_sdxl"] = "l2l_sdxl"

    # Inputs
    # fmt: off
    positive_conditioning: Optional[ConditioningField] = Field(description="Positive conditioning for generation")
    negative_conditioning: Optional[ConditioningField] = Field(description="Negative conditioning for generation")
    noise: Optional[LatentsField] = Field(description="The noise to use")
    steps:       int = Field(default=10, gt=0, description="The number of steps to use to generate the image")
    cfg_scale: Union[float, List[float]] = Field(default=7.5, ge=1, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )
    scheduler: SAMPLER_NAME_VALUES = Field(default="euler", description="The scheduler to use" )
    unet: UNetField = Field(default=None, description="UNet submodel")
    latents: Optional[LatentsField] = Field(description="Initial latents")

    denoising_start: float = Field(default=0.0, ge=0, le=1, description="")
    denoising_end: float = Field(default=1.0, ge=0, le=1, description="")

    # control: Union[ControlField, list[ControlField]] = Field(default=None, description="The control to use")
    # seamless:   bool = Field(default=False, description="Whether or not to generate an image that can tile without seams", )
    # seamless_axes: str = Field(default="", description="The axes to tile the image on, 'x' and/or 'y'")
    # fmt: on

    @validator("cfg_scale")
    def ge_one(cls, v):
        """validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError("cfg_scale must be greater than 1")
        else:
            if v < 1:
                raise ValueError("cfg_scale must be greater than 1")
        return v

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "SDXL Latents to Latents",
                "tags": ["latents"],
                "type_hints": {
                    "model": "model",
                    # "cfg_scale": "float",
                    "cfg_scale": "number",
                },
            },
        }

    def dispatch_progress(
        self,
        context: InvocationContext,
        source_node_id: str,
        sample,
        step,
        total_steps,
    ) -> None:
        stable_diffusion_xl_step_callback(
            context=context,
            node=self.dict(),
            source_node_id=source_node_id,
            sample=sample,
            step=step,
            total_steps=total_steps,
        )

    # based on
    # https://github.com/huggingface/diffusers/blob/3ebbaf7c96801271f9e6c21400033b6aa5ffcf29/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L375
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
        source_node_id = graph_execution_state.prepared_source_mapping[self.id]
        latents = context.services.latents.get(self.latents.latents_name)

        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        prompt_embeds = positive_cond_data.conditionings[0].embeds
        pooled_prompt_embeds = positive_cond_data.conditionings[0].pooled_embeds
        add_time_ids = positive_cond_data.conditionings[0].add_time_ids

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        negative_prompt_embeds = negative_cond_data.conditionings[0].embeds
        negative_pooled_prompt_embeds = negative_cond_data.conditionings[0].pooled_embeds
        add_neg_time_ids = negative_cond_data.conditionings[0].add_time_ids

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
        )

        # apply denoising_start
        num_inference_steps = self.steps
        scheduler.set_timesteps(num_inference_steps)

        t_start = int(round(self.denoising_start * num_inference_steps))
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        num_inference_steps = num_inference_steps - t_start

        # apply noise(if provided)
        if self.noise is not None and timesteps.shape[0] > 0:
            noise = context.services.latents.get(self.noise.latents_name)
            latents = scheduler.add_noise(latents, noise, timesteps[:1])
            del noise

        unet_info = context.services.model_manager.get_model(
            **self.unet.unet.dict(),
            context=context,
        )
        do_classifier_free_guidance = True
        cross_attention_kwargs = None
        with unet_info as unet:
            # apply scheduler extra args
            extra_step_kwargs = dict()
            if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
                extra_step_kwargs.update(
                    eta=0.0,
                )
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                extra_step_kwargs.update(
                    generator=torch.Generator(device=unet.device).manual_seed(0),
                )

            num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

            # apply denoising_end
            skipped_final_steps = int(round((1 - self.denoising_end) * self.steps))
            num_inference_steps = num_inference_steps - skipped_final_steps
            timesteps = timesteps[: num_warmup_steps + scheduler.order * num_inference_steps]

            if not context.services.configuration.sequential_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_text_embeds = add_text_embeds.to(device=unet.device, dtype=unet.dtype)
                add_time_ids = add_time_ids.to(device=unet.device, dtype=unet.dtype)
                latents = latents.to(device=unet.device, dtype=unet.dtype)

                with tqdm(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                            # del noise_pred_uncond
                            # del noise_pred_text

                        # if do_classifier_free_guidance and guidance_rescale > 0.0:
                        #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                            progress_bar.update()
                            self.dispatch_progress(context, source_node_id, latents, i, num_inference_steps)
                            # if callback is not None and i % callback_steps == 0:
                            #    callback(i, t, latents)
            else:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_neg_time_ids = add_neg_time_ids.to(device=unet.device, dtype=unet.dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                prompt_embeds = prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_time_ids = add_time_ids.to(device=unet.device, dtype=unet.dtype)
                latents = latents.to(device=unet.device, dtype=unet.dtype)

                with tqdm(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        latent_model_input = scheduler.scale_model_input(latents, t)

                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()

                        # predict the noise residual

                        added_cond_kwargs = {"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_time_ids}
                        noise_pred_uncond = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=negative_prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                        noise_pred_text = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

                        # del noise_pred_text
                        # del noise_pred_uncond
                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()

                        # if do_classifier_free_guidance and guidance_rescale > 0.0:
                        #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        # del noise_pred
                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                            progress_bar.update()
                            self.dispatch_progress(context, source_node_id, latents, i, num_inference_steps)
                            # if callback is not None and i % callback_steps == 0:
                            #    callback(i, t, latents)

        #################

        latents = latents.to("cpu")
        torch.cuda.empty_cache()

        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, latents)
        return build_latents_output(latents_name=name, latents=latents)
