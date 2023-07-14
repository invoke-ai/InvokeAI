import copy
import torch
import inspect
from tqdm import tqdm
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator

from ...backend.model_management import BaseModelType, ModelType, SubModelType
from .baseinvocation import (BaseInvocation, BaseInvocationOutput,
                             InvocationConfig, InvocationContext)

from .model import UNetField, ClipField, VaeField, MainModelField, ModelInfo
from .compel import ConditioningField
from .latent import LatentsField, SAMPLER_NAME_VALUES, LatentsOutput, get_scheduler, build_latents_output

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
    #control: Union[ControlField, list[ControlField]] = Field(default=None, description="The control to use")
    #seamless:   bool = Field(default=False, description="Whether or not to generate an image that can tile without seams", )
    #seamless_axes: str = Field(default="", description="The axes to tile the image on, 'x' and/or 'y'")
    # fmt: on

    @validator("cfg_scale")
    def ge_one(cls, v):
        """validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError('cfg_scale must be greater than 1')
        else:
            if v < 1:
                raise ValueError('cfg_scale must be greater than 1')
        return v

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["latents"],
                "type_hints": {
                  "model": "model",
                  # "cfg_scale": "float",
                  "cfg_scale": "number"
                }
            },
        }

    # based on
    # https://github.com/huggingface/diffusers/blob/3ebbaf7c96801271f9e6c21400033b6aa5ffcf29/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L375
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.services.latents.get(self.noise.latents_name)

        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        prompt_embeds = positive_cond_data.conditionings[0].embeds
        pooled_prompt_embeds = positive_cond_data.conditionings[0].pooled_embeds

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        negative_prompt_embeds = negative_cond_data.conditionings[0].embeds
        negative_pooled_prompt_embeds = negative_cond_data.conditionings[0].pooled_embeds

        add_time_ids = torch.tensor([(latents.shape[2] * 8, latents.shape[3] * 8) + (0, 0) + (latents.shape[2] * 8, latents.shape[3] * 8)])

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
        )

        scheduler.set_timesteps(self.steps)
        timesteps = scheduler.timesteps

        latents = latents * scheduler.init_noise_sigma

        extra_step_kwargs = dict()
        if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
            extra_step_kwargs.update(
                eta=0.0,
            )

        #################

        unet_info = context.services.model_manager.get_model(
            **self.unet.unet.dict()
        )
        do_classifier_free_guidance = True
        cross_attention_kwargs = None
        with unet_info as unet:

            if not context.services.configuration.sequential_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_text_embeds = add_text_embeds.to(device=unet.device, dtype=unet.dtype)
                add_time_ids = add_time_ids.to(device=unet.device, dtype=unet.dtype)
                latents = latents.to(device=unet.device, dtype=unet.dtype)

                num_warmup_steps = len(timesteps) - self.steps * scheduler.order
                with tqdm(total=self.steps) as progress_bar:
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
                            #del noise_pred_uncond
                            #del noise_pred_text

                        #if do_classifier_free_guidance and guidance_rescale > 0.0:
                        #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                            progress_bar.update()
                            #if callback is not None and i % callback_steps == 0:
                            #    callback(i, t, latents)
            else:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                prompt_embeds = prompt_embeds.to(device=unet.device, dtype=unet.dtype)
                add_time_ids = add_time_ids.to(device=unet.device, dtype=unet.dtype)
                latents = latents.to(device=unet.device, dtype=unet.dtype)

                num_warmup_steps = len(timesteps) - self.steps * scheduler.order
                with tqdm(total=self.steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        #latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                        latent_model_input = scheduler.scale_model_input(latents, t)

                        #import gc
                        #gc.collect()
                        #torch.cuda.empty_cache()

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

                        #del noise_pred_text
                        #del noise_pred_uncond
                        #import gc
                        #gc.collect()
                        #torch.cuda.empty_cache()

                        #if do_classifier_free_guidance and guidance_rescale > 0.0:
                        #    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        #    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        #del noise_pred
                        #import gc
                        #gc.collect()
                        #torch.cuda.empty_cache()

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                            progress_bar.update()
                            #if callback is not None and i % callback_steps == 0:
                            #    callback(i, t, latents)



        #################

        torch.cuda.empty_cache()

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.save(name, latents)
        return build_latents_output(latents_name=name, latents=latents)
