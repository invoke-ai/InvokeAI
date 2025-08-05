# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union

import diffusers
import numpy as np
import torch
from diffusers import AutoencoderKL  # Waiting for diffusers udpdate
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers
from diffusers.utils import USE_PEFT_BACKEND, logging
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
)

from invokeai.backend.bria.bria_utils import get_original_sigmas, get_t5_prompt_embeds, is_ng_none
from invokeai.backend.bria.controlnet_bria import BriaControlNetModel
from invokeai.backend.bria.pipeline_bria import BriaPipeline
from invokeai.backend.bria.transformer_bria import BriaTransformer2DModel
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState

logger = logging.get_logger(__name__)


class BriaControlNetPipeline(BriaPipeline):
    r"""
    Bria pipeline that supports optional ControlNet models.
    
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(  # EYAL - removed clip text encoder + tokenizer
        self,
        transformer: BriaTransformer2DModel,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, KarrasDiffusionSchedulers],
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        controlnet: BriaControlNetModel,
    ):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer
        )
        self.register_modules(controlnet=controlnet)

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_control(self, control_image, width, height, batch_size, num_images_per_prompt, device, control_mode):
        num_channels_latents = self.transformer.config.in_channels // 4
        control_image = self.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=self.vae.dtype,
        )
        height, width = control_image.shape[-2:]

        # vae encode
        control_image = self.vae.encode(control_image).latent_dist.sample()
        control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # pack
        height_control_image, width_control_image = control_image.shape[2:]
        control_image = self._pack_latents(
            control_image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_control_image,
            width_control_image,
        )

        # Here we ensure that `control_mode` has the same length as the control_image.
        if control_mode is not None:
            if not isinstance(control_mode, int):
                raise ValueError(" For `BriaControlNet`, `control_mode` should be an `int` or `None`")
            control_mode = torch.tensor(control_mode).to(device, dtype=torch.long)
            control_mode = control_mode.view(-1, 1).expand(control_image.shape[0], 1)

        return control_image, control_mode

    def prepare_multi_control(
        self, control_image, width, height, batch_size, num_images_per_prompt, device, control_mode
    ):
        num_channels_latents = self.transformer.config.in_channels // 4
        control_images = []
        for _, control_image_ in enumerate(control_image):
            control_image_ = self.prepare_image(
                image=control_image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.vae.dtype,
            )
            height, width = control_image_.shape[-2:]

            # vae encode
            control_image_ = self.vae.encode(control_image_).latent_dist.sample()
            control_image_ = (control_image_ - self.vae.config.shift_factor) * self.vae.config.scaling_factor

            # pack
            height_control_image, width_control_image = control_image_.shape[2:]
            control_image_ = self._pack_latents(
                control_image_,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height_control_image,
                width_control_image,
            )
            control_images.append(control_image_)

        control_image = control_images

        # Here we ensure that `control_mode` has the same length as the control_image.
        if isinstance(control_mode, list) and len(control_mode) != len(control_image):
            raise ValueError(
                "For Multi-ControlNet, `control_mode` must be a list of the same "
                + " length as the number of controlnets (control images) specified"
            )
        if not isinstance(control_mode, list):
            control_mode = [control_mode] * len(control_image)
        # set control mode
        control_modes = []
        for cmode in control_mode:
            if cmode is None:
                cmode = -1
            control_mode = torch.tensor(cmode).expand(control_images[0].shape[0]).to(device, dtype=torch.long)
            control_modes.append(control_mode)
        control_mode = control_modes

        return control_image, control_mode

    def get_controlnet_keep(self, timesteps, control_guidance_start, control_guidance_end):
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end, strict=False)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, BriaControlNetModel) else keeps)
        return controlnet_keep

    def get_control_start_end(self, control_guidance_start, control_guidance_end):
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = 1  # TODO - why is this 1?
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        return control_guidance_start, control_guidance_end

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: Optional[PipelineImageInput] = None,
        control_mode: Optional[Union[int, List[int]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        latent_image_ids: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        text_ids: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        step_callback: Callable[[PipelineIntermediateState], None] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
        Examples:
        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        control_guidance_start, control_guidance_end = self.get_control_start_end(
            control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end
        )

        # 1. Check inputs. Raise error if not correct
        callback_on_step_end_tensor_inputs = (
            ["latents"] if callback_on_step_end_tensor_inputs is None else callback_on_step_end_tensor_inputs
        )
        self.check_inputs(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        device = self._execution_device

        # 4. Prepare timesteps
        if (
            isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)
            and self.scheduler.config["use_dynamic_shifting"]
        ):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

            # Determine image sequence length
            if control_image is not None:
                if isinstance(control_image, list):
                    image_seq_len = control_image[0].shape[1]
                else:
                    image_seq_len = control_image.shape[1]
            else:
                # Use latents sequence length when no control image is provided
                image_seq_len = latents.shape[1]

            print(f"Using dynamic shift in pipeline with sequence length {image_seq_len}")

            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                timesteps=None,
                sigmas=sigmas,
                mu=mu,
            )
        else:
            # 5. Prepare timesteps
            sigmas = get_original_sigmas(
                num_train_timesteps=self.scheduler.config.num_train_timesteps, num_inference_steps=num_inference_steps
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, sigmas=sigmas
            )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Create tensor stating which controlnets to keep
        if control_image is not None:
            controlnet_keep = self.get_controlnet_keep(
                timesteps=timesteps,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
            )
        if text_ids is None:
            text_ids = torch.zeros(1, prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
            text_ids = text_ids.repeat(1, 1, 1)

        if diffusers.__version__ >= "0.32.0":
            latent_image_ids = latent_image_ids[0]
            text_ids = text_ids[0]

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # Init Invoke step callback
        step_callback(
            PipelineIntermediateState(
                step=0,
                order=1,
                total_steps=num_inference_steps,
                timestep=int(timesteps[0]),
                latents=latents.view(1, 64, 64, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 4, 128, 128),
            ),
        )

        # EYAL - added the CFG loop
        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # if type(self.scheduler) != FlowMatchEulerDiscreteScheduler:
                if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Handling ControlNet
                if control_image is not None:
                    if isinstance(controlnet_keep[i], list):
                        if isinstance(controlnet_conditioning_scale, list):
                            cond_scale = controlnet_conditioning_scale
                        else:
                            cond_scale = [
                                c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i], strict=False)
                            ]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep,
                        # guidance=guidance,
                        # pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )
                else:
                    controlnet_block_samples, controlnet_single_block_samples = None, None

                # This is predicts "v" from flow-matching
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    step_callback(
                        PipelineIntermediateState(
                            step=i + 1,
                            order=1,
                            total_steps=num_inference_steps,
                            timestep=int(t),
                            latents=self._unpack_latents(latents, height, width, self.vae_scale_factor),
                        ),
                    )

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


def encode_prompt(
    prompt: Union[str, List[str]],
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    max_sequence_length: int = 128,
    lora_scale: Optional[float] = None,
):
    r"""

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
    """
    device = device or torch.device("cuda")

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    # dynamically adjust the LoRA scale
    if text_encoder is not None and USE_PEFT_BACKEND:
        scale_lora_layers(text_encoder, lora_scale)

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    dtype = text_encoder.dtype if text_encoder is not None else torch.float32
    if prompt_embeds is None:
        prompt_embeds = get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        ).to(dtype=dtype)

    if negative_prompt_embeds is None:
        if not is_ng_none(negative_prompt):
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = get_t5_prompt_embeds(
                tokenizer,
                text_encoder,
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            ).to(dtype=dtype)
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

    if text_encoder is not None:
        if USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(text_encoder, lora_scale)


    return prompt_embeds, negative_prompt_embeds


def prepare_latents(
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
    latents: Optional[torch.FloatTensor] = None,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    vae_scale_factor = 16
    height = 2 * (int(height) // vae_scale_factor)
    width = 2 * (int(width) // vae_scale_factor)

    shape = (batch_size, num_channels_latents, height, width)

    if latents is not None:
        latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        return latents.to(device=device, dtype=dtype), latent_image_ids

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = _pack_latents(latents, batch_size, num_channels_latents, height, width)

    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

    return latents, latent_image_ids


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.repeat(batch_size, 1, 1, 1)
    latent_image_ids = latent_image_ids.reshape(
        batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents
