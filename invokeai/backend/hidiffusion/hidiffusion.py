import importlib.resources

import torch
import math
from typing import Type, Dict, Any, Tuple, Callable, Optional, Union, List
import torch.nn.functional as F
from .utils import isinstance_str
import diffusers
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, deprecate
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, apply_freeu
from diffusers.pipelines import auto_pipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.models import ControlNetModel
from diffusers.models.attention import _chunked_feed_forward
import warnings

diffusers_version = diffusers.__version__
if diffusers_version < "0.27.0":
    from diffusers.models.unet_2d_condition import UNet2DConditionOutput
    old_diffusers = True
else:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
    old_diffusers = False

def sd15_hidiffusion_key():
    modified_key = dict()
    modified_key['down_module_key'] = ['down_blocks.0.downsamplers.0.conv']
    modified_key['down_module_key_extra'] = ['down_blocks.1']
    modified_key['up_module_key'] = ['up_blocks.2.upsamplers.0.conv']
    modified_key['up_module_key_extra'] = ['up_blocks.2']
    modified_key['windown_attn_module_key'] = ['down_blocks.0.attentions.0.transformer_blocks.0',
                               'down_blocks.0.attentions.1.transformer_blocks.0',
                               'up_blocks.3.attentions.0.transformer_blocks.0',
                               'up_blocks.3.attentions.1.transformer_blocks.0',
                               'up_blocks.3.attentions.2.transformer_blocks.0']
    return modified_key

def sdxl_hidiffusion_key():
    modified_key = dict()
    modified_key['down_module_key'] = ['down_blocks.1']
    modified_key['down_module_key_extra'] = ['down_blocks.1.downsamplers.0.conv']
    modified_key['up_module_key'] = ['up_blocks.1']
    modified_key['up_module_key_extra'] = ['up_blocks.0.upsamplers.0.conv']
    modified_key['windown_attn_module_key'] = ['down_blocks.1.attentions.0.transformer_blocks.0',
                               'down_blocks.1.attentions.0.transformer_blocks.1',
                               'down_blocks.1.attentions.1.transformer_blocks.0',
                               'down_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.0.transformer_blocks.0',
                               'up_blocks.1.attentions.0.transformer_blocks.1',
                               'up_blocks.1.attentions.1.transformer_blocks.0',
                               'up_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.2.transformer_blocks.0',
                               'up_blocks.1.attentions.2.transformer_blocks.1']

    return modified_key


def sdxl_turbo_hidiffusion_key():
    modified_key = dict()
    modified_key['down_module_key'] = ['down_blocks.1']
    modified_key['up_module_key'] = ['up_blocks.1']
    modified_key['windown_attn_module_key'] = ['down_blocks.1.attentions.0.transformer_blocks.0',
                               'down_blocks.1.attentions.0.transformer_blocks.1',
                               'down_blocks.1.attentions.1.transformer_blocks.0',
                               'down_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.0.transformer_blocks.0',
                               'up_blocks.1.attentions.0.transformer_blocks.1',
                               'up_blocks.1.attentions.1.transformer_blocks.0',
                               'up_blocks.1.attentions.1.transformer_blocks.1',
                               'up_blocks.1.attentions.2.transformer_blocks.0',
                               'up_blocks.1.attentions.2.transformer_blocks.1']

    return modified_key

# supported official model. If you use non-official model based on the following models/pipelines, hidiffusion will automatically select the best strategy to fit it.
supported_official_model = [
    'runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-1-base',
    'stabilityai/stable-diffusion-xl-base-1.0', 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
    'stabilityai/sdxl-turbo'
]


# T1_ratio: see T1 introduced in the main paper. T1 = number_inference_step * T1_ratio. A higher T1_ratio can better mitigate object duplication. We set T1_ratio=0.4 by default. You'd better adjust it to fit your prompt. Only active when apply_raunet=True.
# T2_ratio: see T2 introduced in the appendix, used in extreme resolution image generation. T2 = number_inference_step * T2_ratio. A higher T2_ratio can better mitigate object duplication. Only active when apply_raunet=True
switching_threshold_ratio_dict = {
    'sd15_1024': {'T1_ratio': 0.4, 'T2_ratio': 0.0},
    'sd15_2048': {'T1_ratio': 0.7, 'T2_ratio': 0.3},
    'sdxl_2048': {'T1_ratio': 0.4, 'T2_ratio': 0.0},
    'sdxl_4096': {'T1_ratio': 0.7, 'T2_ratio': 0.3},
    'sdxl_turbo_1024': {'T1_ratio': 0.5, 'T2_ratio': 0.0},
}

text_to_img_controlnet_switching_threshold_ratio_dict = {
    'sdxl_2048': {'T1_ratio': 0.5, 'T2_ratio': 0.0},
}
controlnet_apply_steps_rate = 0.6

is_aggressive_raunet = True
aggressive_step = 8

inpainting_is_aggressive_raunet = False
playground_is_aggressive_raunet = False


with importlib.resources.open_text(
        f"{__package__}.sd_module_key", "sd15_module_key.txt", encoding="utf-8") as f:
    sd15_module_key = f.read().splitlines()

with importlib.resources.open_text(
        f"{__package__}.sd_module_key", "sdxl_module_key.txt", encoding="utf-8") as f:
    sdxl_module_key = f.read().splitlines()


def _get_max_timesteps(info_dict: dict) -> int:
    """
    Helper function to get the maximum number of timesteps from a pipeline.
    """
    pipeline = info_dict['pipeline']
    if hasattr(pipeline, '_num_timesteps'):
        return pipeline._num_timesteps
    else:
        return len(pipeline.scheduler.timesteps)


def make_diffusers_sdxl_controlnet_ppl(block_class):

    class sdxl_controlnet_ppl(block_class):
        # Save for unpatching later
        _parent = block_class

        @torch.no_grad()
        def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            image: PipelineImageInput = None,
            control_image: PipelineImageInput = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            strength: float = 0.8,
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
            guess_mode: bool = False,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
        ):
            r"""
            Function invoked when calling the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                    instead.
                prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                    used in both text-encoders
                image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                        `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                    The initial image will be used as the starting point for the image generation process. Can also accept
                    image latents as `image`, if passing latents directly, it will not be encoded again.
                control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                        `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                    The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                    the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                    also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                    height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                    specified in init, images must be passed as a list such that each element of the list can be correctly
                    batched for input to a single controlnet.
                height (`int`, *optional*, defaults to the size of control_image):
                    The height in pixels of the generated image. Anything below 512 pixels won't work well for
                    [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                    and checkpoints that are not specifically fine-tuned on low resolutions.
                width (`int`, *optional*, defaults to the size of control_image):
                    The width in pixels of the generated image. Anything below 512 pixels won't work well for
                    [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                    and checkpoints that are not specifically fine-tuned on low resolutions.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                strength (`float`, *optional*, defaults to 0.3):
                    Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                    will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                    denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                    be maximum and the denoising process will run for the full number of iterations specified in
                    `num_inference_steps`.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                    `guidance_scale` is defined as `w` of equation 2. of [Imagen
                    Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                    1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                    usually at the expense of lower image quality.
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                negative_prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                    `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                eta (`float`, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                    [`schedulers.DDIMScheduler`], will be ignored for others.
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
                pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                    If not provided, pooled text embeddings will be generated from `prompt` input argument.
                negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                    Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                    input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generate image. Choose between
                    [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                    The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                    to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                    corresponding scale as a list.
                guess_mode (`bool`, *optional*, defaults to `False`):
                    In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                    you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
                control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                    The percentage of total steps at which the controlnet starts applying.
                control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                    The percentage of total steps at which the controlnet stops applying.
                original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                    `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                    explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                    `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                    `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                    `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    For most cases, `target_size` should be set to the desired height and width of the generated image. If
                    not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                    section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                    To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                    To negatively condition the generation process based on a target image resolution. It should be as same
                    as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                    information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
                aesthetic_score (`float`, *optional*, defaults to 6.0):
                    Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                    Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
                negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                    Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                    simulate an aesthetic score of the generated image by influencing the negative text condition.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, *optional*):
                    A function that calls at the end of each denoising steps during the inference. The function is called
                    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                    `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeine class.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple`
                containing the output images.
            """

            # convert image to control_image to fit sdxl_controlnet ppl.
            if control_image is None:
                control_image = image
                image = None
                self.info['text_to_img_controlnet'] = True
            else:
                self.info['text_to_img_controlnet'] = False

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )

            controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
                control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
                control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
                mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
                control_guidance_start, control_guidance_end = (
                    mult * [control_guidance_start],
                    mult * [control_guidance_end],
                )

            # 1. Check inputs. Raise error if not correct
            if image is not None:
                # image-to-image controlnet
                if old_diffusers:
                    self.check_inputs(
                        prompt,
                        prompt_2,
                        control_image,
                        strength,
                        num_inference_steps,
                        callback_steps,
                        negative_prompt,
                        negative_prompt_2,
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        controlnet_conditioning_scale,
                        control_guidance_start,
                        control_guidance_end,
                        callback_on_step_end_tensor_inputs,
                    )
                else:
                    self.check_inputs(
                        prompt,
                        prompt_2,
                        control_image,
                        strength,
                        num_inference_steps,
                        callback_steps,
                        negative_prompt,
                        negative_prompt_2,
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        None,
                        None,
                        controlnet_conditioning_scale,
                        control_guidance_start,
                        control_guidance_end,
                        callback_on_step_end_tensor_inputs,
                    )
            else:
                # text-to-image controlnet
                if old_diffusers:
                    self.check_inputs(
                        prompt,
                        prompt_2,
                        control_image,
                        callback_steps,
                        negative_prompt,
                        negative_prompt_2,
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        controlnet_conditioning_scale,
                        control_guidance_start,
                        control_guidance_end,
                        callback_on_step_end_tensor_inputs,
                    )
                else:
                    self.check_inputs(
                        prompt,
                        prompt_2,
                        control_image,
                        callback_steps,
                        negative_prompt,
                        negative_prompt_2,
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        None,
                        None,
                        negative_pooled_prompt_embeds,
                        controlnet_conditioning_scale,
                        control_guidance_start,
                        control_guidance_end,
                        callback_on_step_end_tensor_inputs,
                    )

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt,
                prompt_2,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                negative_prompt_2,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )

            # 4. Prepare image and controlnet_conditioning_image
            if image is not None:
                image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
                if isinstance(controlnet, ControlNetModel):
                    control_image = self.prepare_control_image(
                        image=control_image,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    height, width = control_image.shape[-2:]
                elif isinstance(controlnet, MultiControlNetModel):
                    control_images = []

                    for control_image_ in control_image:
                        control_image_ = self.prepare_control_image(
                            image=control_image_,
                            width=width,
                            height=height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=controlnet.dtype,
                            do_classifier_free_guidance=self.do_classifier_free_guidance,
                            guess_mode=guess_mode,
                        )

                        control_images.append(control_image_)

                    control_image = control_images
                    height, width = control_image[0].shape[-2:]
                else:
                    assert False
            else:
                if isinstance(controlnet, ControlNetModel):
                    control_image = self.prepare_image(
                        image=control_image,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )
                    height, width = control_image.shape[-2:]
                elif isinstance(controlnet, MultiControlNetModel):
                    images = []

                    for image_ in control_image:
                        image_ = self.prepare_image(
                            image=image_,
                            width=width,
                            height=height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=controlnet.dtype,
                            do_classifier_free_guidance=self.do_classifier_free_guidance,
                            guess_mode=guess_mode,
                        )

                        images.append(image_)

                    control_image = images
                    height, width = image[0].shape[-2:]
                else:
                    assert False
            # 5. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            if image is not None:
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
                latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            else:
                timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            # 6. Prepare latent variables
            if image is not None:
                # image-to-image controlnet
                latents = self.prepare_latents(
                    image,
                    latent_timestep,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    True,
                )
            else:
                # text-to-image controlnet
                num_channels_latents = self.unet.config.in_channels
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )
                # num_channels_latents = self.unet.config.in_channels
                # shape = (batch_size * num_images_per_prompt, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
                # if isinstance(generator, list) and len(generator) != batch_size:
                #     raise ValueError(
                #         f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                #         f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                #     )

                # if latents is None:
                #     latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                # else:
                #     latents = latents.to(device)

                # # scale the initial noise by the standard deviation required by the scheduler
                # latents = latents * self.scheduler.init_noise_sigma

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7.1 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

            # 7.2 Prepare added time ids & embeddings
            if image is not None:
                if isinstance(control_image, list):
                    original_size = original_size or control_image[0].shape[-2:]
                else:
                    original_size = original_size or control_image.shape[-2:]
                target_size = target_size or (height, width)

                if negative_original_size is None:
                    negative_original_size = original_size
                if negative_target_size is None:
                    negative_target_size = target_size
                add_text_embeds = pooled_prompt_embeds

                if self.text_encoder_2 is None:
                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                else:
                    text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

                add_time_ids, add_neg_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    aesthetic_score,
                    negative_aesthetic_score,
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
                add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                add_text_embeds = add_text_embeds.to(device)
                add_time_ids = add_time_ids.to(device)
            else:
                if isinstance(control_image, list):
                    original_size = original_size or control_image[0].shape[-2:]
                else:
                    original_size = original_size or control_image.shape[-2:]
                target_size = target_size or (height, width)

                add_text_embeds = pooled_prompt_embeds
                if self.text_encoder_2 is None:
                    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
                else:
                    text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

                add_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )

                if negative_original_size is not None and negative_target_size is not None:
                    negative_add_time_ids = self._get_add_time_ids(
                        negative_original_size,
                        negative_crops_coords_top_left,
                        negative_target_size,
                        dtype=prompt_embeds.dtype,
                        text_encoder_projection_dim=text_encoder_projection_dim,
                    )
                else:
                    negative_add_time_ids = add_time_ids

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                add_text_embeds = add_text_embeds.to(device)
                add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    # controlnet(s) inference
                    if guess_mode and self.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                        controlnet_added_cond_kwargs = {
                            "text_embeds": add_text_embeds.chunk(2)[1],
                            "time_ids": add_time_ids.chunk(2)[1],
                        }
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds
                        controlnet_added_cond_kwargs = added_cond_kwargs

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]


                    if i < controlnet_apply_steps_rate * num_inference_steps:

                        original_h, original_w = (128,128)
                        _, _, model_input_h, model_input_w = control_model_input.shape
                        downsample_factor = max(model_input_h/original_h, model_input_w/original_w)
                        downsample_size = (int(model_input_h//downsample_factor)//8*8, int(model_input_w//downsample_factor)//8*8)

                        # original_pixel_h, original_pixel_w = (1024,1024)
                        # _, _, pixel_h, pixel_w = control_image.shape
                        # downsample_pixel_factor = max(pixel_h/original_pixel_h, pixel_w/original_pixel_w)
                        # downsample_pixel_size = (int(pixel_h//downsample_pixel_factor)//8*8, int(pixel_w//downsample_pixel_factor)//8*8)
                        downsample_pixel_size = [downsample_size[0]*8, downsample_size[1]*8]

                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            F.interpolate(control_model_input, downsample_size),
                            # control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=F.interpolate(control_image, downsample_pixel_size),
                            # controlnet_cond=control_image,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            added_cond_kwargs=controlnet_added_cond_kwargs,
                            return_dict=False,
                        )

                    if guess_mode and self.do_classifier_free_guidance:
                        # Infered ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # predict the noise residual
                    if i < controlnet_apply_steps_rate * num_inference_steps:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=None,
                            mid_block_additional_residual=None,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            # If we do sequential model offloading, let's offload unet and controlnet
            # manually for max memory savings
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.unet.to("cpu")
                self.controlnet.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not output_type == "latent":
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                if needs_upcasting:
                    self.upcast_vae()
                    latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
            else:
                image = latents
                return StableDiffusionXLPipelineOutput(images=image)

            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return StableDiffusionXLPipelineOutput(images=image)

    # let's be nice and not change the __name__ of the pipeline class
    # this messes up some important pipeline detection code in dgenerate.

    sdxl_controlnet_ppl.__name__ = block_class.__name__

    return sdxl_controlnet_ppl


def make_diffusers_unet_2d_condition(block_class):

    class unet_2d_condition(block_class):
        # Save for unpatching later
        _parent = block_class
        def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:
            r"""
            The [`UNet2DConditionModel`] forward method.

            Args:
                sample (`torch.FloatTensor`):
                    The noisy input tensor with the following shape `(batch, channel, height, width)`.
                timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.FloatTensor`):
                    The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                    Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                    through the `self.time_embedding` layer to obtain the timestep embeddings.
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                added_cond_kwargs: (`dict`, *optional*):
                    A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                    are passed along to the UNet blocks.
                down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                    A tuple of tensors that if specified are added to the residuals of down unet blocks.
                mid_block_additional_residual: (`torch.Tensor`, *optional*):
                    A tensor that if specified is added to the residual of the middle unet block.
                encoder_attention_mask (`torch.Tensor`):
                    A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                    `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                    which adds large negative values to the attention scores corresponding to "discard" tokens.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                    tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
                added_cond_kwargs: (`dict`, *optional*):
                    A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                    are passed along to the UNet blocks.
                down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                    additional residuals to be added to UNet long skip connections from down blocks to up blocks for
                    example from ControlNet side model(s)
                mid_block_additional_residual (`torch.Tensor`, *optional*):
                    additional residual to be added to UNet mid block output, for example from ControlNet side model
                down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                    additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)

            Returns:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                    a `tuple` is returned where the first element is the sample tensor.
            """
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            for dim in sample.shape[-2:]:
                if dim % default_overall_up_factor != 0:
                    # Forward upsample size to force interpolation output size.
                    forward_upsample_size = True
                    break

            # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
            # expects mask of shape:
            #   [batch, key_tokens]
            # adds singleton query_tokens dimension:
            #   [batch,                    1, key_tokens]
            # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
            #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
            #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
            if attention_mask is not None:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None:
                encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            # 0. center input if necessary
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                    # `Timesteps` does not contain any weights and will always return f32 tensors
                    # there might be better ways to encapsulate this.
                    class_labels = class_labels.to(dtype=sample.dtype)

                class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_image":
                # Kandinsky 2.1 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )

                image_embs = added_cond_kwargs.get("image_embeds")
                text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
                aug_emb = self.add_embedding(text_embs, image_embs)
            elif self.config.addition_embed_type == "text_time":
                # SDXL - style
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)
            elif self.config.addition_embed_type == "image":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                    )
                image_embs = added_cond_kwargs.get("image_embeds")
                aug_emb = self.add_embedding(image_embs)
            elif self.config.addition_embed_type == "image_hint":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                    )
                image_embs = added_cond_kwargs.get("image_embeds")
                hint = added_cond_kwargs.get("hint")
                aug_emb, hint = self.add_embedding(image_embs, hint)
                sample = torch.cat([sample, hint], dim=1)

            emb = emb + aug_emb if aug_emb is not None else emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
                # Kadinsky 2.1 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )

                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
                # Kandinsky 2.2 - style
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(image_embeds)
            elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                    )
                image_embeds = added_cond_kwargs.get("image_embeds")
                image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

            # 2. pre-process
            sample = self.conv_in(sample)

            # 2.5 GLIGEN position net
            if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                cross_attention_kwargs = cross_attention_kwargs.copy()
                gligen_args = cross_attention_kwargs.pop("gligen")
                cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

            # 3. down
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
            if USE_PEFT_BACKEND:
                # weight the lora layers by setting `lora_scale` for each PEFT layer
                scale_lora_layers(self, lora_scale)

            is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
            # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
            is_adapter = down_intrablock_additional_residuals is not None
            # maintain backward compatibility for legacy usage, where
            #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
            #       but can only use one or the other
            if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
                deprecate(
                    "T2I should not use down_block_additional_residuals",
                    "1.3.0",
                    "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                        and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                        for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                    standard_warn=False,
                )
                down_intrablock_additional_residuals = down_block_additional_residuals
                is_adapter = True

            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    # For t2i-adapter CrossAttnDownBlock2D
                    additional_residuals = {}
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                        **additional_residuals,
                    )
                else:
                    # sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                    if is_adapter and len(down_intrablock_additional_residuals) > 0:
                        sample += down_intrablock_additional_residuals.pop(0)

                down_block_res_samples += res_samples

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    _, _, ori_H, ori_W = down_block_res_sample.shape
                    down_block_additional_residual = F.interpolate(down_block_additional_residual, (ori_H, ori_W), mode='bicubic')
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0
                    and sample.shape == down_intrablock_additional_residuals[0].shape
                ):
                    sample += down_intrablock_additional_residuals.pop(0)

            if is_controlnet:
                _, _, ori_H, ori_W = sample.shape
                mid_block_additional_residual = F.interpolate(mid_block_additional_residual, (ori_H, ori_W), mode='bicubic')
                sample = sample + mid_block_additional_residual

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        # scale=lora_scale,
                    )
                    # sample = upsample_block(
                    #     hidden_states=sample,
                    #     temb=emb,
                    #     res_hidden_states_tuple=res_samples,
                    #     upsample_size=upsample_size,
                    #     scale=lora_scale,
                    # )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (sample,)

            return UNet2DConditionOutput(sample=sample)
    return unet_2d_condition


def make_diffusers_transformer_block(block_class: Type[torch.nn.Module], generator: torch.Generator) -> Type[torch.nn.Module]:
    # replace global self-attention with MSW-MSA
    class transformer_block(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.FloatTensor:

            # reference: https://github.com/microsoft/Swin-Transformer
            def window_partition(x, window_size, shift_size, H, W):
                """
                Args:
                    x: (B, H, W, C)
                    window_size (int): window size

                Returns:
                    windows: (num_windows*B, window_size, window_size, C)
                """
                B, N, C = x.shape
                x = x.view(B,H,W,C)
                if H % 2 != 0 or W % 2 != 0:
                    warnings.warn(
                        f"HiDiffusion Warning: The feature size is {(H,W)} and cannot be directly partitioned into windows. We interpolate the size to {(window_size[0]*2, window_size[1]*2)} "
                        f"to enable the window partition. Even though the generation is OK, the image quality would be largely decreased. "
                        f"We suggest removing window attention by setting apply_hidiffusion(pipe, apply_window_attn=False) for better image quality."
                    )
                    x = F.interpolate(x.permute(0,3,1,2).contiguous(), size=(window_size[0]*2, window_size[1]*2), mode='bicubic').permute(0,2,3,1).contiguous()
                if type(shift_size) is list or type(shift_size) is tuple:
                    if shift_size[0] > 0:
                        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
                else:
                    if shift_size > 0:
                        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
                x = x.view(B, 2, window_size[0], 2, window_size[1], C)
                windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
                windows = windows.view(-1, window_size[0] * window_size[1], C)
                return windows


            def window_reverse(windows, window_size, H, W, shift_size):
                """
                Args:
                    windows: (num_windows*B, window_size, window_size, C)
                    window_size (int): Window size
                    H (int): Height of image
                    W (int): Width of image

                Returns:
                    x: (B, H, W, C)
                """
                B, N, C = windows.shape
                windows = windows.view(-1, window_size[0], window_size[1], C)
                B = int(windows.shape[0] / 4) # 2x2
                x = windows.view(B, 2, 2, window_size[0], window_size[1], -1)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, window_size[0]*2, window_size[1]*2, -1)
                if type(shift_size) is list or type(shift_size) is tuple:
                    if shift_size[0] > 0:
                        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
                else:
                    if shift_size > 0:
                        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
                if H % 2 != 0 or W % 2 != 0:
                    x = F.interpolate(x.permute(0,3,1,2).contiguous(), size=(H, W), mode='bicubic').permute(0,2,3,1).contiguous()
                x = x.view(B, H*W, C)
                return x

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.use_layer_norm:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.use_ada_layer_norm_single:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # MSW-MSA
            if generator is not None:
                rand_num = torch.rand(1, generator=generator, device=generator.device)
            else:
                rand_num = torch.rand(1)

            B, N, C = hidden_states.shape
            ori_H, ori_W = self.info['size']
            downsample_ratio = round(((ori_H*ori_W) / N)**0.5)
            H, W = (math.ceil(ori_H/downsample_ratio), math.ceil(ori_W/downsample_ratio))
            widow_size = (math.ceil(H/2), math.ceil(W/2))
            if rand_num <= 0.25:
                shift_size = (0,0)
            if rand_num > 0.25 and rand_num <= 0.5:
                shift_size = (widow_size[0]//4, widow_size[1]//4)
            if rand_num > 0.5 and rand_num <= 0.75:
                shift_size = (widow_size[0]//4*2, widow_size[1]//4*2)
            if rand_num > 0.75 and rand_num <= 1:
                shift_size = (widow_size[0]//4*3, widow_size[1]//4*3)
            norm_hidden_states = window_partition(norm_hidden_states, widow_size, shift_size, H, W)
            # 1. Retrieve lora scale.
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            # 2. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            attn_output = window_reverse(attn_output, widow_size, H, W, shift_size)

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.use_ada_layer_norm_single:
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.use_ada_layer_norm_continuous:
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if self.use_ada_layer_norm_continuous:
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
                )
                # ff_output = _chunked_feed_forward(
                #     self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
                # )
            else:
                ff_output = self.ff(norm_hidden_states)
                # ff_output = self.ff(norm_hidden_states, scale=lora_scale)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.use_ada_layer_norm_single:
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

    return transformer_block


def make_diffusers_cross_attn_down_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional downsampler with resolution-aware downsampler
    class cross_attn_down_block(block_class):
        # Save for unpatching later
        _parent = block_class
        timestep = 0
        aggressive_raunet = False
        T1_ratio = 0
        T1_start = 0
        T1_end = 0
        aggressive_raunet = False
        T1 = 0 # to avoid confict with sdxl-turbo
        max_timestep = 50
        info: dict = None
        model: str = None

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            additional_residuals: Optional[torch.FloatTensor] = None,
        ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:

            self.max_timestep = _get_max_timesteps(self.info)
            ori_H, ori_W = self.info['size']
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]

                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    elif self.info['is_playground']:
                        self.aggressive_raunet = playground_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise Exception("Error model. HiDiffusion now only supports sd15, sd21, sdxl, sdxl-turbo.")

            if self.aggressive_raunet:
                # self.T1_start = min(int(self.max_timestep * self.T1_ratio * 0.4), int(8/50 * self.max_timestep))
                self.T1_start = int(aggressive_step/50 * self.max_timestep)
                self.T1_end = int(self.max_timestep * self.T1_ratio)
                self.T1 = 0 # to avoid confict with sdxl-turbo
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)

            output_states = ()
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            blocks = list(zip(self.resnets, self.attentions))

            for i, (resnet, attn) in enumerate(blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    # hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                # apply additional residuals to the output of the last pair of resnet and attention blocks
                if i == len(blocks) - 1 and additional_residuals is not None:
                    hidden_states = hidden_states + additional_residuals

                if i == 0:
                    if self.aggressive_raunet and self.timestep >= self.T1_start and self.timestep < self.T1_end:
                        self.info["upsample_size"] = (hidden_states.shape[2], hidden_states.shape[3])
                        hidden_states = F.avg_pool2d(hidden_states, kernel_size=(2,2),ceil_mode=True)
                    elif self.timestep < self.T1:
                        self.info["upsample_size"] = (hidden_states.shape[2], hidden_states.shape[3])
                        hidden_states = F.avg_pool2d(hidden_states, kernel_size=(2,2),ceil_mode=True)
                output_states = output_states + (hidden_states,)

            if self.downsamplers is not None:
                for downsampler in self.downsamplers:
                    hidden_states = downsampler(hidden_states)
                    # hidden_states = downsampler(hidden_states, scale=lora_scale)

                output_states = output_states + (hidden_states,)

            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0

            return hidden_states, output_states
    return cross_attn_down_block

def make_diffusers_cross_attn_up_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional downsampler with resolution-aware downsampler
    class cross_attn_up_block(block_class):
        # Save for unpatching later
        _parent = block_class
        timestep = 0
        aggressive_raunet = False
        T1_ratio = 0
        T1_start = 0
        T1_end = 0
        aggressive_raunet = False
        T1 = 0 # to avoid confict with sdxl-turbo
        max_timestep = 50

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ) -> torch.FloatTensor:

            self.max_timestep = _get_max_timesteps(self.info)
            ori_H, ori_W = self.info['size']
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]

                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    elif self.info['is_playground']:
                        self.aggressive_raunet = playground_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet

                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise Exception("Error model. HiDiffusion now only supports sd15, sd21, sdxl, sdxl-turbo.")

            if self.aggressive_raunet:
                # self.T1_start = min(int(self.max_timestep * self.T1_ratio * 0.4), int(8/50 * self.max_timestep))
                self.T1_start = int(aggressive_step/50 * self.max_timestep)
                self.T1_end = int(self.max_timestep * self.T1_ratio)
                self.T1 = 0 # to avoid confict with sdxl-turbo
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)

            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
            is_freeu_enabled = (
                getattr(self, "s1", None)
                and getattr(self, "s2", None)
                and getattr(self, "b1", None)
                and getattr(self, "b2", None)
            )

            for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FreeU: Only operate on the first two stages
                if is_freeu_enabled:
                    hidden_states, res_hidden_states = apply_freeu(
                        self.resolution_idx,
                        hidden_states,
                        res_hidden_states,
                        s1=self.s1,
                        s2=self.s2,
                        b1=self.b1,
                        b2=self.b2,
                    )

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    # hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                    if i == 1:
                        if self.aggressive_raunet and self.timestep >= self.T1_start and self.timestep < self.T1_end:
                            hidden_states = F.interpolate(hidden_states, size=self.info["upsample_size"], mode='bicubic')
                        elif self.timestep < self.T1:
                            hidden_states = F.interpolate(hidden_states, size=self.info["upsample_size"], mode='bicubic')
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)
                    # hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)

            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0

            return hidden_states
    return cross_attn_up_block



def make_diffusers_downsampler_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional downsampler with resolution-aware downsampler
    class downsampler_block(block_class):
        # Save for unpatching later
        _parent = block_class
        T1_ratio = 0
        T1 = 0
        timestep = 0
        aggressive_raunet = False
        max_timestep = 50

        def forward(self, hidden_states: torch.Tensor, scale = 1.0) -> torch.Tensor:
            self.max_timestep = _get_max_timesteps(self.info)
            ori_H, ori_W = self.info['size']
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]

                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    elif self.info['is_playground']:
                        self.aggressive_raunet = playground_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise Exception("Error model. HiDiffusion now only supports sd15, sd21, sdxl, sdxl-turbo.")

            if self.aggressive_raunet:
                # self.T1 = min(int(self.max_timestep * self.T1_ratio), int(8/50 * self.max_timestep))
                self.T1 = int(aggressive_step/50 * self.max_timestep)
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)
            if self.timestep < self.T1:
                self.ori_stride = self.stride
                self.ori_padding = self.padding
                self.ori_dilation = self.dilation

                self.stride = (4,4)
                self.padding = (2,2)
                self.dilation = (2,2)

            if old_diffusers:
                if self.lora_layer is None:
                    # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
                    # see: https://github.com/huggingface/diffusers/pull/4315
                    hidden_states = F.conv2d(
                        hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
                    )
                    if self.timestep < self.T1:
                        self.stride = self.ori_stride
                        self.padding = self.ori_padding
                        self.dilation = self.ori_dilation
                    self.timestep += 1
                    if self.timestep == self.max_timestep:
                        self.timestep = 0
                    return hidden_states
                else:
                    original_outputs = F.conv2d(
                        hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
                    )
                    return original_outputs + (scale * self.lora_layer(hidden_states))
            else:
                hidden_states = F.conv2d(
                    hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
                )
                if self.timestep < self.T1:
                    self.stride = self.ori_stride
                    self.padding = self.ori_padding
                    self.dilation = self.ori_dilation
                self.timestep += 1
                if self.timestep == self.max_timestep:
                    self.timestep = 0
                return hidden_states
    return downsampler_block


def make_diffusers_upsampler_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    # replace conventional upsampler with resolution-aware downsampler
    class upsampler_block(block_class):
        # Save for unpatching later
        _parent = block_class
        T1_ratio = 0
        T1 = 0
        timestep = 0
        aggressive_raunet = False
        max_timestep = 50
        info: dict = None

        def forward(self, hidden_states: torch.Tensor, scale = 1.0) -> torch.Tensor:
            self.max_timestep = _get_max_timesteps(self.info)
            ori_H, ori_W = self.info['size']
            if self.model == 'sd15':
                if ori_H < 256 or ori_W < 256:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_1024'][self.switching_threshold_ratio]
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sd15_2048'][self.switching_threshold_ratio]
            elif self.model == 'sdxl':
                if ori_H < 512 or ori_W < 512:
                    if self.info['text_to_img_controlnet']:
                        self.T1_ratio = text_to_img_controlnet_switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]
                    else:
                        self.T1_ratio = switching_threshold_ratio_dict['sdxl_2048'][self.switching_threshold_ratio]

                    if self.info['is_inpainting_task']:
                        self.aggressive_raunet = inpainting_is_aggressive_raunet
                    elif self.info['is_playground']:
                        self.aggressive_raunet = playground_is_aggressive_raunet
                    else:
                        self.aggressive_raunet = is_aggressive_raunet
                else:
                    self.T1_ratio = switching_threshold_ratio_dict['sdxl_4096'][self.switching_threshold_ratio]
            elif self.model == 'sdxl_turbo':
                self.T1_ratio = switching_threshold_ratio_dict['sdxl_turbo_1024'][self.switching_threshold_ratio]
            else:
                raise Exception("Error model. HiDiffusion now only supports sd15, sd21, sdxl, sdxl-turbo.")


            if self.aggressive_raunet:
                # self.T1 = min(int(self.max_timestep * self.T1_ratio), int(8/50 * self.max_timestep))
                self.T1 = int(aggressive_step/50 * self.max_timestep)
            else:
                self.T1 = int(self.max_timestep * self.T1_ratio)
            self.timestep += 1
            if self.timestep == self.max_timestep:
                self.timestep = 0

            if old_diffusers:
                if self.lora_layer is None:
                    # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
                    # see: https://github.com/huggingface/diffusers/pull/4315
                    return F.conv2d(
                        hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
                    )
                else:
                    original_outputs = F.conv2d(
                        hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
                    )
                    return original_outputs + (scale * self.lora_layer(hidden_states))
            else:
                return F.conv2d(
                    hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
                )
    return upsampler_block



def hook_diffusion_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_hidiffusion. """
    def hook(module, args):
        module.info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model.info["hooks"].append(model.register_forward_pre_hook(hook))



def apply_hidiffusion(
        model: torch.nn.Module,
        apply_raunet: bool = True,
        apply_window_attn: bool = True,
        is_playground = False,
        generator: torch.Generator | None = None):
    """
    model: diffusers model. We support SD 1.5, 2.1, XL, XL Turbo.

    apply_raunet: whether to apply RAU-Net

    apply_window_attn: whether to apply MSW-MSA.
    """

    # Make sure the module is not currently patched
    remove_hidiffusion(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        raise RuntimeError("Provided model was not a diffusers model/pipeline, as expected.")
    else:
        # Check if the pipeline is a ControlNet pipeline
        is_sdxl_controlnet = hasattr(model, 'controlnet') and isinstance_str(model, "StableDiffusionXLControlNet", prefix=True)
        is_sd_controlnet = hasattr(model, 'controlnet') and isinstance_str(model, "StableDiffusionControlNet", prefix=True)

        # Check for ControlNet Inpaint pipelines
        is_sdxl_controlnet_inpaint = is_sdxl_controlnet and isinstance_str(model, 'Inpaint', contains=True)
        is_sd_controlnet_inpaint = is_sd_controlnet and isinstance_str(model, 'Inpaint', contains=True)

        if is_sdxl_controlnet_inpaint or is_sd_controlnet_inpaint:
            # For ControlNet Inpaint pipelines, we don't patch the pipeline class
            # because they already have all the necessary inpainting logic
            # We only patch the UNet for HiDiffusion optimizations
            make_block_fn = make_diffusers_unet_2d_condition
            model.unet.__class__ = make_block_fn(model.unet.__class__)
        elif is_sdxl_controlnet:
            make_ppl_fn = make_diffusers_sdxl_controlnet_ppl
            model.__class__ = make_ppl_fn(model.__class__)

            make_block_fn = make_diffusers_unet_2d_condition
            model.unet.__class__ = make_block_fn(model.unet.__class__)
        elif is_sd_controlnet:
            # For SD 1.5 ControlNet, we don't need to patch the pipeline class
            # Just patch the UNet for consistency
            make_block_fn = make_diffusers_unet_2d_condition
            model.unet.__class__ = make_block_fn(model.unet.__class__)

        diffusion_model = model.unet if hasattr(model, "unet") else model

    # Hack, avoid non-square problem. See unet_2d_condition.py in diffusers
    diffusion_model.num_upsamplers += 12

    name_or_path = model.name_or_path
    diffusion_model_module_key = []
    if name_or_path not in supported_official_model:
        for key, module in diffusion_model.named_modules():
            diffusion_model_module_key.append(key)
        if set(sd15_module_key) < set(diffusion_model_module_key):
            name_or_path = 'runwayml/stable-diffusion-v1-5'
        elif set(sdxl_module_key) < set(diffusion_model_module_key):
            name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'

    diffusion_model.info = {
        'size': None,
        'upsample_size': None,
        'hooks': [],
        'text_to_img_controlnet': hasattr(model, 'controlnet'),
        'is_inpainting_task': model.__class__ in auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.values(),
        'is_playground': is_playground,
        'pipeline': model
    }
    model.info = diffusion_model.info
    hook_diffusion_model(diffusion_model)

    if name_or_path in ['runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-1-base']:
        modified_key = sd15_hidiffusion_key()
        for key, module in diffusion_model.named_modules():
            if apply_raunet and key in modified_key['down_module_key']:
                make_block_fn = make_diffusers_downsampler_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'
            if apply_raunet and key in modified_key['down_module_key_extra']:
                make_block_fn = make_diffusers_cross_attn_down_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'
            if apply_raunet and key in modified_key['up_module_key']:
                make_block_fn = make_diffusers_upsampler_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'
            if apply_raunet and key in modified_key['up_module_key_extra']:
                make_block_fn = make_diffusers_cross_attn_up_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'
            if apply_window_attn and key in modified_key['windown_attn_module_key']:
                make_block_fn = make_diffusers_transformer_block
                module.__class__ = make_block_fn(module.__class__, generator)
            module.model = 'sd15'
            module.info = diffusion_model.info

    elif name_or_path in ['stabilityai/stable-diffusion-xl-base-1.0', 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1']:
        modified_key = sdxl_hidiffusion_key()
        for key, module in diffusion_model.named_modules():
            if apply_raunet and key in modified_key['down_module_key']:
                make_block_fn = make_diffusers_cross_attn_down_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'

            if apply_raunet and key in modified_key['down_module_key_extra']:
                make_block_fn = make_diffusers_downsampler_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'

            if apply_raunet and key in modified_key['up_module_key']:
                make_block_fn = make_diffusers_cross_attn_up_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'

            if apply_raunet and key in modified_key['up_module_key_extra']:
                make_block_fn = make_diffusers_upsampler_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T2_ratio'

            if apply_window_attn and key in modified_key['windown_attn_module_key']:
                make_block_fn = make_diffusers_transformer_block
                module.__class__ = make_block_fn(module.__class__, generator)
            module.model = 'sdxl'
            module.info = diffusion_model.info

    elif name_or_path == 'stabilityai/sdxl-turbo':
        modified_key = sdxl_turbo_hidiffusion_key()
        for key, module in diffusion_model.named_modules():
            if apply_raunet and key in modified_key['down_module_key']:
                make_block_fn = make_diffusers_cross_attn_down_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'

            if apply_raunet and key in modified_key['up_module_key']:
                make_block_fn = make_diffusers_cross_attn_up_block
                module.__class__ = make_block_fn(module.__class__)
                module.switching_threshold_ratio = 'T1_ratio'

            if apply_window_attn and key in modified_key['windown_attn_module_key']:
                make_block_fn = make_diffusers_transformer_block
                module.__class__ = make_block_fn(module.__class__, generator)

            module.model = 'sdxl_turbo'
            module.info = diffusion_model.info
    else:
        raise Exception(f'{model.name_or_path} is not a supported model. HiDiffusion now only supports runwayml/stable-diffusion-v1-5, stabilityai/stable-diffusion-2-1-base, stabilityai/stable-diffusion-xl-base-1.0, stabilityai/sdxl-turbo, diffusers/stable-diffusion-xl-1.0-inpainting-0.1 and their derivative models/pipelines.')
    return model





def remove_hidiffusion(model: torch.nn.Module):
    """ Removes hidiffusion from a Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "info"):
            for hook in module.info["hooks"]:
                hook.remove()
            module.info["hooks"].clear()

        if hasattr(module, "_parent"):
            module.__class__ = module._parent

    return model
