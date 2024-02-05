# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

import math
from contextlib import ExitStack
from functools import singledispatchmethod
from typing import List, Literal, Optional, Union

import einops
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.adapter import T2IAdapter
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.schedulers import DPMSolverSDEScheduler
from diffusers.schedulers import SchedulerMixin as Scheduler
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.fields import (
    ConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    LatentsField,
    OutputField,
    UIType,
    WithMetadata,
)
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.primitives import (
    DenoiseMaskOutput,
    ImageOutput,
    LatentsOutput,
)
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus
from invokeai.backend.model_management.models import ModelType, SilenceWarnings
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData, IPAdapterConditioningInfo

from ...backend.model_management.lora import ModelPatcher
from ...backend.model_management.models import BaseModelType
from ...backend.model_management.seamless import set_seamless
from ...backend.stable_diffusion import PipelineIntermediateState
from ...backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    IPAdapterData,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
    image_resized_to_grid_as_tensor,
)
from ...backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from ...backend.stable_diffusion.schedulers import SCHEDULER_MAP
from ...backend.util.devices import choose_precision, choose_torch_device
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from .controlnet_image_processors import ControlField
from .model import ModelInfo, UNetField, VaeField

if choose_torch_device() == torch.device("mps"):
    from torch import mps

DEFAULT_PRECISION = choose_precision(choose_torch_device())

SAMPLER_NAME_VALUES = Literal[tuple(SCHEDULER_MAP.keys())]

# HACK: Many nodes are currently hard-coded to use a fixed latent scale factor of 8. This is fragile, and will need to
# be addressed if future models use a different latent scale factor. Also, note that there may be places where the scale
# factor is hard-coded to a literal '8' rather than using this constant.
# The ratio of image:latent dimensions is LATENT_SCALE_FACTOR:1, or 8:1.
LATENT_SCALE_FACTOR = 8


@invocation_output("scheduler_output")
class SchedulerOutput(BaseInvocationOutput):
    scheduler: SAMPLER_NAME_VALUES = OutputField(description=FieldDescriptions.scheduler, ui_type=UIType.Scheduler)


@invocation(
    "scheduler",
    title="Scheduler",
    tags=["scheduler"],
    category="latents",
    version="1.0.0",
)
class SchedulerInvocation(BaseInvocation):
    """Selects a scheduler."""

    scheduler: SAMPLER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )

    def invoke(self, context: InvocationContext) -> SchedulerOutput:
        return SchedulerOutput(scheduler=self.scheduler)


@invocation(
    "create_denoise_mask",
    title="Create Denoise Mask",
    tags=["mask", "denoise"],
    category="latents",
    version="1.0.1",
)
class CreateDenoiseMaskInvocation(BaseInvocation):
    """Creates mask for denoising model run."""

    vae: VaeField = InputField(description=FieldDescriptions.vae, input=Input.Connection, ui_order=0)
    image: Optional[ImageField] = InputField(default=None, description="Image which will be masked", ui_order=1)
    mask: ImageField = InputField(description="The mask to use when pasting", ui_order=2)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled, ui_order=3)
    fp32: bool = InputField(
        default=DEFAULT_PRECISION == "float32",
        description=FieldDescriptions.fp32,
        ui_order=4,
    )

    def prep_mask_tensor(self, mask_image):
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        # if shape is not None:
        #    mask_tensor = tv_resize(mask_tensor, shape, T.InterpolationMode.BILINEAR)
        return mask_tensor

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> DenoiseMaskOutput:
        if self.image is not None:
            image = context.images.get_pil(self.image.image_name)
            image = image_resized_to_grid_as_tensor(image.convert("RGB"))
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            image = None

        mask = self.prep_mask_tensor(
            context.images.get_pil(self.mask.image_name),
        )

        if image is not None:
            vae_info = context.models.load(**self.vae.vae.model_dump())

            img_mask = tv_resize(mask, image.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
            masked_image = image * torch.where(img_mask < 0.5, 0.0, 1.0)
            # TODO:
            masked_latents = ImageToLatentsInvocation.vae_encode(vae_info, self.fp32, self.tiled, masked_image.clone())

            masked_latents_name = context.latents.save(tensor=masked_latents)
        else:
            masked_latents_name = None

        mask_name = context.latents.save(tensor=mask)

        return DenoiseMaskOutput.build(
            mask_name=mask_name,
            masked_latents_name=masked_latents_name,
        )


def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelInfo,
    scheduler_name: str,
    seed: int,
) -> Scheduler:
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP["ddim"])
    orig_scheduler_info = context.models.load(**scheduler_info.model_dump())
    with orig_scheduler_info as orig_scheduler:
        scheduler_config = orig_scheduler.config

    if "_backup" in scheduler_config:
        scheduler_config = scheduler_config["_backup"]
    scheduler_config = {
        **scheduler_config,
        **scheduler_extra_config,
        "_backup": scheduler_config,
    }

    # make dpmpp_sde reproducable(seed can be passed only in initializer)
    if scheduler_class is DPMSolverSDEScheduler:
        scheduler_config["noise_sampler_seed"] = seed

    scheduler = scheduler_class.from_config(scheduler_config)

    # hack copied over from generate.py
    if not hasattr(scheduler, "uses_inpainting_model"):
        scheduler.uses_inpainting_model = lambda: False
    return scheduler


@invocation(
    "denoise_latents",
    title="Denoise Latents",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.5.2",
)
class DenoiseLatentsInvocation(BaseInvocation):
    """Denoises noisy latents to decodable images"""

    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection, ui_order=0
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection, ui_order=1
    )
    noise: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
        ui_order=3,
    )
    steps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps)
    cfg_scale: Union[float, List[float]] = InputField(
        default=7.5, ge=1, description=FieldDescriptions.cfg_scale, title="CFG Scale"
    )
    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    scheduler: SAMPLER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
    )
    unet: UNetField = InputField(
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
        ui_order=2,
    )
    control: Optional[Union[ControlField, list[ControlField]]] = InputField(
        default=None,
        input=Input.Connection,
        ui_order=5,
    )
    ip_adapter: Optional[Union[IPAdapterField, list[IPAdapterField]]] = InputField(
        description=FieldDescriptions.ip_adapter,
        title="IP-Adapter",
        default=None,
        input=Input.Connection,
        ui_order=6,
    )
    t2i_adapter: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = InputField(
        description=FieldDescriptions.t2i_adapter,
        title="T2I-Adapter",
        default=None,
        input=Input.Connection,
        ui_order=7,
    )
    cfg_rescale_multiplier: float = InputField(
        title="CFG Rescale Multiplier", default=0, ge=0, lt=1, description=FieldDescriptions.cfg_rescale_multiplier
    )
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
        ui_order=4,
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.mask,
        input=Input.Connection,
        ui_order=8,
    )

    @field_validator("cfg_scale")
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

    def get_conditioning_data(
        self,
        context: InvocationContext,
        scheduler,
        unet,
        seed,
    ) -> ConditioningData:
        positive_cond_data = context.conditioning.get(self.positive_conditioning.conditioning_name)
        c = positive_cond_data.conditionings[0].to(device=unet.device, dtype=unet.dtype)
        extra_conditioning_info = c.extra_conditioning

        negative_cond_data = context.conditioning.get(self.negative_conditioning.conditioning_name)
        uc = negative_cond_data.conditionings[0].to(device=unet.device, dtype=unet.dtype)

        conditioning_data = ConditioningData(
            unconditioned_embeddings=uc,
            text_embeddings=c,
            guidance_scale=self.cfg_scale,
            guidance_rescale_multiplier=self.cfg_rescale_multiplier,
            extra=extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=0.0,  # threshold,
                warmup=0.2,  # warmup,
                h_symmetry_time_pct=None,  # h_symmetry_time_pct,
                v_symmetry_time_pct=None,  # v_symmetry_time_pct,
            ),
        )

        conditioning_data = conditioning_data.add_scheduler_args_if_applicable(
            scheduler,
            # for ddim scheduler
            eta=0.0,  # ddim_eta
            # for ancestral and sde schedulers
            # flip all bits to have noise different from initial
            generator=torch.Generator(device=unet.device).manual_seed(seed ^ 0xFFFFFFFF),
        )
        return conditioning_data

    def create_pipeline(
        self,
        unet,
        scheduler,
    ) -> StableDiffusionGeneratorPipeline:
        # TODO:
        # configure_model_padding(
        #    unet,
        #    self.seamless,
        #    self.seamless_axes,
        # )

        class FakeVae:
            class FakeVaeConfig:
                def __init__(self):
                    self.block_out_channels = [0]

            def __init__(self):
                self.config = FakeVae.FakeVaeConfig()

        return StableDiffusionGeneratorPipeline(
            vae=FakeVae(),  # TODO: oh...
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    def prep_control_data(
        self,
        context: InvocationContext,
        control_input: Union[ControlField, List[ControlField]],
        latents_shape: List[int],
        exit_stack: ExitStack,
        do_classifier_free_guidance: bool = True,
    ) -> List[ControlNetData]:
        # Assuming fixed dimensional scaling of LATENT_SCALE_FACTOR.
        control_height_resize = latents_shape[2] * LATENT_SCALE_FACTOR
        control_width_resize = latents_shape[3] * LATENT_SCALE_FACTOR
        if control_input is None:
            control_list = None
        elif isinstance(control_input, list) and len(control_input) == 0:
            control_list = None
        elif isinstance(control_input, ControlField):
            control_list = [control_input]
        elif isinstance(control_input, list) and len(control_input) > 0 and isinstance(control_input[0], ControlField):
            control_list = control_input
        else:
            control_list = None
        if control_list is None:
            return None
        # After above handling, any control that is not None should now be of type list[ControlField].

        # FIXME: add checks to skip entry if model or image is None
        #        and if weight is None, populate with default 1.0?
        controlnet_data = []
        for control_info in control_list:
            control_model = exit_stack.enter_context(
                context.models.load(
                    model_name=control_info.control_model.model_name,
                    model_type=ModelType.ControlNet,
                    base_model=control_info.control_model.base_model,
                )
            )

            # control_models.append(control_model)
            control_image_field = control_info.image
            input_image = context.images.get_pil(control_image_field.image_name)
            # self.image.image_type, self.image.image_name
            # FIXME: still need to test with different widths, heights, devices, dtypes
            #        and add in batch_size, num_images_per_prompt?
            #        and do real check for classifier_free_guidance?
            # prepare_control_image should return torch.Tensor of shape(batch_size, 3, height, width)
            control_image = prepare_control_image(
                image=input_image,
                do_classifier_free_guidance=do_classifier_free_guidance,
                width=control_width_resize,
                height=control_height_resize,
                # batch_size=batch_size * num_images_per_prompt,
                # num_images_per_prompt=num_images_per_prompt,
                device=control_model.device,
                dtype=control_model.dtype,
                control_mode=control_info.control_mode,
                resize_mode=control_info.resize_mode,
            )
            control_item = ControlNetData(
                model=control_model,  # model object
                image_tensor=control_image,
                weight=control_info.control_weight,
                begin_step_percent=control_info.begin_step_percent,
                end_step_percent=control_info.end_step_percent,
                control_mode=control_info.control_mode,
                # any resizing needed should currently be happening in prepare_control_image(),
                #    but adding resize_mode to ControlNetData in case needed in the future
                resize_mode=control_info.resize_mode,
            )
            controlnet_data.append(control_item)
            # MultiControlNetModel has been refactored out, just need list[ControlNetData]

        return controlnet_data

    def prep_ip_adapter_data(
        self,
        context: InvocationContext,
        ip_adapter: Optional[Union[IPAdapterField, list[IPAdapterField]]],
        conditioning_data: ConditioningData,
        exit_stack: ExitStack,
    ) -> Optional[list[IPAdapterData]]:
        """If IP-Adapter is enabled, then this function loads the requisite models, and adds the image prompt embeddings
        to the `conditioning_data` (in-place).
        """
        if ip_adapter is None:
            return None

        # ip_adapter could be a list or a single IPAdapterField. Normalize to a list here.
        if not isinstance(ip_adapter, list):
            ip_adapter = [ip_adapter]

        if len(ip_adapter) == 0:
            return None

        ip_adapter_data_list = []
        conditioning_data.ip_adapter_conditioning = []
        for single_ip_adapter in ip_adapter:
            ip_adapter_model: Union[IPAdapter, IPAdapterPlus] = exit_stack.enter_context(
                context.models.load(
                    model_name=single_ip_adapter.ip_adapter_model.model_name,
                    model_type=ModelType.IPAdapter,
                    base_model=single_ip_adapter.ip_adapter_model.base_model,
                )
            )

            image_encoder_model_info = context.models.load(
                model_name=single_ip_adapter.image_encoder_model.model_name,
                model_type=ModelType.CLIPVision,
                base_model=single_ip_adapter.image_encoder_model.base_model,
            )

            # `single_ip_adapter.image` could be a list or a single ImageField. Normalize to a list here.
            single_ipa_images = single_ip_adapter.image
            if not isinstance(single_ipa_images, list):
                single_ipa_images = [single_ipa_images]

            single_ipa_images = [context.images.get_pil(image.image_name) for image in single_ipa_images]

            # TODO(ryand): With some effort, the step of running the CLIP Vision encoder could be done before any other
            # models are needed in memory. This would help to reduce peak memory utilization in low-memory environments.
            with image_encoder_model_info as image_encoder_model:
                # Get image embeddings from CLIP and ImageProjModel.
                image_prompt_embeds, uncond_image_prompt_embeds = ip_adapter_model.get_image_embeds(
                    single_ipa_images, image_encoder_model
                )

                conditioning_data.ip_adapter_conditioning.append(
                    IPAdapterConditioningInfo(image_prompt_embeds, uncond_image_prompt_embeds)
                )

            ip_adapter_data_list.append(
                IPAdapterData(
                    ip_adapter_model=ip_adapter_model,
                    weight=single_ip_adapter.weight,
                    begin_step_percent=single_ip_adapter.begin_step_percent,
                    end_step_percent=single_ip_adapter.end_step_percent,
                )
            )

        return ip_adapter_data_list

    def run_t2i_adapters(
        self,
        context: InvocationContext,
        t2i_adapter: Optional[Union[T2IAdapterField, list[T2IAdapterField]]],
        latents_shape: list[int],
        do_classifier_free_guidance: bool,
    ) -> Optional[list[T2IAdapterData]]:
        if t2i_adapter is None:
            return None

        # Handle the possibility that t2i_adapter could be a list or a single T2IAdapterField.
        if isinstance(t2i_adapter, T2IAdapterField):
            t2i_adapter = [t2i_adapter]

        if len(t2i_adapter) == 0:
            return None

        t2i_adapter_data = []
        for t2i_adapter_field in t2i_adapter:
            t2i_adapter_model_info = context.models.load(
                model_name=t2i_adapter_field.t2i_adapter_model.model_name,
                model_type=ModelType.T2IAdapter,
                base_model=t2i_adapter_field.t2i_adapter_model.base_model,
            )
            image = context.images.get_pil(t2i_adapter_field.image.image_name)

            # The max_unet_downscale is the maximum amount that the UNet model downscales the latent image internally.
            if t2i_adapter_field.t2i_adapter_model.base_model == BaseModelType.StableDiffusion1:
                max_unet_downscale = 8
            elif t2i_adapter_field.t2i_adapter_model.base_model == BaseModelType.StableDiffusionXL:
                max_unet_downscale = 4
            else:
                raise ValueError(
                    f"Unexpected T2I-Adapter base model type: '{t2i_adapter_field.t2i_adapter_model.base_model}'."
                )

            t2i_adapter_model: T2IAdapter
            with t2i_adapter_model_info as t2i_adapter_model:
                total_downscale_factor = t2i_adapter_model.total_downscale_factor

                # Resize the T2I-Adapter input image.
                # We select the resize dimensions so that after the T2I-Adapter's total_downscale_factor is applied, the
                # result will match the latent image's dimensions after max_unet_downscale is applied.
                t2i_input_height = latents_shape[2] // max_unet_downscale * total_downscale_factor
                t2i_input_width = latents_shape[3] // max_unet_downscale * total_downscale_factor

                # Note: We have hard-coded `do_classifier_free_guidance=False`. This is because we only want to prepare
                # a single image. If CFG is enabled, we will duplicate the resultant tensor after applying the
                # T2I-Adapter model.
                #
                # Note: We re-use the `prepare_control_image(...)` from ControlNet for T2I-Adapter, because it has many
                # of the same requirements (e.g. preserving binary masks during resize).
                t2i_image = prepare_control_image(
                    image=image,
                    do_classifier_free_guidance=False,
                    width=t2i_input_width,
                    height=t2i_input_height,
                    num_channels=t2i_adapter_model.config.in_channels,
                    device=t2i_adapter_model.device,
                    dtype=t2i_adapter_model.dtype,
                    resize_mode=t2i_adapter_field.resize_mode,
                )

                adapter_state = t2i_adapter_model(t2i_image)

            if do_classifier_free_guidance:
                for idx, value in enumerate(adapter_state):
                    adapter_state[idx] = torch.cat([value] * 2, dim=0)

            t2i_adapter_data.append(
                T2IAdapterData(
                    adapter_state=adapter_state,
                    weight=t2i_adapter_field.weight,
                    begin_step_percent=t2i_adapter_field.begin_step_percent,
                    end_step_percent=t2i_adapter_field.end_step_percent,
                )
            )

        return t2i_adapter_data

    # original idea by https://github.com/AmericanPresidentJimmyCarter
    # TODO: research more for second order schedulers timesteps
    def init_scheduler(self, scheduler, device, steps, denoising_start, denoising_end):
        if scheduler.config.get("cpu_only", False):
            scheduler.set_timesteps(steps, device="cpu")
            timesteps = scheduler.timesteps.to(device=device)
        else:
            scheduler.set_timesteps(steps, device=device)
            timesteps = scheduler.timesteps

        # skip greater order timesteps
        _timesteps = timesteps[:: scheduler.order]

        # get start timestep index
        t_start_val = int(round(scheduler.config.num_train_timesteps * (1 - denoising_start)))
        t_start_idx = len(list(filter(lambda ts: ts >= t_start_val, _timesteps)))

        # get end timestep index
        t_end_val = int(round(scheduler.config.num_train_timesteps * (1 - denoising_end)))
        t_end_idx = len(list(filter(lambda ts: ts >= t_end_val, _timesteps[t_start_idx:])))

        # apply order to indexes
        t_start_idx *= scheduler.order
        t_end_idx *= scheduler.order

        init_timestep = timesteps[t_start_idx : t_start_idx + 1]
        timesteps = timesteps[t_start_idx : t_start_idx + t_end_idx]
        num_inference_steps = len(timesteps) // scheduler.order

        return num_inference_steps, timesteps, init_timestep

    def prep_inpaint_mask(self, context: InvocationContext, latents):
        if self.denoise_mask is None:
            return None, None

        mask = context.latents.get(self.denoise_mask.mask_name)
        mask = tv_resize(mask, latents.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
        if self.denoise_mask.masked_latents_name is not None:
            masked_latents = context.latents.get(self.denoise_mask.masked_latents_name)
        else:
            masked_latents = None

        return 1 - mask, masked_latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with SilenceWarnings():  # this quenches NSFW nag from diffusers
            seed = None
            noise = None
            if self.noise is not None:
                noise = context.latents.get(self.noise.latents_name)
                seed = self.noise.seed

            if self.latents is not None:
                latents = context.latents.get(self.latents.latents_name)
                if seed is None:
                    seed = self.latents.seed

                if noise is not None and noise.shape[1:] != latents.shape[1:]:
                    raise Exception(f"Incompatable 'noise' and 'latents' shapes: {latents.shape=} {noise.shape=}")

            elif noise is not None:
                latents = torch.zeros_like(noise)
            else:
                raise Exception("'latents' or 'noise' must be provided!")

            if seed is None:
                seed = 0

            mask, masked_latents = self.prep_inpaint_mask(context, latents)

            # TODO(ryand): I have hard-coded `do_classifier_free_guidance=True` to mirror the behaviour of ControlNets,
            # below. Investigate whether this is appropriate.
            t2i_adapter_data = self.run_t2i_adapters(
                context,
                self.t2i_adapter,
                latents.shape,
                do_classifier_free_guidance=True,
            )

            def step_callback(state: PipelineIntermediateState):
                context.util.sd_step_callback(state, self.unet.unet.base_model)

            def _lora_loader():
                for lora in self.unet.loras:
                    lora_info = context.models.load(**lora.model_dump(exclude={"weight"}))
                    yield (lora_info.context.model, lora.weight)
                    del lora_info
                return

            unet_info = context.models.load(**self.unet.unet.model_dump())
            with (
                ExitStack() as exit_stack,
                ModelPatcher.apply_freeu(unet_info.context.model, self.unet.freeu_config),
                set_seamless(unet_info.context.model, self.unet.seamless_axes),
                unet_info as unet,
                # Apply the LoRA after unet has been moved to its target device for faster patching.
                ModelPatcher.apply_lora_unet(unet, _lora_loader()),
            ):
                latents = latents.to(device=unet.device, dtype=unet.dtype)
                if noise is not None:
                    noise = noise.to(device=unet.device, dtype=unet.dtype)
                if mask is not None:
                    mask = mask.to(device=unet.device, dtype=unet.dtype)
                if masked_latents is not None:
                    masked_latents = masked_latents.to(device=unet.device, dtype=unet.dtype)

                scheduler = get_scheduler(
                    context=context,
                    scheduler_info=self.unet.scheduler,
                    scheduler_name=self.scheduler,
                    seed=seed,
                )

                pipeline = self.create_pipeline(unet, scheduler)
                conditioning_data = self.get_conditioning_data(context, scheduler, unet, seed)

                controlnet_data = self.prep_control_data(
                    context=context,
                    control_input=self.control,
                    latents_shape=latents.shape,
                    # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                    do_classifier_free_guidance=True,
                    exit_stack=exit_stack,
                )

                ip_adapter_data = self.prep_ip_adapter_data(
                    context=context,
                    ip_adapter=self.ip_adapter,
                    conditioning_data=conditioning_data,
                    exit_stack=exit_stack,
                )

                num_inference_steps, timesteps, init_timestep = self.init_scheduler(
                    scheduler,
                    device=unet.device,
                    steps=self.steps,
                    denoising_start=self.denoising_start,
                    denoising_end=self.denoising_end,
                )

                (
                    result_latents,
                    result_attention_map_saver,
                ) = pipeline.latents_from_embeddings(
                    latents=latents,
                    timesteps=timesteps,
                    init_timestep=init_timestep,
                    noise=noise,
                    seed=seed,
                    mask=mask,
                    masked_latents=masked_latents,
                    num_inference_steps=num_inference_steps,
                    conditioning_data=conditioning_data,
                    control_data=controlnet_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                    callback=step_callback,
                )

            # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
            result_latents = result_latents.to("cpu")
            torch.cuda.empty_cache()
            if choose_torch_device() == torch.device("mps"):
                mps.empty_cache()

            name = context.latents.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=seed)


@invocation(
    "l2i",
    title="Latents to Image",
    tags=["latents", "image", "vae", "l2i"],
    category="latents",
    version="1.2.1",
)
class LatentsToImageInvocation(BaseInvocation, WithMetadata):
    """Generates an image from latents."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VaeField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    fp32: bool = InputField(default=DEFAULT_PRECISION == "float32", description=FieldDescriptions.fp32)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.latents.get(self.latents.latents_name)

        vae_info = context.models.load(**self.vae.vae.model_dump())

        with set_seamless(vae_info.context.model, self.vae.seamless_axes), vae_info as vae:
            latents = latents.to(vae.device)
            if self.fp32:
                vae.to(dtype=torch.float32)

                use_torch_2_0_or_xformers = isinstance(
                    vae.decoder.mid_block.attentions[0].processor,
                    (
                        AttnProcessor2_0,
                        XFormersAttnProcessor,
                        LoRAXFormersAttnProcessor,
                        LoRAAttnProcessor2_0,
                    ),
                )
                # if xformers or torch_2_0 is used attention block does not need
                # to be in float32 which can save lots of memory
                if use_torch_2_0_or_xformers:
                    vae.post_quant_conv.to(latents.dtype)
                    vae.decoder.conv_in.to(latents.dtype)
                    vae.decoder.mid_block.to(latents.dtype)
                else:
                    latents = latents.float()

            else:
                vae.to(dtype=torch.float16)
                latents = latents.half()

            if self.tiled or context.config.get().tiled_decode:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            # clear memory as vae decode can request a lot
            torch.cuda.empty_cache()
            if choose_torch_device() == torch.device("mps"):
                mps.empty_cache()

            with torch.inference_mode():
                # copied from diffusers pipeline
                latents = latents / vae.config.scaling_factor
                image = vae.decode(latents, return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)  # denormalize
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                np_image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                image = VaeImageProcessor.numpy_to_pil(np_image)[0]

        torch.cuda.empty_cache()
        if choose_torch_device() == torch.device("mps"):
            mps.empty_cache()

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)


LATENTS_INTERPOLATION_MODE = Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]


@invocation(
    "lresize",
    title="Resize Latents",
    tags=["latents", "resize"],
    category="latents",
    version="1.0.1",
)
class ResizeLatentsInvocation(BaseInvocation):
    """Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    width: int = InputField(
        ge=64,
        multiple_of=LATENT_SCALE_FACTOR,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        ge=64,
        multiple_of=LATENT_SCALE_FACTOR,
        description=FieldDescriptions.width,
    )
    mode: LATENTS_INTERPOLATION_MODE = InputField(default="bilinear", description=FieldDescriptions.interp_mode)
    antialias: bool = InputField(default=False, description=FieldDescriptions.torch_antialias)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.latents.get(self.latents.latents_name)

        # TODO:
        device = choose_torch_device()

        resized_latents = torch.nn.functional.interpolate(
            latents.to(device),
            size=(self.height // LATENT_SCALE_FACTOR, self.width // LATENT_SCALE_FACTOR),
            mode=self.mode,
            antialias=self.antialias if self.mode in ["bilinear", "bicubic"] else False,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        resized_latents = resized_latents.to("cpu")
        torch.cuda.empty_cache()
        if device == torch.device("mps"):
            mps.empty_cache()

        name = context.latents.save(tensor=resized_latents)
        return LatentsOutput.build(latents_name=name, latents=resized_latents, seed=self.latents.seed)


@invocation(
    "lscale",
    title="Scale Latents",
    tags=["latents", "resize"],
    category="latents",
    version="1.0.1",
)
class ScaleLatentsInvocation(BaseInvocation):
    """Scales latents by a given factor."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    scale_factor: float = InputField(gt=0, description=FieldDescriptions.scale_factor)
    mode: LATENTS_INTERPOLATION_MODE = InputField(default="bilinear", description=FieldDescriptions.interp_mode)
    antialias: bool = InputField(default=False, description=FieldDescriptions.torch_antialias)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.latents.get(self.latents.latents_name)

        # TODO:
        device = choose_torch_device()

        # resizing
        resized_latents = torch.nn.functional.interpolate(
            latents.to(device),
            scale_factor=self.scale_factor,
            mode=self.mode,
            antialias=self.antialias if self.mode in ["bilinear", "bicubic"] else False,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        resized_latents = resized_latents.to("cpu")
        torch.cuda.empty_cache()
        if device == torch.device("mps"):
            mps.empty_cache()

        name = context.latents.save(tensor=resized_latents)
        return LatentsOutput.build(latents_name=name, latents=resized_latents, seed=self.latents.seed)


@invocation(
    "i2l",
    title="Image to Latents",
    tags=["latents", "image", "vae", "i2l"],
    category="latents",
    version="1.0.1",
)
class ImageToLatentsInvocation(BaseInvocation):
    """Encodes an image into latents."""

    image: ImageField = InputField(
        description="The image to encode",
    )
    vae: VaeField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    fp32: bool = InputField(default=DEFAULT_PRECISION == "float32", description=FieldDescriptions.fp32)

    @staticmethod
    def vae_encode(vae_info, upcast, tiled, image_tensor):
        with vae_info as vae:
            orig_dtype = vae.dtype
            if upcast:
                vae.to(dtype=torch.float32)

                use_torch_2_0_or_xformers = isinstance(
                    vae.decoder.mid_block.attentions[0].processor,
                    (
                        AttnProcessor2_0,
                        XFormersAttnProcessor,
                        LoRAXFormersAttnProcessor,
                        LoRAAttnProcessor2_0,
                    ),
                )
                # if xformers or torch_2_0 is used attention block does not need
                # to be in float32 which can save lots of memory
                if use_torch_2_0_or_xformers:
                    vae.post_quant_conv.to(orig_dtype)
                    vae.decoder.conv_in.to(orig_dtype)
                    vae.decoder.mid_block.to(orig_dtype)
                # else:
                #    latents = latents.float()

            else:
                vae.to(dtype=torch.float16)
                # latents = latents.half()

            if tiled:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            # non_noised_latents_from_image
            image_tensor = image_tensor.to(device=vae.device, dtype=vae.dtype)
            with torch.inference_mode():
                latents = ImageToLatentsInvocation._encode_to_tensor(vae, image_tensor)

            latents = vae.config.scaling_factor * latents
            latents = latents.to(dtype=orig_dtype)

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        vae_info = context.models.load(**self.vae.vae.model_dump())

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        latents = self.vae_encode(vae_info, self.fp32, self.tiled, image_tensor)

        latents = latents.to("cpu")
        name = context.latents.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    @singledispatchmethod
    @staticmethod
    def _encode_to_tensor(vae: AutoencoderKL, image_tensor: torch.FloatTensor) -> torch.FloatTensor:
        image_tensor_dist = vae.encode(image_tensor).latent_dist
        latents = image_tensor_dist.sample().to(dtype=vae.dtype)  # FIXME: uses torch.randn. make reproducible!
        return latents

    @_encode_to_tensor.register
    @staticmethod
    def _(vae: AutoencoderTiny, image_tensor: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(image_tensor).latents


@invocation(
    "lblend",
    title="Blend Latents",
    tags=["latents", "blend"],
    category="latents",
    version="1.0.1",
)
class BlendLatentsInvocation(BaseInvocation):
    """Blend two latents using a given alpha. Latents must have same size."""

    latents_a: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    latents_b: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    alpha: float = InputField(default=0.5, description=FieldDescriptions.blend_alpha)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents_a = context.latents.get(self.latents_a.latents_name)
        latents_b = context.latents.get(self.latents_b.latents_name)

        if latents_a.shape != latents_b.shape:
            raise Exception("Latents to blend must be the same size.")

        # TODO:
        device = choose_torch_device()

        def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
            """
            Spherical linear interpolation
            Args:
                t (float/np.ndarray): Float value between 0.0 and 1.0
                v0 (np.ndarray): Starting vector
                v1 (np.ndarray): Final vector
                DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                    colineal. Not recommended to alter this.
            Returns:
                v2 (np.ndarray): Interpolation vector between v0 and v1
            """
            inputs_are_torch = False
            if not isinstance(v0, np.ndarray):
                inputs_are_torch = True
                v0 = v0.detach().cpu().numpy()
            if not isinstance(v1, np.ndarray):
                inputs_are_torch = True
                v1 = v1.detach().cpu().numpy()

            dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
            if np.abs(dot) > DOT_THRESHOLD:
                v2 = (1 - t) * v0 + t * v1
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                theta_t = theta_0 * t
                sin_theta_t = np.sin(theta_t)
                s0 = np.sin(theta_0 - theta_t) / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                v2 = s0 * v0 + s1 * v1

            if inputs_are_torch:
                v2 = torch.from_numpy(v2).to(device)

            return v2

        # blend
        blended_latents = slerp(self.alpha, latents_a, latents_b)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        blended_latents = blended_latents.to("cpu")
        torch.cuda.empty_cache()
        if device == torch.device("mps"):
            mps.empty_cache()

        name = context.latents.save(tensor=blended_latents)
        return LatentsOutput.build(latents_name=name, latents=blended_latents)


# The Crop Latents node was copied from @skunkworxdark's implementation here:
# https://github.com/skunkworxdark/XYGrid_nodes/blob/74647fa9c1fa57d317a94bd43ca689af7f0aae5e/images_to_grids.py#L1117C1-L1167C80
@invocation(
    "crop_latents",
    title="Crop Latents",
    tags=["latents", "crop"],
    category="latents",
    version="1.0.1",
)
# TODO(ryand): Named `CropLatentsCoreInvocation` to prevent a conflict with custom node `CropLatentsInvocation`.
# Currently, if the class names conflict then 'GET /openapi.json' fails.
class CropLatentsCoreInvocation(BaseInvocation):
    """Crops a latent-space tensor to a box specified in image-space. The box dimensions and coordinates must be
    divisible by the latent scale factor of 8.
    """

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    x: int = InputField(
        ge=0,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The left x coordinate (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )
    y: int = InputField(
        ge=0,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The top y coordinate (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )
    width: int = InputField(
        ge=1,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The width (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )
    height: int = InputField(
        ge=1,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The height (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.latents.get(self.latents.latents_name)

        x1 = self.x // LATENT_SCALE_FACTOR
        y1 = self.y // LATENT_SCALE_FACTOR
        x2 = x1 + (self.width // LATENT_SCALE_FACTOR)
        y2 = y1 + (self.height // LATENT_SCALE_FACTOR)

        cropped_latents = latents[..., y1:y2, x1:x2]

        name = context.latents.save(tensor=cropped_latents)

        return LatentsOutput.build(latents_name=name, latents=cropped_latents)


@invocation_output("ideal_size_output")
class IdealSizeOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    width: int = OutputField(description="The ideal width of the image (in pixels)")
    height: int = OutputField(description="The ideal height of the image (in pixels)")


@invocation(
    "ideal_size",
    title="Ideal Size",
    tags=["latents", "math", "ideal_size"],
    version="1.0.2",
)
class IdealSizeInvocation(BaseInvocation):
    """Calculates the ideal size for generation to avoid duplication"""

    width: int = InputField(default=1024, description="Final image width")
    height: int = InputField(default=576, description="Final image height")
    unet: UNetField = InputField(default=None, description=FieldDescriptions.unet)
    multiplier: float = InputField(
        default=1.0,
        description="Amount to multiply the model's dimensions by when calculating the ideal size (may result in initial generation artifacts if too large)",
    )

    def trim_to_multiple_of(self, *args, multiple_of=LATENT_SCALE_FACTOR):
        return tuple((x - x % multiple_of) for x in args)

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        aspect = self.width / self.height
        dimension = 512
        if self.unet.unet.base_model == BaseModelType.StableDiffusion2:
            dimension = 768
        elif self.unet.unet.base_model == BaseModelType.StableDiffusionXL:
            dimension = 1024
        dimension = dimension * self.multiplier
        min_dimension = math.floor(dimension * 0.5)
        model_area = dimension * dimension  # hardcoded for now since all models are trained on square images

        if aspect > 1.0:
            init_height = max(min_dimension, math.sqrt(model_area / aspect))
            init_width = init_height * aspect
        else:
            init_width = max(min_dimension, math.sqrt(model_area * aspect))
            init_height = init_width / aspect

        scaled_width, scaled_height = self.trim_to_multiple_of(
            math.floor(init_width),
            math.floor(init_height),
        )

        return IdealSizeOutput(width=scaled_width, height=scaled_height)
