# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)
import inspect
import os
from contextlib import ExitStack
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as T
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.adapter import T2IAdapter
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_tcd import TCDScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin as Scheduler
from PIL import Image
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPVisionModelWithProjection

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.fields import (
    ConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import ModelIdentifierField, UNetField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.model_manager import BaseModelType, ModelVariantType
from invokeai.backend.model_manager.config import AnyModelConfig
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion import PipelineIntermediateState
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext, DenoiseInputs
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    IPAdapterConditioningInfo,
    IPAdapterData,
    Range,
    SDXLConditioningInfo,
    TextConditioningData,
    TextConditioningRegions,
)
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.controlnet import ControlNetExt
from invokeai.backend.stable_diffusion.extensions.freeu import FreeUExt
from invokeai.backend.stable_diffusion.extensions.inpaint import InpaintExt
from invokeai.backend.stable_diffusion.extensions.inpaint_model import InpaintModelExt
from invokeai.backend.stable_diffusion.extensions.lora import LoRAExt
from invokeai.backend.stable_diffusion.extensions.preview import PreviewExt
from invokeai.backend.stable_diffusion.extensions.rescale_cfg import RescaleCFGExt
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.extensions.t2i_adapter import T2IAdapterExt
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.hotfixes import ControlNetModel
from invokeai.backend.util.mask import to_standard_float_mask
from invokeai.backend.util.silence_warnings import SilenceWarnings


def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelIdentifierField,
    scheduler_name: str,
    seed: int,
    unet_config: AnyModelConfig,
) -> Scheduler:
    """Load a scheduler and apply some scheduler-specific overrides."""
    # TODO(ryand): Silently falling back to ddim seems like a bad idea. Look into why this was added and remove if
    # possible.
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP["ddim"])
    orig_scheduler_info = context.models.load(scheduler_info)

    with orig_scheduler_info as orig_scheduler:
        scheduler_config = orig_scheduler.config

    if "_backup" in scheduler_config:
        scheduler_config = scheduler_config["_backup"]
    scheduler_config = {
        **scheduler_config,
        **scheduler_extra_config,  # FIXME
        "_backup": scheduler_config,
    }

    if hasattr(unet_config, "prediction_type"):
        scheduler_config["prediction_type"] = unet_config.prediction_type

    # make dpmpp_sde reproducable(seed can be passed only in initializer)
    if scheduler_class is DPMSolverSDEScheduler:
        scheduler_config["noise_sampler_seed"] = seed

    if scheduler_class is DPMSolverMultistepScheduler or scheduler_class is DPMSolverSinglestepScheduler:
        if scheduler_config["_class_name"] == "DEISMultistepScheduler" and scheduler_config["algorithm_type"] == "deis":
            scheduler_config["algorithm_type"] = "dpmsolver++"

    scheduler = scheduler_class.from_config(scheduler_config)

    # hack copied over from generate.py
    if not hasattr(scheduler, "uses_inpainting_model"):
        scheduler.uses_inpainting_model = lambda: False
    assert isinstance(scheduler, Scheduler)
    return scheduler


@invocation(
    "denoise_latents",
    title="Denoise - SD1.5, SDXL",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.5.4",
)
class DenoiseLatentsInvocation(BaseInvocation):
    """Denoises noisy latents to decodable images"""

    positive_conditioning: Union[ConditioningField, list[ConditioningField]] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection, ui_order=0
    )
    negative_conditioning: Union[ConditioningField, list[ConditioningField]] = InputField(
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
        default=7.5, description=FieldDescriptions.cfg_scale, title="CFG Scale"
    )
    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    scheduler: SCHEDULER_NAME_VALUES = InputField(
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
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
        ui_order=8,
    )

    @field_validator("cfg_scale")
    def ge_one(cls, v: Union[List[float], float]) -> Union[List[float], float]:
        """validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError("cfg_scale must be greater than 1")
        else:
            if v < 1:
                raise ValueError("cfg_scale must be greater than 1")
        return v

    @staticmethod
    def _get_text_embeddings_and_masks(
        cond_list: list[ConditioningField],
        context: InvocationContext,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Union[list[BasicConditioningInfo], list[SDXLConditioningInfo]], list[Optional[torch.Tensor]]]:
        """Get the text embeddings and masks from the input conditioning fields."""
        text_embeddings: Union[list[BasicConditioningInfo], list[SDXLConditioningInfo]] = []
        text_embeddings_masks: list[Optional[torch.Tensor]] = []
        for cond in cond_list:
            cond_data = context.conditioning.load(cond.conditioning_name)
            text_embeddings.append(cond_data.conditionings[0].to(device=device, dtype=dtype))

            mask = cond.mask
            if mask is not None:
                mask = context.tensors.load(mask.tensor_name)
            text_embeddings_masks.append(mask)

        return text_embeddings, text_embeddings_masks

    @staticmethod
    def _preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor], target_height: int, target_width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target height and width.
        If mask is None, returns a mask of all ones with the target height and width.
        If mask is not None, resizes the mask to the target height and width using 'nearest' interpolation.

        Returns:
            torch.Tensor: The processed mask. shape: (1, 1, target_height, target_width).
        """

        if mask is None:
            return torch.ones((1, 1, target_height, target_width), dtype=dtype)

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (target_height, target_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)
        return resized_mask

    @staticmethod
    def _concat_regional_text_embeddings(
        text_conditionings: Union[list[BasicConditioningInfo], list[SDXLConditioningInfo]],
        masks: Optional[list[Optional[torch.Tensor]]],
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
    ) -> tuple[Union[BasicConditioningInfo, SDXLConditioningInfo], Optional[TextConditioningRegions]]:
        """Concatenate regional text embeddings into a single embedding and track the region masks accordingly."""
        if masks is None:
            masks = [None] * len(text_conditionings)
        assert len(text_conditionings) == len(masks)

        is_sdxl = type(text_conditionings[0]) is SDXLConditioningInfo

        all_masks_are_none = all(mask is None for mask in masks)

        text_embedding = []
        pooled_embedding = None
        add_time_ids = None
        cur_text_embedding_len = 0
        processed_masks = []
        embedding_ranges = []

        for prompt_idx, text_embedding_info in enumerate(text_conditionings):
            mask = masks[prompt_idx]

            if is_sdxl:
                # We choose a random SDXLConditioningInfo's pooled_embeds and add_time_ids here, with a preference for
                # prompts without a mask. We prefer prompts without a mask, because they are more likely to contain
                # global prompt information.  In an ideal case, there should be exactly one global prompt without a
                # mask, but we don't enforce this.

                # HACK(ryand): The fact that we have to choose a single pooled_embedding and add_time_ids here is a
                # fundamental interface issue. The SDXL Compel nodes are not designed to be used in the way that we use
                # them for regional prompting. Ideally, the DenoiseLatents invocation should accept a single
                # pooled_embeds tensor and a list of standard text embeds with region masks. This change would be a
                # pretty major breaking change to a popular node, so for now we use this hack.
                if pooled_embedding is None or mask is None:
                    pooled_embedding = text_embedding_info.pooled_embeds
                if add_time_ids is None or mask is None:
                    add_time_ids = text_embedding_info.add_time_ids

            text_embedding.append(text_embedding_info.embeds)
            if not all_masks_are_none:
                embedding_ranges.append(
                    Range(
                        start=cur_text_embedding_len, end=cur_text_embedding_len + text_embedding_info.embeds.shape[1]
                    )
                )
                processed_masks.append(
                    DenoiseLatentsInvocation._preprocess_regional_prompt_mask(
                        mask, latent_height, latent_width, dtype=dtype
                    )
                )

            cur_text_embedding_len += text_embedding_info.embeds.shape[1]

        text_embedding = torch.cat(text_embedding, dim=1)
        assert len(text_embedding.shape) == 3  # batch_size, seq_len, token_len

        regions = None
        if not all_masks_are_none:
            regions = TextConditioningRegions(
                masks=torch.cat(processed_masks, dim=1),
                ranges=embedding_ranges,
            )

        if is_sdxl:
            return (
                SDXLConditioningInfo(embeds=text_embedding, pooled_embeds=pooled_embedding, add_time_ids=add_time_ids),
                regions,
            )
        return BasicConditioningInfo(embeds=text_embedding), regions

    @staticmethod
    def get_conditioning_data(
        context: InvocationContext,
        positive_conditioning_field: Union[ConditioningField, list[ConditioningField]],
        negative_conditioning_field: Union[ConditioningField, list[ConditioningField]],
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
        cfg_scale: float | list[float],
        steps: int,
        cfg_rescale_multiplier: float,
    ) -> TextConditioningData:
        # Normalize positive_conditioning_field and negative_conditioning_field to lists.
        cond_list = positive_conditioning_field
        if not isinstance(cond_list, list):
            cond_list = [cond_list]
        uncond_list = negative_conditioning_field
        if not isinstance(uncond_list, list):
            uncond_list = [uncond_list]

        cond_text_embeddings, cond_text_embedding_masks = DenoiseLatentsInvocation._get_text_embeddings_and_masks(
            cond_list, context, device, dtype
        )
        uncond_text_embeddings, uncond_text_embedding_masks = DenoiseLatentsInvocation._get_text_embeddings_and_masks(
            uncond_list, context, device, dtype
        )

        cond_text_embedding, cond_regions = DenoiseLatentsInvocation._concat_regional_text_embeddings(
            text_conditionings=cond_text_embeddings,
            masks=cond_text_embedding_masks,
            latent_height=latent_height,
            latent_width=latent_width,
            dtype=dtype,
        )
        uncond_text_embedding, uncond_regions = DenoiseLatentsInvocation._concat_regional_text_embeddings(
            text_conditionings=uncond_text_embeddings,
            masks=uncond_text_embedding_masks,
            latent_height=latent_height,
            latent_width=latent_width,
            dtype=dtype,
        )

        if isinstance(cfg_scale, list):
            assert len(cfg_scale) == steps, "cfg_scale (list) must have the same length as the number of steps"

        conditioning_data = TextConditioningData(
            uncond_text=uncond_text_embedding,
            cond_text=cond_text_embedding,
            uncond_regions=uncond_regions,
            cond_regions=cond_regions,
            guidance_scale=cfg_scale,
            guidance_rescale_multiplier=cfg_rescale_multiplier,
        )
        return conditioning_data

    @staticmethod
    def create_pipeline(
        unet: UNet2DConditionModel,
        scheduler: Scheduler,
    ) -> StableDiffusionGeneratorPipeline:
        class FakeVae:
            class FakeVaeConfig:
                def __init__(self) -> None:
                    self.block_out_channels = [0]

            def __init__(self) -> None:
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

    @staticmethod
    def prep_control_data(
        context: InvocationContext,
        control_input: ControlField | list[ControlField] | None,
        latents_shape: List[int],
        device: torch.device,
        exit_stack: ExitStack,
        do_classifier_free_guidance: bool = True,
    ) -> list[ControlNetData] | None:
        # Normalize control_input to a list.
        control_list: list[ControlField]
        if isinstance(control_input, ControlField):
            control_list = [control_input]
        elif isinstance(control_input, list):
            control_list = control_input
        elif control_input is None:
            control_list = []
        else:
            raise ValueError(f"Unexpected control_input type: {type(control_input)}")

        if len(control_list) == 0:
            return None

        # Assuming fixed dimensional scaling of LATENT_SCALE_FACTOR.
        _, _, latent_height, latent_width = latents_shape
        control_height_resize = latent_height * LATENT_SCALE_FACTOR
        control_width_resize = latent_width * LATENT_SCALE_FACTOR

        controlnet_data: list[ControlNetData] = []
        for control_info in control_list:
            control_model = exit_stack.enter_context(context.models.load(control_info.control_model))
            assert isinstance(control_model, ControlNetModel)

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
                device=device,
                dtype=control_model.dtype,
                control_mode=control_info.control_mode,
                resize_mode=control_info.resize_mode,
            )
            control_item = ControlNetData(
                model=control_model,
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

    @staticmethod
    def parse_controlnet_field(
        exit_stack: ExitStack,
        context: InvocationContext,
        control_input: ControlField | list[ControlField] | None,
        ext_manager: ExtensionsManager,
    ) -> None:
        # Normalize control_input to a list.
        control_list: list[ControlField]
        if isinstance(control_input, ControlField):
            control_list = [control_input]
        elif isinstance(control_input, list):
            control_list = control_input
        elif control_input is None:
            control_list = []
        else:
            raise ValueError(f"Unexpected control_input type: {type(control_input)}")

        for control_info in control_list:
            model = exit_stack.enter_context(context.models.load(control_info.control_model))
            ext_manager.add_extension(
                ControlNetExt(
                    model=model,
                    image=context.images.get_pil(control_info.image.image_name),
                    weight=control_info.control_weight,
                    begin_step_percent=control_info.begin_step_percent,
                    end_step_percent=control_info.end_step_percent,
                    control_mode=control_info.control_mode,
                    resize_mode=control_info.resize_mode,
                )
            )

    @staticmethod
    def parse_t2i_adapter_field(
        exit_stack: ExitStack,
        context: InvocationContext,
        t2i_adapters: Optional[Union[T2IAdapterField, list[T2IAdapterField]]],
        ext_manager: ExtensionsManager,
        bgr_mode: bool = False,
    ) -> None:
        if t2i_adapters is None:
            return

        # Handle the possibility that t2i_adapters could be a list or a single T2IAdapterField.
        if isinstance(t2i_adapters, T2IAdapterField):
            t2i_adapters = [t2i_adapters]

        for t2i_adapter_field in t2i_adapters:
            image = context.images.get_pil(t2i_adapter_field.image.image_name)
            if bgr_mode:  # SDXL t2i trained on cv2's BGR outputs, but PIL won't convert straight to BGR
                r, g, b = image.split()
                image = Image.merge("RGB", (b, g, r))
            ext_manager.add_extension(
                T2IAdapterExt(
                    node_context=context,
                    model_id=t2i_adapter_field.t2i_adapter_model,
                    image=context.images.get_pil(t2i_adapter_field.image.image_name),
                    weight=t2i_adapter_field.weight,
                    begin_step_percent=t2i_adapter_field.begin_step_percent,
                    end_step_percent=t2i_adapter_field.end_step_percent,
                    resize_mode=t2i_adapter_field.resize_mode,
                )
            )

    def prep_ip_adapter_image_prompts(
        self,
        context: InvocationContext,
        ip_adapters: List[IPAdapterField],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Run the IPAdapter CLIPVisionModel, returning image prompt embeddings."""
        image_prompts = []
        for single_ip_adapter in ip_adapters:
            with context.models.load(single_ip_adapter.ip_adapter_model) as ip_adapter_model:
                assert isinstance(ip_adapter_model, IPAdapter)
                # `single_ip_adapter.image` could be a list or a single ImageField. Normalize to a list here.
                single_ipa_image_fields = single_ip_adapter.image
                if not isinstance(single_ipa_image_fields, list):
                    single_ipa_image_fields = [single_ipa_image_fields]

                single_ipa_images = [
                    context.images.get_pil(image.image_name, mode="RGB") for image in single_ipa_image_fields
                ]
                with context.models.load(single_ip_adapter.image_encoder_model) as image_encoder_model:
                    assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)
                    # Get image embeddings from CLIP and ImageProjModel.
                    image_prompt_embeds, uncond_image_prompt_embeds = ip_adapter_model.get_image_embeds(
                        single_ipa_images, image_encoder_model
                    )
                    image_prompts.append((image_prompt_embeds, uncond_image_prompt_embeds))

        return image_prompts

    def prep_ip_adapter_data(
        self,
        context: InvocationContext,
        ip_adapters: List[IPAdapterField],
        image_prompts: List[Tuple[torch.Tensor, torch.Tensor]],
        exit_stack: ExitStack,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
    ) -> Optional[List[IPAdapterData]]:
        """If IP-Adapter is enabled, then this function loads the requisite models and adds the image prompt conditioning data."""
        ip_adapter_data_list = []
        for single_ip_adapter, (image_prompt_embeds, uncond_image_prompt_embeds) in zip(
            ip_adapters, image_prompts, strict=True
        ):
            ip_adapter_model = exit_stack.enter_context(context.models.load(single_ip_adapter.ip_adapter_model))

            mask_field = single_ip_adapter.mask
            mask = context.tensors.load(mask_field.tensor_name) if mask_field is not None else None
            mask = self._preprocess_regional_prompt_mask(mask, latent_height, latent_width, dtype=dtype)

            ip_adapter_data_list.append(
                IPAdapterData(
                    ip_adapter_model=ip_adapter_model,
                    weight=single_ip_adapter.weight,
                    target_blocks=single_ip_adapter.target_blocks,
                    begin_step_percent=single_ip_adapter.begin_step_percent,
                    end_step_percent=single_ip_adapter.end_step_percent,
                    ip_adapter_conditioning=IPAdapterConditioningInfo(image_prompt_embeds, uncond_image_prompt_embeds),
                    mask=mask,
                )
            )

        return ip_adapter_data_list if len(ip_adapter_data_list) > 0 else None

    def run_t2i_adapters(
        self,
        context: InvocationContext,
        t2i_adapter: Optional[Union[T2IAdapterField, list[T2IAdapterField]]],
        latents_shape: list[int],
        device: torch.device,
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
            t2i_adapter_model_config = context.models.get_config(t2i_adapter_field.t2i_adapter_model.key)
            image = context.images.get_pil(t2i_adapter_field.image.image_name, mode="RGB")

            # The max_unet_downscale is the maximum amount that the UNet model downscales the latent image internally.
            if t2i_adapter_model_config.base == BaseModelType.StableDiffusion1:
                max_unet_downscale = 8
            elif t2i_adapter_model_config.base == BaseModelType.StableDiffusionXL:
                max_unet_downscale = 4

                # SDXL adapters are trained on cv2's BGR outputs
                r, g, b = image.split()
                image = Image.merge("RGB", (b, g, r))
            else:
                raise ValueError(f"Unexpected T2I-Adapter base model type: '{t2i_adapter_model_config.base}'.")

            t2i_adapter_model: T2IAdapter
            with context.models.load(t2i_adapter_field.t2i_adapter_model) as t2i_adapter_model:
                total_downscale_factor = t2i_adapter_model.total_downscale_factor

                # Note: We have hard-coded `do_classifier_free_guidance=False`. This is because we only want to prepare
                # a single image. If CFG is enabled, we will duplicate the resultant tensor after applying the
                # T2I-Adapter model.
                #
                # Note: We re-use the `prepare_control_image(...)` from ControlNet for T2I-Adapter, because it has many
                # of the same requirements (e.g. preserving binary masks during resize).

                # Assuming fixed dimensional scaling of LATENT_SCALE_FACTOR.
                _, _, latent_height, latent_width = latents_shape
                control_height_resize = latent_height * LATENT_SCALE_FACTOR
                control_width_resize = latent_width * LATENT_SCALE_FACTOR
                t2i_image = prepare_control_image(
                    image=image,
                    do_classifier_free_guidance=False,
                    width=control_width_resize,
                    height=control_height_resize,
                    num_channels=t2i_adapter_model.config["in_channels"],  # mypy treats this as a FrozenDict
                    device=device,
                    dtype=t2i_adapter_model.dtype,
                    resize_mode=t2i_adapter_field.resize_mode,
                )

                # Resize the T2I-Adapter input image.
                # We select the resize dimensions so that after the T2I-Adapter's total_downscale_factor is applied, the
                # result will match the latent image's dimensions after max_unet_downscale is applied.
                # We crop the image to this size so that the positions match the input image on non-standard resolutions
                t2i_input_height = latents_shape[2] // max_unet_downscale * total_downscale_factor
                t2i_input_width = latents_shape[3] // max_unet_downscale * total_downscale_factor
                if t2i_image.shape[2] > t2i_input_height or t2i_image.shape[3] > t2i_input_width:
                    t2i_image = t2i_image[
                        :, :, : min(t2i_image.shape[2], t2i_input_height), : min(t2i_image.shape[3], t2i_input_width)
                    ]

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
    @staticmethod
    def init_scheduler(
        scheduler: Union[Scheduler, ConfigMixin],
        device: torch.device,
        steps: int,
        denoising_start: float,
        denoising_end: float,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert isinstance(scheduler, ConfigMixin)
        if scheduler.config.get("cpu_only", False):
            scheduler.set_timesteps(steps, device="cpu")
            timesteps = scheduler.timesteps.to(device=device)
        else:
            scheduler.set_timesteps(steps, device=device)
            timesteps = scheduler.timesteps

        # skip greater order timesteps
        _timesteps = timesteps[:: scheduler.order]

        # get start timestep index
        t_start_val = int(round(scheduler.config["num_train_timesteps"] * (1 - denoising_start)))
        t_start_idx = len(list(filter(lambda ts: ts >= t_start_val, _timesteps)))

        # get end timestep index
        t_end_val = int(round(scheduler.config["num_train_timesteps"] * (1 - denoising_end)))
        t_end_idx = len(list(filter(lambda ts: ts >= t_end_val, _timesteps[t_start_idx:])))

        # apply order to indexes
        t_start_idx *= scheduler.order
        t_end_idx *= scheduler.order

        init_timestep = timesteps[t_start_idx : t_start_idx + 1]
        timesteps = timesteps[t_start_idx : t_start_idx + t_end_idx]

        scheduler_step_kwargs: Dict[str, Any] = {}
        scheduler_step_signature = inspect.signature(scheduler.step)
        if "generator" in scheduler_step_signature.parameters:
            # At some point, someone decided that schedulers that accept a generator should use the original seed with
            # all bits flipped. I don't know the original rationale for this, but now we must keep it like this for
            # reproducibility.
            #
            # These Invoke-supported schedulers accept a generator as of 2024-06-04:
            #   - DDIMScheduler
            #   - DDPMScheduler
            #   - DPMSolverMultistepScheduler
            #   - EulerAncestralDiscreteScheduler
            #   - EulerDiscreteScheduler
            #   - KDPM2AncestralDiscreteScheduler
            #   - LCMScheduler
            #   - TCDScheduler
            scheduler_step_kwargs.update({"generator": torch.Generator(device=device).manual_seed(seed ^ 0xFFFFFFFF)})
        if isinstance(scheduler, TCDScheduler):
            scheduler_step_kwargs.update({"eta": 1.0})

        return timesteps, init_timestep, scheduler_step_kwargs

    def prep_inpaint_mask(
        self, context: InvocationContext, latents: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        if self.denoise_mask is None:
            return None, None, False

        mask = context.tensors.load(self.denoise_mask.mask_name)
        mask = tv_resize(mask, latents.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
        if self.denoise_mask.masked_latents_name is not None:
            masked_latents = context.tensors.load(self.denoise_mask.masked_latents_name)
        else:
            masked_latents = torch.where(mask < 0.5, 0.0, latents)

        return mask, masked_latents, self.denoise_mask.gradient

    @staticmethod
    def prepare_noise_and_latents(
        context: InvocationContext, noise_field: LatentsField | None, latents_field: LatentsField | None
    ) -> Tuple[int, torch.Tensor | None, torch.Tensor]:
        """Depending on the workflow, we expect different combinations of noise and latents to be provided. This
        function handles preparing these values accordingly.

        Expected workflows:
        - Text-to-Image Denoising: `noise` is provided, `latents` is not. `latents` is initialized to zeros.
        - Image-to-Image Denoising: `noise` and `latents` are both provided.
        - Text-to-Image SDXL Refiner Denoising: `latents` is provided, `noise` is not.
        - Image-to-Image SDXL Refiner Denoising: `latents` is provided, `noise` is not.

        NOTE(ryand): I wrote this docstring, but I am not the original author of this code. There may be other workflows
        I haven't considered.
        """
        noise = None
        if noise_field is not None:
            noise = context.tensors.load(noise_field.latents_name)

        if latents_field is not None:
            latents = context.tensors.load(latents_field.latents_name)
        elif noise is not None:
            latents = torch.zeros_like(noise)
        else:
            raise ValueError("'latents' or 'noise' must be provided!")

        if noise is not None and noise.shape[1:] != latents.shape[1:]:
            raise ValueError(f"Incompatible 'noise' and 'latents' shapes: {latents.shape=} {noise.shape=}")

        # The seed comes from (in order of priority): the noise field, the latents field, or 0.
        seed = 0
        if noise_field is not None and noise_field.seed is not None:
            seed = noise_field.seed
        elif latents_field is not None and latents_field.seed is not None:
            seed = latents_field.seed
        else:
            seed = 0

        return seed, noise, latents

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        if os.environ.get("USE_MODULAR_DENOISE", False):
            return self._new_invoke(context)
        else:
            return self._old_invoke(context)

    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def _new_invoke(self, context: InvocationContext) -> LatentsOutput:
        ext_manager = ExtensionsManager(is_canceled=context.util.is_canceled)

        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_torch_dtype()

        seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
        _, _, latent_height, latent_width = latents.shape

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        conditioning_data = self.get_conditioning_data(
            context=context,
            positive_conditioning_field=self.positive_conditioning,
            negative_conditioning_field=self.negative_conditioning,
            cfg_scale=self.cfg_scale,
            steps=self.steps,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
            dtype=dtype,
            # TODO: old backend, remove
            cfg_rescale_multiplier=self.cfg_rescale_multiplier,
        )

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
            seed=seed,
            unet_config=unet_config,
        )

        timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
            scheduler,
            seed=seed,
            device=device,
            steps=self.steps,
            denoising_start=self.denoising_start,
            denoising_end=self.denoising_end,
        )

        ### preview
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        ext_manager.add_extension(PreviewExt(step_callback))

        ### cfg rescale
        if self.cfg_rescale_multiplier > 0:
            ext_manager.add_extension(RescaleCFGExt(self.cfg_rescale_multiplier))

        ### freeu
        if self.unet.freeu_config:
            ext_manager.add_extension(FreeUExt(self.unet.freeu_config))

        ### lora
        if self.unet.loras:
            for lora_field in self.unet.loras:
                ext_manager.add_extension(
                    LoRAExt(
                        node_context=context,
                        model_id=lora_field.lora,
                        weight=lora_field.weight,
                    )
                )
        ### seamless
        if self.unet.seamless_axes:
            ext_manager.add_extension(SeamlessExt(self.unet.seamless_axes))

        ### inpaint
        mask, masked_latents, is_gradient_mask = self.prep_inpaint_mask(context, latents)
        # NOTE: We used to identify inpainting models by inspecting the shape of the loaded UNet model weights. Now we
        # use the ModelVariantType config. During testing, there was a report of a user with models that had an
        # incorrect ModelVariantType value. Re-installing the model fixed the issue. If this issue turns out to be
        # prevalent, we will have to revisit how we initialize the inpainting extensions.
        if unet_config.variant == ModelVariantType.Inpaint:
            ext_manager.add_extension(InpaintModelExt(mask, masked_latents, is_gradient_mask))
        elif mask is not None:
            ext_manager.add_extension(InpaintExt(mask, is_gradient_mask))

        # Initialize context for modular denoise
        latents = latents.to(device=device, dtype=dtype)
        if noise is not None:
            noise = noise.to(device=device, dtype=dtype)
        denoise_ctx = DenoiseContext(
            inputs=DenoiseInputs(
                orig_latents=latents,
                timesteps=timesteps,
                init_timestep=init_timestep,
                noise=noise,
                seed=seed,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                attention_processor_cls=CustomAttnProcessor2_0,
            ),
            unet=None,
            scheduler=scheduler,
        )

        # context for loading additional models
        with ExitStack() as exit_stack:
            # later should be smth like:
            # for extension_field in self.extensions:
            #    ext = extension_field.to_extension(exit_stack, context, ext_manager)
            #    ext_manager.add_extension(ext)
            self.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
            bgr_mode = self.unet.unet.base == BaseModelType.StableDiffusionXL
            self.parse_t2i_adapter_field(exit_stack, context, self.t2i_adapter, ext_manager, bgr_mode)

            # ext: t2i/ip adapter
            ext_manager.run_callback(ExtensionCallbackType.SETUP, denoise_ctx)

            with (
                context.models.load(self.unet.unet).model_on_device() as (cached_weights, unet),
                ModelPatcher.patch_unet_attention_processor(unet, denoise_ctx.inputs.attention_processor_cls),
                # ext: controlnet
                ext_manager.patch_extensions(denoise_ctx),
                # ext: freeu, seamless, ip adapter, lora
                ext_manager.patch_unet(unet, cached_weights),
            ):
                sd_backend = StableDiffusionBackend(unet, scheduler)
                denoise_ctx.unet = unet
                result_latents = sd_backend.latents_from_embeddings(denoise_ctx, ext_manager)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.detach().to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)

    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def _old_invoke(self, context: InvocationContext) -> LatentsOutput:
        device = TorchDevice.choose_torch_device()
        seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)

        mask, masked_latents, gradient_mask = self.prep_inpaint_mask(context, latents)
        # At this point, the mask ranges from 0 (leave unchanged) to 1 (inpaint).
        # We invert the mask here for compatibility with the old backend implementation.
        if mask is not None:
            mask = 1 - mask

        # TODO(ryand): I have hard-coded `do_classifier_free_guidance=True` to mirror the behaviour of ControlNets,
        # below. Investigate whether this is appropriate.
        t2i_adapter_data = self.run_t2i_adapters(
            context,
            self.t2i_adapter,
            latents.shape,
            device=device,
            do_classifier_free_guidance=True,
        )

        ip_adapters: List[IPAdapterField] = []
        if self.ip_adapter is not None:
            # ip_adapter could be a list or a single IPAdapterField. Normalize to a list here.
            if isinstance(self.ip_adapter, list):
                ip_adapters = self.ip_adapter
            else:
                ip_adapters = [self.ip_adapter]

        # If there are IP adapters, the following line runs the adapters' CLIPVision image encoders to return
        # a series of image conditioning embeddings. This is being done here rather than in the
        # big model context below in order to use less VRAM on low-VRAM systems.
        # The image prompts are then passed to prep_ip_adapter_data().
        image_prompts = self.prep_ip_adapter_image_prompts(context=context, ip_adapters=ip_adapters)

        # get the unet's config so that we can pass the base to sd_step_callback()
        unet_config = context.models.get_config(self.unet.unet.key)

        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, unet_config.base)

        def _lora_loader() -> Iterator[Tuple[ModelPatchRaw, float]]:
            for lora in self.unet.loras:
                lora_info = context.models.load(lora.lora)
                assert isinstance(lora_info.model, ModelPatchRaw)
                yield (lora_info.model, lora.weight)
                del lora_info
            return

        with (
            ExitStack() as exit_stack,
            context.models.load(self.unet.unet).model_on_device() as (cached_weights, unet),
            ModelPatcher.apply_freeu(unet, self.unet.freeu_config),
            SeamlessExt.static_patch_model(unet, self.unet.seamless_axes),  # FIXME
            # Apply the LoRA after unet has been moved to its target device for faster patching.
            LayerPatcher.apply_smart_model_patches(
                model=unet,
                patches=_lora_loader(),
                prefix="lora_unet_",
                dtype=unet.dtype,
                cached_weights=cached_weights,
            ),
        ):
            assert isinstance(unet, UNet2DConditionModel)
            latents = latents.to(device=device, dtype=unet.dtype)
            if noise is not None:
                noise = noise.to(device=device, dtype=unet.dtype)
            if mask is not None:
                mask = mask.to(device=device, dtype=unet.dtype)
            if masked_latents is not None:
                masked_latents = masked_latents.to(device=device, dtype=unet.dtype)

            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.unet.scheduler,
                scheduler_name=self.scheduler,
                seed=seed,
                unet_config=unet_config,
            )

            pipeline = self.create_pipeline(unet, scheduler)

            _, _, latent_height, latent_width = latents.shape
            conditioning_data = self.get_conditioning_data(
                context=context,
                positive_conditioning_field=self.positive_conditioning,
                negative_conditioning_field=self.negative_conditioning,
                device=device,
                dtype=unet.dtype,
                latent_height=latent_height,
                latent_width=latent_width,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                cfg_rescale_multiplier=self.cfg_rescale_multiplier,
            )

            controlnet_data = self.prep_control_data(
                context=context,
                control_input=self.control,
                latents_shape=latents.shape,
                device=device,
                # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                do_classifier_free_guidance=True,
                exit_stack=exit_stack,
            )

            ip_adapter_data = self.prep_ip_adapter_data(
                context=context,
                ip_adapters=ip_adapters,
                image_prompts=image_prompts,
                exit_stack=exit_stack,
                latent_height=latent_height,
                latent_width=latent_width,
                dtype=unet.dtype,
            )

            timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
                scheduler,
                device=device,
                steps=self.steps,
                denoising_start=self.denoising_start,
                denoising_end=self.denoising_end,
                seed=seed,
            )

            result_latents = pipeline.latents_from_embeddings(
                latents=latents,
                timesteps=timesteps,
                init_timestep=init_timestep,
                noise=noise,
                seed=seed,
                mask=mask,
                masked_latents=masked_latents,
                is_gradient_mask=gradient_mask,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                control_data=controlnet_data,
                ip_adapter_data=ip_adapter_data,
                t2i_adapter_data=t2i_adapter_data,
                callback=step_callback,
            )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
