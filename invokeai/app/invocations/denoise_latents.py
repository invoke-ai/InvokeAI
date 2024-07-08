# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)
import inspect
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as T
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_tcd import TCDScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin as Scheduler
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
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
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extensions import (
    ControlNetExt,
    T2IAdapterExt,
    IPAdapterExt,
    RescaleCFGExt,
    PreviewExt,
    InpaintExt,
    TiledDenoiseExt,
    SeamlessExt,
    FreeUExt,
    LoRAPatcherExt,
    PipelineIntermediateState,
)
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionBackend
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    Range,
    SDXLConditioningInfo,
    TextConditioningData,
    TextConditioningRegions,
)
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask
from invokeai.backend.util.silence_warnings import SilenceWarnings
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0


def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelIdentifierField,
    scheduler_name: str,
    seed: int,
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

    # make dpmpp_sde reproducable(seed can be passed only in initializer)
    if scheduler_class is DPMSolverSDEScheduler:
        scheduler_config["noise_sampler_seed"] = seed

    scheduler = scheduler_class.from_config(scheduler_config)
    assert isinstance(scheduler, Scheduler)
    return scheduler


@invocation(
    "denoise_latents",
    title="Denoise Latents",
    tags=["latents", "denoise", "txt2img", "t2i", "t2l", "img2img", "i2i", "l2l"],
    category="latents",
    version="1.5.3",
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
        description=FieldDescriptions.mask,
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
        )
        return conditioning_data

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
                    priority=100,
                )
            )
            # MultiControlNetModel has been refactored out, just need list[ControlNetData]

    @staticmethod
    def parse_ip_adapter_field(
        exit_stack: ExitStack,
        context: InvocationContext,
        ip_adapters: List[IPAdapterField],
        ext_manager: ExtensionsManager,
    ) -> None:

        if ip_adapters is None:
            return

        if not isinstance(ip_adapters, list):
            ip_adapters = [ip_adapters]

        for single_ip_adapter in ip_adapters:
            # `single_ip_adapter.image` could be a list or a single ImageField. Normalize to a list here.
            single_ipa_image_fields = single_ip_adapter.image
            if not isinstance(single_ipa_image_fields, list):
                single_ipa_image_fields = [single_ipa_image_fields]

            single_ipa_images = [context.images.get_pil(image.image_name) for image in single_ipa_image_fields]

            mask_field = single_ip_adapter.mask
            mask = context.tensors.load(mask_field.tensor_name) if mask_field is not None else None

            ext_manager.add_extension(
                IPAdapterExt(
                    node_context=context,
                    exit_stack=exit_stack,
                    model_id=single_ip_adapter.ip_adapter_model,
                    image_encoder_model_id=single_ip_adapter.image_encoder_model,
                    images=single_ipa_images,
                    weight=single_ip_adapter.weight,
                    target_blocks=single_ip_adapter.target_blocks,
                    begin_step_percent=single_ip_adapter.begin_step_percent,
                    end_step_percent=single_ip_adapter.end_step_percent,
                    mask=mask,
                    priority=100,
                )
            )

    @staticmethod
    def parse_t2i_field(
        exit_stack: ExitStack,
        context: InvocationContext,
        t2i_adapters: Optional[Union[T2IAdapterField, list[T2IAdapterField]]],
        ext_manager: ExtensionsManager,
    ) -> None:
        if t2i_adapters is None:
            return

        # Handle the possibility that t2i_adapters could be a list or a single T2IAdapterField.
        if isinstance(t2i_adapters, T2IAdapterField):
            t2i_adapters = [t2i_adapters]

        for t2i_adapter_field in t2i_adapters:
            ext_manager.add_extension(
                T2IAdapterExt(
                    node_context=context,
                    exit_stack=exit_stack,
                    model_id=t2i_adapter_field.t2i_adapter_model,
                    image=context.images.get_pil(t2i_adapter_field.image.image_name),
                    adapter_state=None,
                    weight=t2i_adapter_field.weight,
                    begin_step_percent=t2i_adapter_field.begin_step_percent,
                    end_step_percent=t2i_adapter_field.end_step_percent,
                    resize_mode=t2i_adapter_field.resize_mode,
                    priority=100,
                )
            )

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

        return 1 - mask, masked_latents, self.denoise_mask.gradient

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

    @torch.no_grad()
    @SilenceWarnings()  # This quenches the NSFW nag from diffusers.
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with ExitStack() as exit_stack:
            ext_manager = ExtensionsManager()

            device = TorchDevice.choose_torch_device()
            dtype = TorchDevice.choose_torch_dtype()

            seed, noise, latents = self.prepare_noise_and_latents(context, self.noise, self.latents)
            latents = latents.to(device=device, dtype=dtype)
            if noise is not None:
                noise = noise.to(device=device, dtype=dtype)

            _, _, latent_height, latent_width = latents.shape

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
            )

            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.unet.scheduler,
                scheduler_name=self.scheduler,
                seed=seed,
            )

            timesteps, init_timestep, scheduler_step_kwargs = self.init_scheduler(
                scheduler,
                seed=seed,
                device=device,
                steps=self.steps,
                denoising_start=self.denoising_start,
                denoising_end=self.denoising_end,
            )

            denoise_ctx = DenoiseContext(
                latents=latents,
                timesteps=timesteps,
                init_timestep=init_timestep,
                noise=noise,
                seed=seed,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                unet=None, # unet,
                scheduler=scheduler,
            )


            # get the unet's config so that we can pass the base to sd_step_callback()
            unet_config = context.models.get_config(self.unet.unet.key)

            ### inpaint 
            mask, masked_latents, gradient_mask = self.prep_inpaint_mask(context, latents)
            if mask is not None or unet_config.variant == "inpaint": # ModelVariantType.Inpaint: # is_inpainting_model(unet):
                ext_manager.add_extension(InpaintExt(mask, masked_latents, is_gradient_mask, priority=200))

            ### preview
            def step_callback(state: PipelineIntermediateState) -> None:
                context.util.sd_step_callback(state, unet_config.base)
            ext_manager.add_extension(PreviewExt(step_callback, priority=99999))

            ### cfg rescale
            if self.cfg_rescale_multiplier > 0:
                ext_manager.add_extension(RescaleCFGExt(self.cfg_rescale_multiplier, priority=100))

            ### seamless
            if self.unet.seamless_axes:
                ext_manager.add_extension(SeamlessExt(self.unet.seamless_axes, priority=100))

            ### freeu
            if self.unet.freeu_config:
                ext_manager.add_extension(FreeUExt(self.unet.freeu_config, priority=100))

            ### lora
            if self.unet.loras:
                ext_manager.add_extension(LoRAPatcherExt(
                    node_context=context,
                    loras=self.unet.loras,
                    prefix="lora_unet_",
                    priority=100,
                ))


            #ext_manager.add_extension(
            #    TiledDenoiseExt(
            #        tile_width=1024,
            #        tile_height=1024,
            #        tile_overlap=32,
            #        priority=100,
            #    )
            #)

            # later will be like:
            # for extension_field in self.extensions:
            #    ext = extension_field.to_extension(exit_stack, context)
            #    ext_manager.add_extension(ext)
            self.parse_t2i_field(exit_stack, context, self.t2i_adapter, ext_manager)
            self.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
            self.parse_ip_adapter_field(exit_stack, context, self.ip_adapter, ext_manager)


            # ext: t2i/ip adapter
            ext_manager.modifiers.pre_unet_load(denoise_ctx, ext_manager)

            unet_info = context.models.load(self.unet.unet)
            assert isinstance(unet_info.model, UNet2DConditionModel)
            with (
                unet_info.model_on_device() as (model_state_dict, unet),
                # ext: controlnet
                ext_manager.patch_attention_processor(unet, CustomAttnProcessor2_0),
                # ext: freeu, seamless, ip adapter, lora
                ext_manager.patch_unet(model_state_dict, unet),
            ):
                sd_backend = StableDiffusionBackend(unet, scheduler)
                denoise_ctx.unet = unet
                result_latents = sd_backend.latents_from_embeddings(denoise_ctx, ext_manager)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        result_latents = result_latents.to("cpu") # TODO: detach?
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
