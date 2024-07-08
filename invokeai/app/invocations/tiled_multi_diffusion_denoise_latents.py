from contextlib import ExitStack
from typing import Optional, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.controlnet_image_processors import ControlField
from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation, get_scheduler
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import UNetField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionBackend
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
from invokeai.backend.stable_diffusion.extensions import (
    FreeUExt,
    LoRAPatcherExt,
    PipelineIntermediateState,
    PreviewExt,
    RescaleCFGExt,
    SeamlessExt,
    TiledDenoiseExt,
)
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "tiled_multi_diffusion_denoise_latents",
    title="Tiled Multi-Diffusion Denoise Latents",
    tags=["upscale", "denoise"],
    category="latents",
    classification=Classification.Beta,
    version="1.0.0",
)
class TiledMultiDiffusionDenoiseLatents(BaseInvocation):
    """Tiled Multi-Diffusion denoising.

    This node handles automatically tiling the input image, and is primarily intended for global refinement of images
    in tiled upscaling workflows. Future Multi-Diffusion nodes should allow the user to specify custom regions with
    different parameters for each region to harness the full power of Multi-Diffusion.

    This node has a similar interface to the `DenoiseLatents` node, but it has a reduced feature set (no IP-Adapter,
    T2I-Adapter, masking, etc.).
    """

    positive_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    noise: LatentsField | None = InputField(
        default=None,
        description=FieldDescriptions.noise,
        input=Input.Connection,
    )
    latents: LatentsField | None = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    tile_height: int = InputField(
        default=1024, gt=0, multiple_of=LATENT_SCALE_FACTOR, description="Height of the tiles in image space."
    )
    tile_width: int = InputField(
        default=1024, gt=0, multiple_of=LATENT_SCALE_FACTOR, description="Width of the tiles in image space."
    )
    tile_overlap: int = InputField(
        default=32,
        multiple_of=LATENT_SCALE_FACTOR,
        gt=0,
        description="The overlap between adjacent tiles in pixel space. (Of course, tile merging is applied in latent "
        "space.) Tiles will be cropped during merging (if necessary) to ensure that they overlap by exactly this "
        "amount.",
    )
    steps: int = InputField(default=18, gt=0, description=FieldDescriptions.steps)
    cfg_scale: float | list[float] = InputField(default=6.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
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
    )
    cfg_rescale_multiplier: float = InputField(
        title="CFG Rescale Multiplier", default=0, ge=0, lt=1, description=FieldDescriptions.cfg_rescale_multiplier
    )
    control: ControlField | list[ControlField] | None = InputField(
        default=None,
        input=Input.Connection,
    )
    t2i_adapter: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = InputField(
        description=FieldDescriptions.t2i_adapter,
        title="T2I-Adapter",
        default=None,
        input=Input.Connection,
    )
    ip_adapter: Optional[Union[IPAdapterField, list[IPAdapterField]]] = InputField(
        description=FieldDescriptions.ip_adapter,
        title="IP-Adapter",
        default=None,
        input=Input.Connection,
    )

    @field_validator("cfg_scale")
    def ge_one(cls, v: list[float] | float) -> list[float] | float:
        """Validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError("cfg_scale must be greater than 1")
        else:
            if v < 1:
                raise ValueError("cfg_scale must be greater than 1")
        return v

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with ExitStack() as exit_stack:
            ext_manager = ExtensionsManager()

            device = TorchDevice.choose_torch_device()
            dtype = TorchDevice.choose_torch_dtype()

            seed, noise, latents = DenoiseLatentsInvocation.prepare_noise_and_latents(context, self.noise, self.latents)
            latents = latents.to(device=device, dtype=dtype)
            if noise is not None:
                noise = noise.to(device=device, dtype=dtype)

            _, _, latent_height, latent_width = latents.shape

            conditioning_data = DenoiseLatentsInvocation.get_conditioning_data(
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

            timesteps, init_timestep, scheduler_step_kwargs = DenoiseLatentsInvocation.init_scheduler(
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
                unet=None,
                scheduler=scheduler,
            )

            # get the unet's config so that we can pass the base to sd_step_callback()
            unet_config = context.models.get_config(self.unet.unet.key)

            ### inpaint
            # mask, masked_latents, is_gradient_mask = self.prep_inpaint_mask(context, latents)
            # if mask is not None or unet_config.variant == "inpaint": # ModelVariantType.Inpaint: # is_inpainting_model(unet):
            #     ext_manager.add_extension(InpaintExt(mask, masked_latents, is_gradient_mask, priority=200))

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
                ext_manager.add_extension(
                    LoRAPatcherExt(
                        node_context=context,
                        loras=self.unet.loras,
                        prefix="lora_unet_",
                        priority=100,
                    )
                )

            ### tiled denoise
            ext_manager.add_extension(
                TiledDenoiseExt(
                    tile_width=self.tile_width,
                    tile_height=self.tile_height,
                    tile_overlap=self.tile_overlap,
                    priority=100,
                )
            )

            # later will be like:
            # for extension_field in self.extensions:
            #    ext = extension_field.to_extension(exit_stack, context)
            #    ext_manager.add_extension(ext)
            DenoiseLatentsInvocation.parse_t2i_field(exit_stack, context, self.t2i_adapter, ext_manager)
            DenoiseLatentsInvocation.parse_controlnet_field(exit_stack, context, self.control, ext_manager)
            # TODO: works fine with tiled too?
            DenoiseLatentsInvocation.parse_ip_adapter_field(exit_stack, context, self.ip_adapter, ext_manager)

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

        result_latents = result_latents.to("cpu")
        # TODO(ryand): I copied this from DenoiseLatentsInvocation. I'm not sure if it's actually important.
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=result_latents)
        return LatentsOutput.build(latents_name=name, latents=result_latents, seed=None)
