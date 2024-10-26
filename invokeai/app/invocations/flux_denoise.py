from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    ImageField,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.extensions.inpaint_extension import InpaintExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.ip_adapter.xlabs_ip_adapter_flux import XlabsIpAdapterFlux
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional,
    generate_img_ids,
    get_noise,
    get_schedule,
    pack,
    unpack,
)
from invokeai.backend.lora.conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.lora.lora_patcher import LoRAPatcher
from invokeai.backend.model_manager.config import ModelFormat
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_denoise",
    title="FLUX Denoise",
    tags=["image", "flux"],
    category="image",
    version="3.2.0",
    classification=Classification.Prototype,
)
class FluxDenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run denoising process with a FLUX transformer model."""

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    # denoise_mask is used for image-to-image inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )
    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    positive_text_conditioning: FluxConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_text_conditioning: FluxConditioningField | None = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    cfg_scale: float | list[float] = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    cfg_scale_start_step: int = InputField(
        default=0,
        title="CFG Scale Start Step",
        description="Index of the first step to apply cfg_scale. Negative indices count backwards from the "
        + "the last step (e.g. a value of -1 refers to the final step).",
    )
    cfg_scale_end_step: int = InputField(
        default=-1,
        title="CFG Scale End Step",
        description="Index of the last step to apply cfg_scale. Negative indices count backwards from the "
        + "last step (e.g. a value of -1 refers to the final step).",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(
        default=4, description="Number of diffusion steps. Recommended values are schnell: 4, dev: 50."
    )
    guidance: float = InputField(
        default=4.0,
        description="The guidance strength. Higher values adhere more strictly to the prompt, and will produce less diverse images. FLUX dev only, ignored for schnell.",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    control: FluxControlNetField | list[FluxControlNetField] | None = InputField(
        default=None, input=Input.Connection, description="ControlNet models."
    )
    controlnet_vae: VAEField | None = InputField(
        default=None,
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    ip_adapter: IPAdapterField | list[IPAdapterField] | None = InputField(
        description=FieldDescriptions.ip_adapter, title="IP-Adapter", default=None, input=Input.Connection
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _load_text_conditioning(
        self, context: InvocationContext, conditioning_name: str, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the conditioning data.
        cond_data = context.conditioning.load(conditioning_name)
        assert len(cond_data.conditionings) == 1
        flux_conditioning = cond_data.conditionings[0]
        assert isinstance(flux_conditioning, FLUXConditioningInfo)
        flux_conditioning = flux_conditioning.to(dtype=dtype)
        t5_embeddings = flux_conditioning.t5_embeds
        clip_embeddings = flux_conditioning.clip_embeds
        return t5_embeddings, clip_embeddings

    def _run_diffusion(
        self,
        context: InvocationContext,
    ):
        inference_dtype = torch.bfloat16

        # Load the conditioning data.
        pos_t5_embeddings, pos_clip_embeddings = self._load_text_conditioning(
            context, self.positive_text_conditioning.conditioning_name, inference_dtype
        )
        neg_t5_embeddings: torch.Tensor | None = None
        neg_clip_embeddings: torch.Tensor | None = None
        if self.negative_text_conditioning is not None:
            neg_t5_embeddings, neg_clip_embeddings = self._load_text_conditioning(
                context, self.negative_text_conditioning.conditioning_name, inference_dtype
            )

        # Load the input latents, if provided.
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=TorchDevice.choose_torch_device(), dtype=inference_dtype)

        # Prepare input noise.
        noise = get_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=TorchDevice.choose_torch_device(),
            dtype=inference_dtype,
            seed=self.seed,
        )

        transformer_info = context.models.load(self.transformer.transformer)
        is_schnell = "schnell" in transformer_info.config.config_path

        # Calculate the timestep schedule.
        image_seq_len = noise.shape[-1] * noise.shape[-2] // 4
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=image_seq_len,
            shift=not is_schnell,
        )

        # Clip the timesteps schedule based on denoising_start and denoising_end.
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

        # Prepare input latent image.
        if init_latents is not None:
            # If init_latents is provided, we are doing image-to-image.

            if is_schnell:
                context.logger.warning(
                    "Running image-to-image with a FLUX schnell model. This is not recommended. The results are likely "
                    "to be poor. Consider using a FLUX dev model instead."
                )

            # Noise the orig_latents by the appropriate amount for the first timestep.
            t_0 = timesteps[0]
            x = t_0 * noise + (1.0 - t_0) * init_latents
        else:
            # init_latents are not provided, so we are not doing image-to-image (i.e. we are starting from pure noise).
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")

            x = noise

        # If len(timesteps) == 1, then short-circuit. We are just noising the input latents, but not taking any
        # denoising steps.
        if len(timesteps) <= 1:
            return x

        inpaint_mask = self._prep_inpaint_mask(context, x)

        b, _c, latent_h, latent_w = x.shape
        img_ids = generate_img_ids(h=latent_h, w=latent_w, batch_size=b, device=x.device, dtype=x.dtype)

        pos_bs, pos_t5_seq_len, _ = pos_t5_embeddings.shape
        pos_txt_ids = torch.zeros(
            pos_bs, pos_t5_seq_len, 3, dtype=inference_dtype, device=TorchDevice.choose_torch_device()
        )
        neg_txt_ids: torch.Tensor | None = None
        if neg_t5_embeddings is not None:
            neg_bs, neg_t5_seq_len, _ = neg_t5_embeddings.shape
            neg_txt_ids = torch.zeros(
                neg_bs, neg_t5_seq_len, 3, dtype=inference_dtype, device=TorchDevice.choose_torch_device()
            )

        # Pack all latent tensors.
        init_latents = pack(init_latents) if init_latents is not None else None
        inpaint_mask = pack(inpaint_mask) if inpaint_mask is not None else None
        noise = pack(noise)
        x = pack(x)

        # Now that we have 'packed' the latent tensors, verify that we calculated the image_seq_len correctly.
        assert image_seq_len == x.shape[1]

        # Prepare inpaint extension.
        inpaint_extension: InpaintExtension | None = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = InpaintExtension(
                init_latents=init_latents,
                inpaint_mask=inpaint_mask,
                noise=noise,
            )

        # Compute the IP-Adapter image prompt clip embeddings.
        # We do this before loading other models to minimize peak memory.
        # TODO(ryand): We should really do this in a separate invocation to benefit from caching.
        ip_adapter_fields = self._normalize_ip_adapter_fields()
        pos_image_prompt_clip_embeds, neg_image_prompt_clip_embeds = self._prep_ip_adapter_image_prompt_clip_embeds(
            ip_adapter_fields, context
        )

        cfg_scale = self.prep_cfg_scale(
            cfg_scale=self.cfg_scale,
            timesteps=timesteps,
            cfg_scale_start_step=self.cfg_scale_start_step,
            cfg_scale_end_step=self.cfg_scale_end_step,
        )

        with ExitStack() as exit_stack:
            # Prepare ControlNet extensions.
            # Note: We do this before loading the transformer model to minimize peak memory (see implementation).
            controlnet_extensions = self._prep_controlnet_extensions(
                context=context,
                exit_stack=exit_stack,
                latent_height=latent_h,
                latent_width=latent_w,
                dtype=inference_dtype,
                device=x.device,
            )

            # Load the transformer model.
            (cached_weights, transformer) = exit_stack.enter_context(transformer_info.model_on_device())
            assert isinstance(transformer, Flux)
            config = transformer_info.config
            assert config is not None

            # Apply LoRA models to the transformer.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            if config.format in [ModelFormat.Checkpoint]:
                # The model is non-quantized, so we can apply the LoRA weights directly into the model.
                exit_stack.enter_context(
                    LoRAPatcher.apply_lora_patches(
                        model=transformer,
                        patches=self._lora_iterator(context),
                        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                        cached_weights=cached_weights,
                    )
                )
            elif config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                # The model is quantized, so apply the LoRA weights as sidecar layers. This results in slower inference,
                # than directly patching the weights, but is agnostic to the quantization format.
                exit_stack.enter_context(
                    LoRAPatcher.apply_lora_sidecar_patches(
                        model=transformer,
                        patches=self._lora_iterator(context),
                        prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                        dtype=inference_dtype,
                    )
                )
            else:
                raise ValueError(f"Unsupported model format: {config.format}")

            # Prepare IP-Adapter extensions.
            pos_ip_adapter_extensions, neg_ip_adapter_extensions = self._prep_ip_adapter_extensions(
                pos_image_prompt_clip_embeds=pos_image_prompt_clip_embeds,
                neg_image_prompt_clip_embeds=neg_image_prompt_clip_embeds,
                ip_adapter_fields=ip_adapter_fields,
                context=context,
                exit_stack=exit_stack,
                dtype=inference_dtype,
            )

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=pos_t5_embeddings,
                txt_ids=pos_txt_ids,
                vec=pos_clip_embeddings,
                neg_txt=neg_t5_embeddings,
                neg_txt_ids=neg_txt_ids,
                neg_vec=neg_clip_embeddings,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                guidance=self.guidance,
                cfg_scale=cfg_scale,
                inpaint_extension=inpaint_extension,
                controlnet_extensions=controlnet_extensions,
                pos_ip_adapter_extensions=pos_ip_adapter_extensions,
                neg_ip_adapter_extensions=neg_ip_adapter_extensions,
            )

        x = unpack(x.float(), self.height, self.width)
        return x

    @classmethod
    def prep_cfg_scale(
        cls, cfg_scale: float | list[float], timesteps: list[float], cfg_scale_start_step: int, cfg_scale_end_step: int
    ) -> list[float]:
        """Prepare the cfg_scale schedule.

        - Clips the cfg_scale schedule based on cfg_scale_start_step and cfg_scale_end_step.
        - If cfg_scale is a list, then it is assumed to be a schedule and is returned as-is.
        - If cfg_scale is a scalar, then a linear schedule is created from cfg_scale_start_step to cfg_scale_end_step.
        """
        # num_steps is the number of denoising steps, which is one less than the number of timesteps.
        num_steps = len(timesteps) - 1

        # Normalize cfg_scale to a list if it is a scalar.
        cfg_scale_list: list[float]
        if isinstance(cfg_scale, float):
            cfg_scale_list = [cfg_scale] * num_steps
        elif isinstance(cfg_scale, list):
            cfg_scale_list = cfg_scale
        else:
            raise ValueError(f"Unsupported cfg_scale type: {type(cfg_scale)}")
        assert len(cfg_scale_list) == num_steps

        # Handle negative indices for cfg_scale_start_step and cfg_scale_end_step.
        start_step_index = cfg_scale_start_step
        if start_step_index < 0:
            start_step_index = num_steps + start_step_index
        end_step_index = cfg_scale_end_step
        if end_step_index < 0:
            end_step_index = num_steps + end_step_index

        # Validate the start and end step indices.
        if not (0 <= start_step_index < num_steps):
            raise ValueError(f"Invalid cfg_scale_start_step. Out of range: {cfg_scale_start_step}.")
        if not (0 <= end_step_index < num_steps):
            raise ValueError(f"Invalid cfg_scale_end_step. Out of range: {cfg_scale_end_step}.")
        if start_step_index > end_step_index:
            raise ValueError(
                f"cfg_scale_start_step ({cfg_scale_start_step}) must be before cfg_scale_end_step "
                + f"({cfg_scale_end_step})."
            )

        # Set values outside the start and end step indices to 1.0. This is equivalent to disabling cfg_scale for those
        # steps.
        clipped_cfg_scale = [1.0] * num_steps
        clipped_cfg_scale[start_step_index : end_step_index + 1] = cfg_scale_list[start_step_index : end_step_index + 1]

        return clipped_cfg_scale

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        """Prepare the inpaint mask.

        - Loads the mask
        - Resizes if necessary
        - Casts to same device/dtype as latents
        - Expands mask to the same shape as latents so that they line up after 'packing'

        Args:
            context (InvocationContext): The invocation context, for loading the inpaint mask.
            latents (torch.Tensor): A latent image tensor. In 'unpacked' format. Used to determine the target shape,
                device, and dtype for the inpaint mask.

        Returns:
            torch.Tensor | None: Inpaint mask. Values of 0.0 represent the regions to be fully denoised, and 1.0
                represent the regions to be preserved.
        """
        if self.denoise_mask is None:
            return None

        mask = context.tensors.load(self.denoise_mask.mask_name)

        # The input denoise_mask contains values in [0, 1], where 0.0 represents the regions to be fully denoised, and
        # 1.0 represents the regions to be preserved.
        # We invert the mask so that the regions to be preserved are 0.0 and the regions to be denoised are 1.0.
        mask = 1.0 - mask

        _, _, latent_height, latent_width = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_height, latent_width],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )

        mask = mask.to(device=latents.device, dtype=latents.dtype)

        # Expand the inpaint mask to the same shape as `latents` so that when we 'pack' `mask` it lines up with
        # `latents`.
        return mask.expand_as(latents)

    def _prep_controlnet_extensions(
        self,
        context: InvocationContext,
        exit_stack: ExitStack,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[XLabsControlNetExtension | InstantXControlNetExtension]:
        # Normalize the controlnet input to list[ControlField].
        controlnets: list[FluxControlNetField]
        if self.control is None:
            controlnets = []
        elif isinstance(self.control, FluxControlNetField):
            controlnets = [self.control]
        elif isinstance(self.control, list):
            controlnets = self.control
        else:
            raise ValueError(f"Unsupported controlnet type: {type(self.control)}")

        # TODO(ryand): Add a field to the model config so that we can distinguish between XLabs and InstantX ControlNets
        # before loading the models. Then make sure that all VAE encoding is done before loading the ControlNets to
        # minimize peak memory.

        # First, load the ControlNet models so that we can determine the ControlNet types.
        controlnet_models = [context.models.load(controlnet.control_model) for controlnet in controlnets]

        # Calculate the controlnet conditioning tensors.
        # We do this before loading the ControlNet models because it may require running the VAE, and we are trying to
        # keep peak memory down.
        controlnet_conds: list[torch.Tensor] = []
        for controlnet, controlnet_model in zip(controlnets, controlnet_models, strict=True):
            image = context.images.get_pil(controlnet.image.image_name)
            if isinstance(controlnet_model.model, InstantXControlNetFlux):
                if self.controlnet_vae is None:
                    raise ValueError("A ControlNet VAE is required when using an InstantX FLUX ControlNet.")
                vae_info = context.models.load(self.controlnet_vae.vae)
                controlnet_conds.append(
                    InstantXControlNetExtension.prepare_controlnet_cond(
                        controlnet_image=image,
                        vae_info=vae_info,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        dtype=dtype,
                        device=device,
                        resize_mode=controlnet.resize_mode,
                    )
                )
            elif isinstance(controlnet_model.model, XLabsControlNetFlux):
                controlnet_conds.append(
                    XLabsControlNetExtension.prepare_controlnet_cond(
                        controlnet_image=image,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        dtype=dtype,
                        device=device,
                        resize_mode=controlnet.resize_mode,
                    )
                )

        # Finally, load the ControlNet models and initialize the ControlNet extensions.
        controlnet_extensions: list[XLabsControlNetExtension | InstantXControlNetExtension] = []
        for controlnet, controlnet_cond, controlnet_model in zip(
            controlnets, controlnet_conds, controlnet_models, strict=True
        ):
            model = exit_stack.enter_context(controlnet_model)

            if isinstance(model, XLabsControlNetFlux):
                controlnet_extensions.append(
                    XLabsControlNetExtension(
                        model=model,
                        controlnet_cond=controlnet_cond,
                        weight=controlnet.control_weight,
                        begin_step_percent=controlnet.begin_step_percent,
                        end_step_percent=controlnet.end_step_percent,
                    )
                )
            elif isinstance(model, InstantXControlNetFlux):
                instantx_control_mode: torch.Tensor | None = None
                if controlnet.instantx_control_mode is not None and controlnet.instantx_control_mode >= 0:
                    instantx_control_mode = torch.tensor(controlnet.instantx_control_mode, dtype=torch.long)
                    instantx_control_mode = instantx_control_mode.reshape([-1, 1])

                controlnet_extensions.append(
                    InstantXControlNetExtension(
                        model=model,
                        controlnet_cond=controlnet_cond,
                        instantx_control_mode=instantx_control_mode,
                        weight=controlnet.control_weight,
                        begin_step_percent=controlnet.begin_step_percent,
                        end_step_percent=controlnet.end_step_percent,
                    )
                )
            else:
                raise ValueError(f"Unsupported ControlNet model type: {type(model)}")

        return controlnet_extensions

    def _normalize_ip_adapter_fields(self) -> list[IPAdapterField]:
        if self.ip_adapter is None:
            return []
        elif isinstance(self.ip_adapter, IPAdapterField):
            return [self.ip_adapter]
        elif isinstance(self.ip_adapter, list):
            return self.ip_adapter
        else:
            raise ValueError(f"Unsupported IP-Adapter type: {type(self.ip_adapter)}")

    def _prep_ip_adapter_image_prompt_clip_embeds(
        self,
        ip_adapter_fields: list[IPAdapterField],
        context: InvocationContext,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run the IPAdapter CLIPVisionModel, returning image prompt embeddings."""
        clip_image_processor = CLIPImageProcessor()

        pos_image_prompt_clip_embeds: list[torch.Tensor] = []
        neg_image_prompt_clip_embeds: list[torch.Tensor] = []
        for ip_adapter_field in ip_adapter_fields:
            # `ip_adapter_field.image` could be a list or a single ImageField. Normalize to a list here.
            ipa_image_fields: list[ImageField]
            if isinstance(ip_adapter_field.image, ImageField):
                ipa_image_fields = [ip_adapter_field.image]
            elif isinstance(ip_adapter_field.image, list):
                ipa_image_fields = ip_adapter_field.image
            else:
                raise ValueError(f"Unsupported IP-Adapter image type: {type(ip_adapter_field.image)}")

            if len(ipa_image_fields) != 1:
                raise ValueError(
                    f"FLUX IP-Adapter only supports a single image prompt (received {len(ipa_image_fields)})."
                )

            ipa_images = [context.images.get_pil(image.image_name, mode="RGB") for image in ipa_image_fields]

            pos_images: list[npt.NDArray[np.uint8]] = []
            neg_images: list[npt.NDArray[np.uint8]] = []
            for ipa_image in ipa_images:
                assert ipa_image.mode == "RGB"
                pos_image = np.array(ipa_image)
                # We use a black image as the negative image prompt for parity with
                # https://github.com/XLabs-AI/x-flux-comfyui/blob/45c834727dd2141aebc505ae4b01f193a8414e38/nodes.py#L592-L593
                # An alternative scheme would be to apply zeros_like() after calling the clip_image_processor.
                neg_image = np.zeros_like(pos_image)
                pos_images.append(pos_image)
                neg_images.append(neg_image)

            with context.models.load(ip_adapter_field.image_encoder_model) as image_encoder_model:
                assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)

                clip_image: torch.Tensor = clip_image_processor(images=pos_images, return_tensors="pt").pixel_values
                clip_image = clip_image.to(device=image_encoder_model.device, dtype=image_encoder_model.dtype)
                pos_clip_image_embeds = image_encoder_model(clip_image).image_embeds

                clip_image = clip_image_processor(images=neg_images, return_tensors="pt").pixel_values
                clip_image = clip_image.to(device=image_encoder_model.device, dtype=image_encoder_model.dtype)
                neg_clip_image_embeds = image_encoder_model(clip_image).image_embeds

            pos_image_prompt_clip_embeds.append(pos_clip_image_embeds)
            neg_image_prompt_clip_embeds.append(neg_clip_image_embeds)

        return pos_image_prompt_clip_embeds, neg_image_prompt_clip_embeds

    def _prep_ip_adapter_extensions(
        self,
        ip_adapter_fields: list[IPAdapterField],
        pos_image_prompt_clip_embeds: list[torch.Tensor],
        neg_image_prompt_clip_embeds: list[torch.Tensor],
        context: InvocationContext,
        exit_stack: ExitStack,
        dtype: torch.dtype,
    ) -> tuple[list[XLabsIPAdapterExtension], list[XLabsIPAdapterExtension]]:
        pos_ip_adapter_extensions: list[XLabsIPAdapterExtension] = []
        neg_ip_adapter_extensions: list[XLabsIPAdapterExtension] = []
        for ip_adapter_field, pos_image_prompt_clip_embed, neg_image_prompt_clip_embed in zip(
            ip_adapter_fields, pos_image_prompt_clip_embeds, neg_image_prompt_clip_embeds, strict=True
        ):
            ip_adapter_model = exit_stack.enter_context(context.models.load(ip_adapter_field.ip_adapter_model))
            assert isinstance(ip_adapter_model, XlabsIpAdapterFlux)
            ip_adapter_model = ip_adapter_model.to(dtype=dtype)
            if ip_adapter_field.mask is not None:
                raise ValueError("IP-Adapter masks are not yet supported in Flux.")
            ip_adapter_extension = XLabsIPAdapterExtension(
                model=ip_adapter_model,
                image_prompt_clip_embed=pos_image_prompt_clip_embed,
                weight=ip_adapter_field.weight,
                begin_step_percent=ip_adapter_field.begin_step_percent,
                end_step_percent=ip_adapter_field.end_step_percent,
            )
            ip_adapter_extension.run_image_proj(dtype=dtype)
            pos_ip_adapter_extensions.append(ip_adapter_extension)

            ip_adapter_extension = XLabsIPAdapterExtension(
                model=ip_adapter_model,
                image_prompt_clip_embed=neg_image_prompt_clip_embed,
                weight=ip_adapter_field.weight,
                begin_step_percent=ip_adapter_field.begin_step_percent,
                end_step_percent=ip_adapter_field.end_step_percent,
            )
            ip_adapter_extension.run_image_proj(dtype=dtype)
            neg_ip_adapter_extensions.append(ip_adapter_extension)

        return pos_ip_adapter_extensions, neg_ip_adapter_extensions

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[LoRAModelRaw, float]]:
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, LoRAModelRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            state.latents = unpack(state.latents.float(), self.height, self.width).squeeze()
            context.util.flux_step_callback(state)

        return step_callback
