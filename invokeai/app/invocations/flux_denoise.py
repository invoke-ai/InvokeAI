from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple, Union

import einops
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as tv_transforms
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    FluxFillConditioningField,
    FluxKontextConditioningField,
    FluxReduxConditioningField,
    ImageField,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import ControlLoRAField, LoRAField, TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.kontext_extension import KontextExtension
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
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
from invokeai.backend.flux.text_conditioning import FluxReduxConditioning, FluxTextConditioning
from invokeai.backend.model_manager.taxonomy import ModelFormat, ModelVariantType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_denoise",
    title="FLUX Denoise",
    tags=["image", "flux"],
    category="image",
    version="4.1.0",
)
class FluxDenoiseInvocation(BaseInvocation):
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
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    control_lora: Optional[ControlLoRAField] = InputField(
        description=FieldDescriptions.control_lora_model, input=Input.Connection, title="Control LoRA", default=None
    )
    positive_text_conditioning: FluxConditioningField | list[FluxConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_text_conditioning: FluxConditioningField | list[FluxConditioningField] | None = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    redux_conditioning: FluxReduxConditioningField | list[FluxReduxConditioningField] | None = InputField(
        default=None,
        description="FLUX Redux conditioning tensor.",
        input=Input.Connection,
    )
    fill_conditioning: FluxFillConditioningField | None = InputField(
        default=None,
        description="FLUX Fill conditioning.",
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
    # This node accepts a images for features like FLUX Fill, ControlNet, and Kontext, but needs to operate on them in
    # latent space. We'll run the VAE to encode them in this node instead of requiring the user to run the VAE in
    # upstream nodes.

    ip_adapter: IPAdapterField | list[IPAdapterField] | None = InputField(
        description=FieldDescriptions.ip_adapter, title="IP-Adapter", default=None, input=Input.Connection
    )

    kontext_conditioning: FluxKontextConditioningField | list[FluxKontextConditioningField] | None = InputField(
        default=None,
        description="FLUX Kontext conditioning (reference image).",
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(
        self,
        context: InvocationContext,
    ):
        inference_dtype = torch.bfloat16

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
        b, _c, latent_h, latent_w = noise.shape
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        # Load the conditioning data.
        pos_text_conditionings = self._load_text_conditioning(
            context=context,
            cond_field=self.positive_text_conditioning,
            packed_height=packed_h,
            packed_width=packed_w,
            dtype=inference_dtype,
            device=TorchDevice.choose_torch_device(),
        )
        neg_text_conditionings: list[FluxTextConditioning] | None = None
        if self.negative_text_conditioning is not None:
            neg_text_conditionings = self._load_text_conditioning(
                context=context,
                cond_field=self.negative_text_conditioning,
                packed_height=packed_h,
                packed_width=packed_w,
                dtype=inference_dtype,
                device=TorchDevice.choose_torch_device(),
            )
        redux_conditionings: list[FluxReduxConditioning] = self._load_redux_conditioning(
            context=context,
            redux_cond_field=self.redux_conditioning,
            packed_height=packed_h,
            packed_width=packed_w,
            device=TorchDevice.choose_torch_device(),
            dtype=inference_dtype,
        )
        pos_regional_prompting_extension = RegionalPromptingExtension.from_text_conditioning(
            text_conditioning=pos_text_conditionings,
            redux_conditioning=redux_conditionings,
            img_seq_len=packed_h * packed_w,
        )
        neg_regional_prompting_extension = (
            RegionalPromptingExtension.from_text_conditioning(
                text_conditioning=neg_text_conditionings, redux_conditioning=[], img_seq_len=packed_h * packed_w
            )
            if neg_text_conditionings
            else None
        )

        transformer_config = context.models.get_config(self.transformer.transformer)
        is_schnell = "schnell" in getattr(transformer_config, "config_path", "")

        # Calculate the timestep schedule.
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=packed_h * packed_w,
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

            if self.add_noise:
                # Noise the orig_latents by the appropriate amount for the first timestep.
                t_0 = timesteps[0]
                x = t_0 * noise + (1.0 - t_0) * init_latents
            else:
                x = init_latents
        else:
            # init_latents are not provided, so we are not doing image-to-image (i.e. we are starting from pure noise).
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")

            x = noise

        # If len(timesteps) == 1, then short-circuit. We are just noising the input latents, but not taking any
        # denoising steps.
        if len(timesteps) <= 1:
            return x

        if is_schnell and self.control_lora:
            raise ValueError("Control LoRAs cannot be used with FLUX Schnell")

        # Prepare the extra image conditioning tensor (img_cond) for either FLUX structural control or FLUX Fill.
        img_cond: torch.Tensor | None = None
        is_flux_fill = transformer_config.variant == ModelVariantType.Inpaint  # type: ignore
        if is_flux_fill:
            img_cond = self._prep_flux_fill_img_cond(
                context, device=TorchDevice.choose_torch_device(), dtype=inference_dtype
            )
        else:
            if self.fill_conditioning is not None:
                raise ValueError("fill_conditioning was provided, but the model is not a FLUX Fill model.")

            if self.control_lora is not None:
                img_cond = self._prep_structural_control_img_cond(context)

        inpaint_mask = self._prep_inpaint_mask(context, x)

        img_ids = generate_img_ids(h=latent_h, w=latent_w, batch_size=b, device=x.device, dtype=x.dtype)

        # Pack all latent tensors.
        init_latents = pack(init_latents) if init_latents is not None else None
        inpaint_mask = pack(inpaint_mask) if inpaint_mask is not None else None
        noise = pack(noise)
        x = pack(x)

        # Now that we have 'packed' the latent tensors, verify that we calculated the image_seq_len, packed_h, and
        # packed_w correctly.
        assert packed_h * packed_w == x.shape[1]

        # Prepare inpaint extension.
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents,
                inpaint_mask=inpaint_mask,
                noise=noise,
            )

        # Compute the IP-Adapter image prompt clip embeddings.
        # We do this before loading other models to minimize peak memory.
        # TODO(ryand): We should really do this in a separate invocation to benefit from caching.
        ip_adapter_fields = self._normalize_ip_adapter_fields()
        pos_image_prompt_clip_embeds, neg_image_prompt_clip_embeds = self._prep_ip_adapter_image_prompt_clip_embeds(
            ip_adapter_fields, context, device=x.device
        )

        cfg_scale = self.prep_cfg_scale(
            cfg_scale=self.cfg_scale,
            timesteps=timesteps,
            cfg_scale_start_step=self.cfg_scale_start_step,
            cfg_scale_end_step=self.cfg_scale_end_step,
        )

        kontext_extension = None
        if self.kontext_conditioning:
            if not self.controlnet_vae:
                raise ValueError("A VAE (e.g., controlnet_vae) must be provided to use Kontext conditioning.")

            kontext_extension = KontextExtension(
                context=context,
                kontext_conditioning=self.kontext_conditioning
                if isinstance(self.kontext_conditioning, list)
                else [self.kontext_conditioning],
                vae_field=self.controlnet_vae,
                device=TorchDevice.choose_torch_device(),
                dtype=inference_dtype,
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
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            assert isinstance(transformer, Flux)
            config = transformer_config
            assert config is not None

            # Determine if the model is quantized.
            # If the model is quantized, then we need to apply the LoRA weights as sidecar layers. This results in
            # slower inference than direct patching, but is agnostic to the quantization format.
            if config.format in [ModelFormat.Checkpoint]:
                model_is_quantized = False
            elif config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported model format: {config.format}")

            # Apply LoRA models to the transformer.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Prepare IP-Adapter extensions.
            pos_ip_adapter_extensions, neg_ip_adapter_extensions = self._prep_ip_adapter_extensions(
                pos_image_prompt_clip_embeds=pos_image_prompt_clip_embeds,
                neg_image_prompt_clip_embeds=neg_image_prompt_clip_embeds,
                ip_adapter_fields=ip_adapter_fields,
                context=context,
                exit_stack=exit_stack,
                dtype=inference_dtype,
            )

            # Prepare Kontext conditioning if provided
            img_cond_seq = None
            img_cond_seq_ids = None
            if kontext_extension is not None:
                # Ensure batch sizes match
                kontext_extension.ensure_batch_size(x.shape[0])
                img_cond_seq, img_cond_seq_ids = kontext_extension.kontext_latents, kontext_extension.kontext_ids

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                pos_regional_prompting_extension=pos_regional_prompting_extension,
                neg_regional_prompting_extension=neg_regional_prompting_extension,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                guidance=self.guidance,
                cfg_scale=cfg_scale,
                inpaint_extension=inpaint_extension,
                controlnet_extensions=controlnet_extensions,
                pos_ip_adapter_extensions=pos_ip_adapter_extensions,
                neg_ip_adapter_extensions=neg_ip_adapter_extensions,
                img_cond=img_cond,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
            )

        x = unpack(x.float(), self.height, self.width)
        return x

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        cond_field: FluxConditioningField | list[FluxConditioningField],
        packed_height: int,
        packed_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[FluxTextConditioning]:
        """Load text conditioning data from a FluxConditioningField or a list of FluxConditioningFields."""
        # Normalize to a list of FluxConditioningFields.
        cond_list = [cond_field] if isinstance(cond_field, FluxConditioningField) else cond_field

        text_conditionings: list[FluxTextConditioning] = []
        for cond_field in cond_list:
            # Load the text embeddings.
            cond_data = context.conditioning.load(cond_field.conditioning_name)
            assert len(cond_data.conditionings) == 1
            flux_conditioning = cond_data.conditionings[0]
            assert isinstance(flux_conditioning, FLUXConditioningInfo)
            flux_conditioning = flux_conditioning.to(dtype=dtype, device=device)
            t5_embeddings = flux_conditioning.t5_embeds
            clip_embeddings = flux_conditioning.clip_embeds

            # Load the mask, if provided.
            mask: Optional[torch.Tensor] = None
            if cond_field.mask is not None:
                mask = context.tensors.load(cond_field.mask.tensor_name)
                mask = mask.to(device=device)
                mask = RegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, packed_height, packed_width, dtype, device
                )

            text_conditionings.append(FluxTextConditioning(t5_embeddings, clip_embeddings, mask))

        return text_conditionings

    def _load_redux_conditioning(
        self,
        context: InvocationContext,
        redux_cond_field: FluxReduxConditioningField | list[FluxReduxConditioningField] | None,
        packed_height: int,
        packed_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[FluxReduxConditioning]:
        # Normalize to a list of FluxReduxConditioningFields.
        if redux_cond_field is None:
            return []

        redux_cond_list = (
            [redux_cond_field] if isinstance(redux_cond_field, FluxReduxConditioningField) else redux_cond_field
        )

        redux_conditionings: list[FluxReduxConditioning] = []
        for redux_cond_field in redux_cond_list:
            # Load the Redux conditioning tensor.
            redux_cond_data = context.tensors.load(redux_cond_field.conditioning.tensor_name)
            redux_cond_data.to(device=device, dtype=dtype)

            # Load the mask, if provided.
            mask: Optional[torch.Tensor] = None
            if redux_cond_field.mask is not None:
                mask = context.tensors.load(redux_cond_field.mask.tensor_name)
                mask = mask.to(device=device)
                mask = RegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, packed_height, packed_width, dtype, device
                )

            redux_conditionings.append(FluxReduxConditioning(redux_embeddings=redux_cond_data, mask=mask))

        return redux_conditionings

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

        # Calculate the controlnet conditioning tensors.
        # We do this before loading the ControlNet models because it may require running the VAE, and we are trying to
        # keep peak memory down.
        controlnet_conds: list[torch.Tensor] = []
        for controlnet in controlnets:
            image = context.images.get_pil(controlnet.image.image_name)

            # HACK(ryand): We have to load the ControlNet model to determine whether the VAE needs to be run. We really
            # shouldn't have to load the model here. There's a risk that the model will be dropped from the model cache
            # before we load it into VRAM and thus we'll have to load it again (context:
            # https://github.com/invoke-ai/InvokeAI/issues/7513).
            controlnet_model = context.models.load(controlnet.control_model)
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
        for controlnet, controlnet_cond in zip(controlnets, controlnet_conds, strict=True):
            model = exit_stack.enter_context(context.models.load(controlnet.control_model))

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

    def _prep_structural_control_img_cond(self, context: InvocationContext) -> torch.Tensor | None:
        if self.control_lora is None:
            return None

        if not self.controlnet_vae:
            raise ValueError("controlnet_vae must be set when using a FLUX Control LoRA.")

        # Load the conditioning image and resize it to the target image size.
        cond_img = context.images.get_pil(self.control_lora.img.image_name)
        cond_img = cond_img.convert("RGB")
        cond_img = cond_img.resize((self.width, self.height), Image.Resampling.BICUBIC)
        cond_img = np.array(cond_img)

        # Normalize the conditioning image to the range [-1, 1].
        # This normalization is based on the original implementations here:
        # https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/modules/image_embedders.py#L34
        # https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/modules/image_embedders.py#L60
        img_cond = torch.from_numpy(cond_img).float() / 127.5 - 1.0
        img_cond = einops.rearrange(img_cond, "h w c -> 1 c h w")

        vae_info = context.models.load(self.controlnet_vae.vae)
        img_cond = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=img_cond)

        return pack(img_cond)

    def _prep_flux_fill_img_cond(
        self, context: InvocationContext, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Prepare the FLUX Fill conditioning. This method should be called iff the model is a FLUX Fill model.

        This logic is based on:
        https://github.com/black-forest-labs/flux/blob/716724eb276d94397be99710a0a54d352664e23b/src/flux/sampling.py#L107-L157
        """
        # Validate inputs.
        if self.fill_conditioning is None:
            raise ValueError("A FLUX Fill model is being used without fill_conditioning.")
        # TODO(ryand): We should probable rename controlnet_vae. It's used for more than just ControlNets.
        if self.controlnet_vae is None:
            raise ValueError("A FLUX Fill model is being used without controlnet_vae.")
        if self.control_lora is not None:
            raise ValueError(
                "A FLUX Fill model is being used, but a control_lora was provided. Control LoRAs are not compatible with FLUX Fill models."
            )

        # Log input warnings related to FLUX Fill usage.
        if self.denoise_mask is not None:
            context.logger.warning(
                "Both fill_conditioning and a denoise_mask were provided. You probably meant to use one or the other."
            )
        if self.guidance < 25.0:
            context.logger.warning("A guidance value of ~30.0 is recommended for FLUX Fill models.")

        # Load the conditioning image and resize it to the target image size.
        cond_img = context.images.get_pil(self.fill_conditioning.image.image_name, mode="RGB")
        cond_img = cond_img.resize((self.width, self.height), Image.Resampling.BICUBIC)
        cond_img = np.array(cond_img)
        cond_img = torch.from_numpy(cond_img).float() / 127.5 - 1.0
        cond_img = einops.rearrange(cond_img, "h w c -> 1 c h w")
        cond_img = cond_img.to(device=device, dtype=dtype)

        # Load the mask and resize it to the target image size.
        mask = context.tensors.load(self.fill_conditioning.mask.tensor_name)
        # We expect mask to be a bool tensor with shape [1, H, W].
        assert mask.dtype == torch.bool
        assert mask.dim() == 3
        assert mask.shape[0] == 1
        mask = tv_resize(mask, size=[self.height, self.width], interpolation=tv_transforms.InterpolationMode.NEAREST)
        mask = mask.to(device=device, dtype=dtype)
        mask = einops.rearrange(mask, "1 h w -> 1 1 h w")

        # Prepare image conditioning.
        cond_img = cond_img * (1 - mask)
        vae_info = context.models.load(self.controlnet_vae.vae)
        cond_img = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=cond_img)
        cond_img = pack(cond_img)

        # Prepare mask conditioning.
        mask = mask[:, 0, :, :]
        # Rearrange mask to a 16-channel representation that matches the shape of the VAE-encoded latent space.
        mask = einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=8, pw=8)
        mask = pack(mask)

        # Merge image and mask conditioning.
        img_cond = torch.cat((cond_img, mask), dim=-1)
        return img_cond

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
        device: torch.device,
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
                clip_image = clip_image.to(device=device, dtype=image_encoder_model.dtype)
                pos_clip_image_embeds = image_encoder_model(clip_image).image_embeds

                clip_image = clip_image_processor(images=neg_images, return_tensors="pt").pixel_values
                clip_image = clip_image.to(device=device, dtype=image_encoder_model.dtype)
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

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        loras: list[Union[LoRAField, ControlLoRAField]] = [*self.transformer.loras]
        if self.control_lora:
            # Note: Since FLUX structural control LoRAs modify the shape of some weights, it is important that they are
            # applied last.
            loras.append(self.control_lora)
        for lora in loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            # The denoise function now handles Kontext conditioning correctly,
            # so we don't need to slice the latents here
            latents = state.latents.float()
            state.latents = unpack(latents, self.height, self.width).squeeze()
            context.util.flux_step_callback(state)

        return step_callback
