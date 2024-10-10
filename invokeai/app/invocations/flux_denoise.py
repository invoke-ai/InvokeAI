from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.extensions.inpaint_extension import InpaintExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
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
    version="3.1.0",
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
        description=FieldDescriptions.vae,
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

        # Load the conditioning data.
        cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(cond_data.conditionings) == 1
        flux_conditioning = cond_data.conditionings[0]
        assert isinstance(flux_conditioning, FLUXConditioningInfo)
        flux_conditioning = flux_conditioning.to(dtype=inference_dtype)
        t5_embeddings = flux_conditioning.t5_embeds
        clip_embeddings = flux_conditioning.clip_embeds

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

        bs, t5_seq_len, _ = t5_embeddings.shape
        txt_ids = torch.zeros(bs, t5_seq_len, 3, dtype=inference_dtype, device=TorchDevice.choose_torch_device())

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

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=t5_embeddings,
                txt_ids=txt_ids,
                vec=clip_embeddings,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                guidance=self.guidance,
                inpaint_extension=inpaint_extension,
                controlnet_extensions=controlnet_extensions,
            )

        x = unpack(x.float(), self.height, self.width)
        return x

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
