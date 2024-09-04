from typing import Callable, Optional

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
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.inpaint_extension import InpaintExtension
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule,
    generate_img_ids,
    get_noise,
    get_schedule,
    pack,
    unpack,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_denoise",
    title="FLUX Denoise",
    tags=["image", "flux"],
    category="image",
    version="1.0.0",
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
        timesteps = clip_timestep_schedule(timesteps, self.denoising_start, self.denoising_end)

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

        b, _c, h, w = x.shape
        img_ids = generate_img_ids(h=h, w=w, batch_size=b, device=x.device, dtype=x.dtype)

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

        with transformer_info as transformer:
            assert isinstance(transformer, Flux)

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
            torch.Tensor | None: Inpaint mask.
        """
        if self.denoise_mask is None:
            return None

        mask = context.tensors.load(self.denoise_mask.mask_name)

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

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            state.latents = unpack(state.latents.float(), self.height, self.width).squeeze()
            context.util.flux_step_callback(state)

        return step_callback
