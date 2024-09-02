import torch
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.flux.sampling import denoise, get_noise, get_schedule, prepare_latent_img_patches, unpack
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_text_to_image",
    title="FLUX Text to Image",
    tags=["image", "flux"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class FluxTextToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Text-to-image generation using a FLUX model."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    positive_text_conditioning: FluxConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(
        default=4, description="Number of diffusion steps. Recommend values are schnell: 4, dev: 50."
    )
    guidance: float = InputField(
        default=4.0,
        description="The guidance strength. Higher values adhere more strictly to the prompt, and will produce less diverse images. FLUX dev only, ignored for schnell.",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = self._run_diffusion(context)
        image = self._run_vae_decoding(context, latents)
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)

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

        transformer_info = context.models.load(self.transformer.transformer)

        # Prepare input noise.
        x = get_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=TorchDevice.choose_torch_device(),
            dtype=inference_dtype,
            seed=self.seed,
        )

        x, img_ids = prepare_latent_img_patches(x)

        is_schnell = "schnell" in transformer_info.config.config_path

        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=x.shape[1],
            shift=not is_schnell,
        )

        bs, t5_seq_len, _ = t5_embeddings.shape
        txt_ids = torch.zeros(bs, t5_seq_len, 3, dtype=inference_dtype, device=TorchDevice.choose_torch_device())

        with transformer_info as transformer:
            assert isinstance(transformer, Flux)

            def step_callback() -> None:
                if context.util.is_canceled():
                    raise CanceledException

                # TODO: Make this look like the image before re-enabling
                # latent_image = unpack(img.float(), self.height, self.width)
                # latent_image = latent_image.squeeze()  # Remove unnecessary dimensions
                # flattened_tensor = latent_image.reshape(-1)  # Flatten to shape [48*128*128]

                # # Create a new tensor of the required shape [255, 255, 3]
                # latent_image = flattened_tensor[: 255 * 255 * 3].reshape(255, 255, 3)  # Reshape to RGB format

                # # Convert to a NumPy array and then to a PIL Image
                # image = Image.fromarray(latent_image.cpu().numpy().astype(np.uint8))

                # (width, height) = image.size
                # width *= 8
                # height *= 8

                # dataURL = image_to_dataURL(image, image_format="JPEG")

                # # TODO: move this whole function to invocation context to properly reference these variables
                # context._services.events.emit_invocation_denoise_progress(
                #     context._data.queue_item,
                #     context._data.invocation,
                #     state,
                #     ProgressImage(dataURL=dataURL, width=width, height=height),
                # )

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=t5_embeddings,
                txt_ids=txt_ids,
                vec=clip_embeddings,
                timesteps=timesteps,
                step_callback=step_callback,
                guidance=self.guidance,
            )

        x = unpack(x.float(), self.height, self.width)

        return x

    def _run_vae_decoding(
        self,
        context: InvocationContext,
        latents: torch.Tensor,
    ) -> Image.Image:
        vae_info = context.models.load(self.vae.vae)
        with vae_info as vae:
            assert isinstance(vae, AutoEncoder)
            latents = latents.to(dtype=TorchDevice.choose_torch_dtype())
            img = vae.decode(latents)

        img = img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        return img_pil
