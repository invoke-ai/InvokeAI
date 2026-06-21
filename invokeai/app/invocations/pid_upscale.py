"""PiD super-resolution upscale invocation.

Stand-alone 4x super-resolution path that does **not** require a Generator
latent. Pipeline::

    image
      -> FLUX VAE encode (denormalised back to raw)
      -> Gemma-2 caption encode
      -> PiD decoder (4x SR)
      -> image (4x linear)

This is the PiD analogue of ESRGAN / SUPIR: a one-shot, end-to-end pixel
upscaler. The FLUX VAE is also valid for Z-Image inputs (they share the
same 16-channel encoder). SD3 / FLUX.2 upscale paths would each need their
own invocation with the matching VAE encode and latent denormalisation;
they are deferred until we have the matching PiD checkpoints to validate
against.
"""

from contextlib import ExitStack

import einops
import torch
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    ImageField,
    Input,
    InputField,
    UIComponent,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.invocations.model import Gemma2EncoderField, PiDDecoderField, VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.util import get_flux_ae_params
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid._src.networks.pid_net import PidNet
from invokeai.backend.pid.decode import (
    PiDDecodeConfig,
    PiDDecoder,
    encode_caption_for_pid,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "pid_upscale",
    title="PiD Upscale (4x) - FLUX VAE",
    tags=["upscale", "image", "pid", "super-resolution", "flux"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class PiDUpscaleInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Upscale any image 4x via FLUX VAE encode + PiD pixel-diffusion decode.

    Works for source images that the FLUX VAE can encode (i.e. natural
    photos / generated images at any size that lands on the VAE's 8-pixel
    grid). The caption is used to condition the PiD decoder; leaving it
    empty produces an unconditional decode and is the cheapest option, but
    the model was distilled with rich captions and benefits from one.
    """

    image: ImageField = InputField(description="Image to upscale.")
    vae: VAEField = InputField(
        description="FLUX-compatible VAE (FLUX.1, Z-Image, anything sharing the 16-channel encoder).",
        input=Input.Connection,
    )
    gemma2_encoder: Gemma2EncoderField = InputField(
        title="Gemma-2 Encoder",
        description="Gemma-2 caption encoder. Required by PiD.",
        input=Input.Connection,
    )
    pid_decoder: PiDDecoderField = InputField(
        title="PiD Decoder",
        description="PiD FLUX decoder checkpoint.",
        input=Input.Connection,
    )
    prompt: str = InputField(
        default="",
        description="Optional caption describing the image. Empty -> empty-caption decode.",
        ui_component=UIComponent.Textarea,
    )
    num_inference_steps: int = InputField(
        default=4,
        ge=1,
        le=8,
        description="Number of PiD distill steps. The released checkpoints are trained for 4.",
    )
    seed: int = InputField(default=0, description="Seed for the PiD decoder's noise.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # 1) Encode the source image into a FLUX raw latent.
        pil_image = context.images.get_pil(self.image.image_name).convert("RGB")
        image_tensor = image_resized_to_grid_as_tensor(pil_image)
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running VAE encode")
        normalised_latent = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=image_tensor)
        # FluxAutoEncoder.encode emits `scale * (raw - shift)`. PiD expects raw,
        # so undo it. Holds for the Z-Image case as well (same VAE constants).
        ae = get_flux_ae_params()
        raw_latent = normalised_latent / ae.scale_factor + ae.shift_factor
        raw_latent = raw_latent.to("cpu")  # park while we swap to Gemma
        del normalised_latent
        TorchDevice.empty_cache()

        # 2) Encode the caption with Gemma-2.
        gemma_text_encoder_info = context.models.load(self.gemma2_encoder.text_encoder)
        gemma_tokenizer_info = context.models.load(self.gemma2_encoder.tokenizer)
        with ExitStack() as stack:
            (_, gemma_encoder) = stack.enter_context(gemma_text_encoder_info.model_on_device())
            (_, gemma_tokenizer) = stack.enter_context(gemma_tokenizer_info.model_on_device())
            if not isinstance(gemma_encoder, PreTrainedModel):
                raise TypeError(f"Expected PreTrainedModel for Gemma encoder, got {type(gemma_encoder).__name__}.")
            if not isinstance(gemma_tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"Expected PreTrainedTokenizerBase for Gemma tokenizer, got {type(gemma_tokenizer).__name__}."
                )
            device = TorchDevice.choose_torch_device()
            encode_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            context.util.signal_progress("Encoding caption with Gemma-2")
            caption_embs, caption_mask = encode_caption_for_pid(
                [self.prompt],
                tokenizer=gemma_tokenizer,
                encoder=gemma_encoder,
                device=device,
                dtype=encode_dtype,
            )
            caption_embs = caption_embs.detach().to("cpu")

            caption_mask = caption_mask.detach().to("cpu")
        del gemma_encoder, gemma_tokenizer
        TorchDevice.empty_cache()

        # 3) Run PiD decode (the loader already returns a live PidNet).
        pid_info = context.models.load(self.pid_decoder.decoder)
        with pid_info.model_on_device() as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            latent_on_device = raw_latent.to(device=device, dtype=dtype)
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.Flux)
            x0 = decoder.decode(
                latent=latent_on_device,
                caption_embs=caption_embs,
                caption_mask=caption_mask,
                config=PiDDecodeConfig(num_inference_steps=self.num_inference_steps, seed=self.seed),
            )

        TorchDevice.empty_cache()

        img = einops.rearrange(x0[0].clamp(-1, 1), "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
