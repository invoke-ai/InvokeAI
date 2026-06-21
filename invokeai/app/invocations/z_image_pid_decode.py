"""Z-Image PiD decode invocation.

Z-Image shares FLUX.1's 16-channel VAE, so the FLUX-trained PiD decoder
(``PiD_res2k_sr4x_official_flux_distill_4step``) is the correct choice for
Z-Image latents. This node replaces the regular Z-Image VAE decode with a
PiD super-resolution decode (4x scale, ~256×256 latent → 2048×2048 image
by default).
"""

from contextlib import ExitStack

import torch
from einops import rearrange
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIComponent,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import Gemma2EncoderField, PiDDecoderField, VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid._src.networks.pid_net import PidNet
from invokeai.backend.pid.decode import (
    PiDDecodeConfig,
    PiDDecoder,
    encode_caption_for_pid,
)
from invokeai.backend.util.devices import TorchDevice

# Fallback Z-Image VAE constants. PiD's pipeline_registry.py explicitly notes
# the exact values depend on the pretrained checkpoint, so prefer reading them
# from the VAE config at runtime (see `vae` input below) and use these only as
# a last resort.
_ZIMAGE_VAE_SCALING_FACTOR_FALLBACK: float = 0.3611
_ZIMAGE_VAE_SHIFT_FACTOR_FALLBACK: float = 0.1159


@invocation(
    "z_image_pid_decode",
    title="Latents to Image - Z-Image + PiD (4x SR)",
    tags=["latents", "image", "pid", "z-image", "upscale"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ZImagePiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode a Z-Image latent with the PiD pixel-diffusion decoder.

    Produces a 4x super-resolved image in a single pass (Z-Image decoder is
    trained on FLUX.1 latents; ``sr_scale=4`` with the FLUX VAE's 8x spatial
    down-factor gives a 32x linear scale from latent to pixel).
    """

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    prompt: str = InputField(
        description="Text prompt the latent was generated from. PiD conditions on it.",
        ui_component=UIComponent.Textarea,
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
    vae: VAEField | None = InputField(
        default=None,
        title="VAE",
        description="Z-Image VAE used to read scaling_factor / shift_factor. "
        "If omitted, the FLUX.1 fallback constants (0.3611 / 0.1159) are used.",
        input=Input.Connection,
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
        latents = context.tensors.load(self.latents.latents_name)

        # 1) Resolve the VAE scaling/shift used to denormalise the stored
        # Z-Image latent. PiD's pipeline_registry says these are
        # checkpoint-specific for Z-Image, so prefer the VAE config when
        # available and fall back to the FLUX values otherwise.
        scaling_factor = _ZIMAGE_VAE_SCALING_FACTOR_FALLBACK
        shift_factor = _ZIMAGE_VAE_SHIFT_FACTOR_FALLBACK
        if self.vae is not None:
            vae_info = context.models.load(self.vae.vae)
            with vae_info.model_on_device() as (_, vae):
                config = getattr(vae, "config", None)
                if config is not None and hasattr(config, "scaling_factor"):
                    scaling_factor = float(config.scaling_factor)
                    shift_factor = float(getattr(config, "shift_factor", None) or 0.0)
                else:
                    # FluxAutoEncoder stores the constants directly on the module.
                    scaling_factor = float(getattr(vae, "scale_factor", scaling_factor))
                    shift_factor = float(getattr(vae, "shift_factor", shift_factor))
            del vae_info
            TorchDevice.empty_cache()
        context.logger.info(
            f"Z-Image PiD decode: latent shape={tuple(latents.shape)} dtype={latents.dtype} "
            f"stats[min={latents.min().item():.3f} max={latents.max().item():.3f} "
            f"mean={latents.mean().item():.3f}] using scale={scaling_factor:.4f} shift={shift_factor:.4f}"
        )

        # 2) Encode caption with Gemma-2.
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
            # Move off-device so Gemma's slot in the cache can be reclaimed.
            caption_embs = caption_embs.detach().to("cpu")

            caption_mask = caption_mask.detach().to("cpu")
        # Drop Gemma references so the cache can evict it before we load PiD.
        del gemma_encoder, gemma_tokenizer
        TorchDevice.empty_cache()

        # 2) Run PiD decode (the loader already returns a live PidNet).
        pid_info = context.models.load(self.pid_decoder.decoder)
        with pid_info.model_on_device() as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            # Z-Image latents come out of the diffusers pipeline normalised
            # by the VAE constants. PiD expects the raw latent.
            denorm_latent = latents.to(device=device, dtype=dtype) / scaling_factor + shift_factor
            context.logger.info(
                f"denorm_latent stats[min={denorm_latent.min().item():.3f} "
                f"max={denorm_latent.max().item():.3f} mean={denorm_latent.mean().item():.3f} "
                f"std={denorm_latent.float().std().item():.3f}]; "
                f"caption_embs shape={tuple(caption_embs.shape)} "
                f"stats[min={caption_embs.min().item():.3f} max={caption_embs.max().item():.3f} "
                f"mean={caption_embs.mean().item():.3f} std={caption_embs.float().std().item():.3f}]"
            )
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.Flux)
            x0 = decoder.decode(
                latent=denorm_latent,
                caption_embs=caption_embs,
                caption_mask=caption_mask,
                config=PiDDecodeConfig(num_inference_steps=self.num_inference_steps, seed=self.seed),
            )
            context.logger.info(
                f"PiD output stats: shape={tuple(x0.shape)} dtype={x0.dtype} "
                f"raw[min={x0.min().item():.3f} max={x0.max().item():.3f} "
                f"mean={x0.mean().item():.3f} std={x0.float().std().item():.3f}] "
                f"nan_count={int(torch.isnan(x0).sum().item())} "
                f"inf_count={int(torch.isinf(x0).sum().item())}"
            )

        TorchDevice.empty_cache()

        # x0 is [B, 3, H, W] in [-1, 1]; convert the first item to a PIL image.
        img = rearrange(x0[0].clamp(-1, 1), "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
