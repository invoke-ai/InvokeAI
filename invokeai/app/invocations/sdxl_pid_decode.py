"""SDXL PiD decode invocation.

Replaces SDXL's AutoencoderKL decode with the PiD pixel-diffusion super-res
decoder (``PiD_res2kto4k_sr4x_official_sdxl_distill_4step``). Produces a 4x
super-resolved image from an SDXL latent in a single 4-step distill pass.

SDXL latents are 4-channel at an 8x spatial down-factor (``_PER_BACKBONE[SDXL]``
in ``backend/pid/decode.py``: ``lq_latent_channels=4``, ``latent_spatial_down_factor=8``),
so - unlike FLUX.2 - no patchify/pack is needed; the stored latent goes straight
to PiD after denormalization.

Denormalization: SDXL's VAE (``AutoencoderKL``) exposes a scalar
``scaling_factor`` (0.13025) and no shift, so the stored latent is denormalized
as ``z / scaling_factor + shift`` (matching the FLUX / Z-Image nodes). We read
the constants from the VAE config at runtime when a ``vae`` is wired, falling
back to the documented SDXL constants otherwise.
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
    estimate_pid_decode_working_memory,
)
from invokeai.backend.util.devices import TorchDevice

# SDXL VAE constants (diffusers `stabilityai/sdxl-vae` config: scaling_factor=0.13025, no shift). Prefer reading
# scaling_factor / shift_factor from the wired VAE config at runtime; use these only as a fallback.
_SDXL_VAE_SCALING_FACTOR_FALLBACK: float = 0.13025
_SDXL_VAE_SHIFT_FACTOR_FALLBACK: float = 0.0


@invocation(
    "sdxl_pid_decode",
    title="Latents to Image - SDXL + PiD (4x SR)",
    tags=["latents", "image", "pid", "sdxl", "upscale"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class SDXLPiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode an SDXL latent with the PiD pixel-diffusion decoder.

    Produces a 4x super-resolved image in a single pass. The SDXL latent is
    4-channel at an 8x down-factor, so it is denormalized (``z / scaling_factor``)
    and handed straight to PiD - no packing needed.
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
        description="PiD SDXL decoder checkpoint.",
        input=Input.Connection,
    )
    vae: VAEField | None = InputField(
        default=None,
        title="VAE",
        description="SDXL VAE, used to read scaling_factor / shift_factor. "
        "If omitted, the SDXL fallback constants (0.13025 / 0.0) are used.",
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

        # 1) Resolve the VAE scaling/shift used to denormalise the stored SDXL latent. Prefer the VAE config; fall
        # back to the documented SDXL constants (0.13025 / 0.0).
        scaling_factor = _SDXL_VAE_SCALING_FACTOR_FALLBACK
        shift_factor = _SDXL_VAE_SHIFT_FACTOR_FALLBACK
        if self.vae is not None:
            vae_info = context.models.load(self.vae.vae)
            with vae_info.model_on_device() as (_, vae):
                config = getattr(vae, "config", None)
                if config is not None and hasattr(config, "scaling_factor"):
                    scaling_factor = float(config.scaling_factor)
                    shift_factor = float(getattr(config, "shift_factor", None) or 0.0)
                else:
                    scaling_factor = float(getattr(vae, "scale_factor", scaling_factor))
                    shift_factor = float(getattr(vae, "shift_factor", shift_factor))
            del vae_info
            TorchDevice.empty_cache()
        context.logger.info(
            f"SDXL PiD decode: latent shape={tuple(latents.shape)} (expect [B, 4, H/8, W/8]) dtype={latents.dtype} "
            f"using scale={scaling_factor:.5f} shift={shift_factor:.5f}"
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
            caption_embs = caption_embs.detach().to("cpu")
            caption_mask = caption_mask.detach().to("cpu")
        del gemma_encoder, gemma_tokenizer
        # Gemma is only needed for the one-shot caption encode above. Offload it from VRAM (keeping it in the RAM
        # cache) so its ~5GB is freed before the PiD decoder loads. The cache offloads anything else it needs to
        # fit the decode on its own, so we deliberately do NOT evict every other model here.
        context.models.offload_from_vram(self.gemma2_encoder.text_encoder)
        TorchDevice.empty_cache()

        # 3) Run PiD decode (the loader already returns a live PidNet).
        pid_info = context.models.load(self.pid_decoder.decoder)
        estimated_working_memory = estimate_pid_decode_working_memory(latents, BaseModelType.StableDiffusionXL)
        with pid_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            # SDXL latents come out of the LDM in the VAE-normalized space; denormalise so PiD sees the raw latent.
            denorm_latent = latents.to(device=device, dtype=dtype) / scaling_factor + shift_factor
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.StableDiffusionXL)
            x0 = decoder.decode(
                latent=denorm_latent,
                caption_embs=caption_embs,
                caption_mask=caption_mask,
                config=PiDDecodeConfig(num_inference_steps=self.num_inference_steps, seed=self.seed),
            )

        TorchDevice.empty_cache()

        img = rearrange(x0[0].clamp(-1, 1), "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
