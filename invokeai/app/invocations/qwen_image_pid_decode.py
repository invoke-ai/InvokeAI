"""Qwen-Image PiD decode invocation.

Replaces Qwen-Image's AutoencoderKLQwenImage decode with the PiD pixel-diffusion
super-res decoder (``PiD_res2kto4k_sr4x_official_qwenimage_distill_4step``).
Produces a 4x super-resolved image from a Qwen-Image latent in a single 4-step
distill pass.

Qwen-Image is 16-channel at an 8x spatial down-factor (``_PER_BACKBONE[QwenImage]``
in ``backend/pid/decode.py``: ``lq_latent_channels=16``, ``latent_spatial_down_factor=8``),
so no packing is needed. Two Qwen-specific wrinkles, both handled below and both
verified against the existing ``qwen_image_l2i`` node:

1. **5D latent.** The denoiser stores a 5D ``(B, 16, num_frames, H, W)`` latent
   (Qwen's VAE is a video-style autoencoder). PiD is a 2D image decoder, so we
   drop the singleton temporal dim before decoding.
2. **Per-channel normalization.** Unlike FLUX / Z-Image / SDXL (a scalar
   ``scaling_factor`` / ``shift``), the Qwen VAE normalizes each of the 16 latent
   channels by its own ``latents_mean`` / ``latents_std`` vector. Denormalization
   is therefore ``z_raw = z_norm * latents_std + latents_mean`` per channel -
   exactly the transform ``qwen_image_l2i`` applies before ``vae.decode``, so PiD
   (which replaces that decode) sees the same raw latent. We read the vectors from
   the VAE config when a ``vae`` is wired, with the diffusers defaults as fallback.
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

# Per-channel Qwen-Image VAE normalization constants (diffusers AutoencoderKLQwenImage defaults, z_dim=16). Used
# only as a fallback when no `vae` is wired; prefer the wired VAE config's latents_mean / latents_std at runtime.
_QWEN_VAE_LATENTS_MEAN_FALLBACK: list[float] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]  # fmt: skip
_QWEN_VAE_LATENTS_STD_FALLBACK: list[float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]  # fmt: skip


@invocation(
    "qwen_image_pid_decode",
    title="Latents to Image - Qwen-Image + PiD (4x SR)",
    tags=["latents", "image", "pid", "qwen-image", "upscale"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImagePiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode a Qwen-Image latent with the PiD pixel-diffusion decoder.

    Produces a 4x super-resolved image in a single pass. The 5D Qwen latent is
    reduced to 2D and per-channel denormalized (``z * std + mean``) before PiD.
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
        description="PiD Qwen-Image decoder checkpoint.",
        input=Input.Connection,
    )
    vae: VAEField | None = InputField(
        default=None,
        title="VAE",
        description="Qwen-Image VAE, used to read the per-channel latents_mean / latents_std. "
        "If omitted, the diffusers default Qwen-Image constants are used.",
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

        # 1) Reduce the stored 5D (B, C, num_frames, H, W) latent to 2D (B, C, H, W). Qwen's VAE is a video-style
        #    autoencoder; for a single image num_frames == 1 (mirrors qwen_image_l2i's `img[:, :, 0]`).
        if latents.ndim == 5:
            if latents.shape[2] != 1:
                raise ValueError(
                    f"Qwen-Image PiD decode expected a single temporal frame, got shape {tuple(latents.shape)}."
                )
            latents = latents[:, :, 0]
        if latents.ndim != 4 or latents.shape[-3] != 16:
            raise ValueError(f"Qwen-Image PiD decode expected a 16-channel latent, got shape {tuple(latents.shape)}.")

        # 2) Resolve the per-channel latents_mean / latents_std used to denormalise the stored latent.
        latents_mean = list(_QWEN_VAE_LATENTS_MEAN_FALLBACK)
        latents_std = list(_QWEN_VAE_LATENTS_STD_FALLBACK)
        if self.vae is not None:
            vae_info = context.models.load(self.vae.vae)
            with vae_info.model_on_device() as (_, vae):
                config = getattr(vae, "config", None)
                cfg_mean = getattr(config, "latents_mean", None) if config is not None else None
                cfg_std = getattr(config, "latents_std", None) if config is not None else None
                if cfg_mean is not None and cfg_std is not None:
                    latents_mean = [float(x) for x in cfg_mean]
                    latents_std = [float(x) for x in cfg_std]
            del vae_info
            TorchDevice.empty_cache()
        if len(latents_mean) != 16 or len(latents_std) != 16:
            raise ValueError(
                f"Qwen-Image VAE latents_mean/latents_std must have 16 entries, got {len(latents_mean)}/{len(latents_std)}."
            )
        context.logger.info(
            f"Qwen-Image PiD decode: latent shape={tuple(latents.shape)} (expect [B, 16, H/8, W/8]) "
            f"dtype={latents.dtype} per-channel denorm (mean/std from {'VAE config' if self.vae else 'fallback'})"
        )

        # 3) Encode caption with Gemma-2.
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

        # 4) Run PiD decode (the loader already returns a live PidNet).
        pid_info = context.models.load(self.pid_decoder.decoder)
        estimated_working_memory = estimate_pid_decode_working_memory(latents, BaseModelType.QwenImage)
        with pid_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            # Per-channel denormalise: z_raw = z_norm * std + mean (the transform qwen_image_l2i applies before
            # vae.decode). mean/std are (16,) -> (1, 16, 1, 1) to broadcast over the (B, 16, H, W) latent.
            mean_t = torch.tensor(latents_mean, device=device, dtype=dtype).view(1, 16, 1, 1)
            std_t = torch.tensor(latents_std, device=device, dtype=dtype).view(1, 16, 1, 1)
            denorm_latent = latents.to(device=device, dtype=dtype) * std_t + mean_t
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.QwenImage)
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
