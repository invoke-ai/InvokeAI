"""FLUX.2 Klein PiD decode invocation.

Replaces the regular FLUX.2 VAE decode with the PiD pixel-diffusion super-res
decoder (``PiD_res2k[to4k]_sr4x_official_flux2_distill_4step``). Produces a 4x
super-resolved image from a FLUX.2 latent in a single 4-step distill pass. The
4B and 9B FLUX.2 Klein variants share the same 32-channel VAE, so this one node
covers both.

Latent layout (the important difference from the FLUX.1 node):

* ``flux2_denoise`` stores an *unpacked* ``(B, 32, H/8, W/8)`` latent that is
  already **BN-denormalized** (``x * bn_std + bn_mean`` is applied before the
  unpack, see ``flux2_denoise.py``). That is exactly the raw latent the FLUX.2
  VAE's conv decoder consumes.
* PiD's FLUX.2 backbone expects the **packed** ``(B, 128, H/16, W/16)``
  representation (``lq_latent_channels=128``, ``latent_spatial_down_factor=16``
  in ``backend/pid/decode.py``). We therefore patchify the stored latent
  (2x2 spatial patches folded into channels: 32*4 = 128) *before* handing it to
  PiD - mirroring ``pack_flux2`` but keeping a spatial ``(B, C, h, w)`` layout
  instead of the transformer's ``(B, seq, C)`` sequence layout.

Denormalization: unlike FLUX.1 (single ``scale``/``shift``) and Z-Image
(checkpoint-specific ``scaling_factor``/``shift_factor``), the FLUX.2 VAE
(``AutoencoderKLFlux2``) exposes **no** scalar ``scaling_factor``/``shift_factor``
at all - its only normalization is the per-channel BatchNorm applied/inverted
*outside* the VAE in ``flux2_denoise``. So the packed latent is already in PiD's
expected raw space and no further scaling is needed (identity fallbacks below).
We still accept an optional ``vae`` input and read the constants at runtime (like
the Z-Image node) so any future FLUX.2 VAE variant that does expose scalar
constants is honored automatically.
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

# FLUX.2 uses per-channel BatchNorm (affine=False) for latent normalization, and
# that BN is already inverted in flux2_denoise before the latent is stored. The
# FLUX.2 VAE (AutoencoderKLFlux2) has no scalar scaling_factor/shift_factor, so
# the identity transform below is the correct default: the stored (packed) latent
# is already the raw representation PiD was trained on.
_FLUX2_VAE_SCALING_FACTOR_FALLBACK: float = 1.0
_FLUX2_VAE_SHIFT_FACTOR_FALLBACK: float = 0.0


@invocation(
    "flux2_pid_decode",
    title="Latents to Image - FLUX.2 + PiD (4x SR)",
    tags=["latents", "image", "pid", "flux2", "klein", "upscale"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2PiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode a FLUX.2 Klein latent with the PiD pixel-diffusion decoder.

    Produces a 4x super-resolved image in a single pass. The stored FLUX.2 latent
    is patchified from ``(B, 32, H/8, W/8)`` to the ``(B, 128, H/16, W/16)`` layout
    PiD's FLUX.2 backbone expects, then decoded directly (it is already in raw,
    BN-denormalized space; see the module docstring).
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
        description="PiD FLUX.2 decoder checkpoint.",
        input=Input.Connection,
    )
    vae: VAEField | None = InputField(
        default=None,
        title="VAE",
        description="FLUX.2 VAE, used only to read a scalar scaling_factor / shift_factor if one exists. "
        "FLUX.2 normalises latents with BatchNorm (already inverted in flux2_denoise), so this is "
        "normally an identity transform and the input can be left unconnected.",
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

        # 1) Patchify the stored FLUX.2 latent into PiD's expected layout.
        #    flux2_denoise stores an unpacked (B, 32, H/8, W/8) latent; PiD's
        #    FLUX.2 backbone wants the packed (B, 128, H/16, W/16) form (32*4=128
        #    channels, spatial halved). This mirrors pack_flux2's 2x2 patchify but
        #    keeps a spatial (B, C, h, w) layout rather than a (B, seq, C) sequence.
        if latents.shape[-3] != 32:
            raise ValueError(
                f"FLUX.2 PiD decode expected a 32-channel latent from flux2_denoise, got shape "
                f"{tuple(latents.shape)}. The upstream node must output the unpacked FLUX.2 latent."
            )
        packed = rearrange(latents, "b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2)
        context.logger.info(
            f"FLUX.2 PiD decode: stored latent shape={tuple(latents.shape)} -> packed for PiD "
            f"shape={tuple(packed.shape)} (expect [B, 128, H/16, W/16]) dtype={packed.dtype}"
        )

        # 2) Resolve the scalar scaling/shift (identity for current FLUX.2 VAEs).
        scaling_factor = _FLUX2_VAE_SCALING_FACTOR_FALLBACK
        shift_factor = _FLUX2_VAE_SHIFT_FACTOR_FALLBACK
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
        # The working-memory estimate scales with the OUTPUT pixel count, so it must see the PACKED latent
        # (spatial H/16), not the unpacked one - otherwise it over-reserves by 4x.
        estimated_working_memory = estimate_pid_decode_working_memory(packed, BaseModelType.Flux2)
        with pid_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            # The packed latent is already BN-denormalized (raw VAE-input space); the scalar transform below is
            # identity for current FLUX.2 VAEs and only bites if a VAE ever exposes real scalar constants.
            denorm_latent = packed.to(device=device, dtype=dtype) / scaling_factor + shift_factor
            context.logger.info(
                f"FLUX.2 PiD denorm_latent stats[min={denorm_latent.min().item():.3f} "
                f"max={denorm_latent.max().item():.3f} mean={denorm_latent.mean().item():.3f}] "
                f"using scale={scaling_factor:.4f} shift={shift_factor:.4f}"
            )
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.Flux2)
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
