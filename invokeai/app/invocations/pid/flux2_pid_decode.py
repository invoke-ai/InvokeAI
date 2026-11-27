"""FLUX.2 Klein PiD decode invocation.

Replaces the regular FLUX.2 VAE decode with the PiD pixel-diffusion super-res
decoder (``PiD_res2k[to4k]_sr4x_official_flux2_distill_4step``). Produces a 4x
super-resolved image from a FLUX.2 latent in a single 4-step distill pass. The
4B and 9B FLUX.2 Klein variants share the same 32-channel VAE, so this one node
covers both.

Latent layout and normalization (the important difference from the FLUX.1 node):

* ``flux2_denoise`` stores an *unpacked* ``(B, 32, H/8, W/8)`` latent that is
  already **BN-denormalized** - the raw latent the FLUX.2 VAE's conv decoder
  consumes.
* PiD's FLUX.2 backbone was trained on what the upstream FLUX.2 tokenizer encodes
  to: the 2x2-patchified ``(B, 128, H/16, W/16)`` latent, **BN-normalized** with
  the VAE's running statistics. ``pack_flux2_latent_for_pid`` restores that, so
  this node needs the FLUX.2 VAE for its BatchNorm statistics.
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
from invokeai.backend.flux2.latent_norm import read_vae_bn_stats
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid._src.networks.pid_net import PidNet
from invokeai.backend.pid.decode import (
    PiDDecodeConfig,
    PiDDecoder,
    assert_pid_decoder_matches_base,
    encode_caption_for_pid,
    estimate_pid_decode_working_memory,
    pack_flux2_latent_for_pid,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_pid_decode",
    title="Latents to Image - FLUX.2 + PiD (4x SR)",
    tags=["latents", "image", "pid", "flux2", "klein", "upscale"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class Flux2PiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode a FLUX.2 Klein latent with the PiD pixel-diffusion decoder.

    Produces a 4x super-resolved image in a single pass. The stored FLUX.2 latent
    is patchified from ``(B, 32, H/8, W/8)`` to ``(B, 128, H/16, W/16)`` and
    BN-normalized with the VAE's statistics, the form PiD's FLUX.2 backbone expects.
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
    vae: VAEField = InputField(
        title="VAE",
        description="FLUX.2 VAE. Its BatchNorm statistics normalize the latent the way PiD was trained on.",
        input=Input.Connection,
    )
    num_inference_steps: int = InputField(
        default=4,
        ge=1,
        le=4,
        description="Number of PiD distill steps. The released checkpoints are distilled for 4; the "
        "student schedule has only 4 transitions, so higher counts cannot add sampling steps.",
    )
    seed: int = InputField(default=0, description="Seed for the PiD decoder's noise.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Fail fast if the connected decoder is for a different backbone (the base-agnostic loader lets
        # the Nodes editor wire any PiD decoder into this FLUX.2-specific node).
        assert_pid_decoder_matches_base(
            context.models.get_config(self.pid_decoder.decoder).base,
            BaseModelType.Flux2,
            node_title="FLUX.2 PiD Decode",
        )

        latents = context.tensors.load(self.latents.latents_name)

        # 1) Patchify and BN-normalize the stored latent into the form PiD's FLUX.2 backbone was trained on.
        with context.models.load(self.vae.vae).model_on_device() as (_, vae):
            bn_stats = read_vae_bn_stats(vae)
        if bn_stats is None:
            raise ValueError(
                "FLUX.2 PiD decode needs the FLUX.2 VAE's BatchNorm statistics to normalize the latent, "
                "but the connected VAE has none. Connect the FLUX.2 VAE."
            )
        packed = pack_flux2_latent_for_pid(latents, *bn_stats)
        context.logger.info(
            f"FLUX.2 PiD decode: stored latent shape={tuple(latents.shape)} -> packed for PiD "
            f"shape={tuple(packed.shape)} (expect [B, 128, H/16, W/16]) dtype={packed.dtype}"
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

            # Encode on the encoder's intended compute device. compute_device honours cpu_only and is
            # stable under partial loading — the first parameter may be offloaded to CPU while later
            # modules load on CUDA, so inferring the device from the first parameter could place caption
            # inputs on the wrong device.
            device = gemma_text_encoder_info.compute_device
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
        # The working-memory estimate scales with the OUTPUT pixel count, so it must see the PACKED latent
        # (spatial H/16), not the unpacked one - otherwise it over-reserves by 4x.
        # Read once: the estimate and the decode must agree, or the cache reserves headroom for a
        # peak that will not happen (or too little for one that will).
        pid_memory_optimization = context.config.get().pid_memory_optimization
        estimated_working_memory = estimate_pid_decode_working_memory(
            packed,
            BaseModelType.Flux2,
            pid_memory_optimization,
            pid_net=pid_info.model,
        )
        with pid_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            latent = packed.to(device=device, dtype=dtype)
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.Flux2)
            x0 = decoder.decode(
                latent=latent,
                caption_embs=caption_embs,
                caption_mask=caption_mask,
                config=PiDDecodeConfig(
                    num_inference_steps=self.num_inference_steps,
                    seed=self.seed,
                    pid_memory_optimization=pid_memory_optimization,
                ),
            )

        TorchDevice.empty_cache()

        img = rearrange(x0[0].clamp(-1, 1), "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
