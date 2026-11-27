"""Qwen-Image PiD decode invocation.

Replaces Qwen-Image's AutoencoderKLQwenImage decode with the PiD pixel-diffusion
super-res decoder (``PiD_res2kto4k_sr4x_official_qwenimage_distill_4step``).
Produces a 4x super-resolved image from a Qwen-Image latent in a single 4-step
distill pass.

Qwen-Image is 16-channel at an 8x spatial down-factor (``_PER_BACKBONE[QwenImage]``
in ``backend/pid/decode.py``: ``lq_latent_channels=16``, ``latent_spatial_down_factor=8``),
so no packing is needed. The denoiser stores a 5D ``(B, 16, num_frames, H, W)``
latent (Qwen's VAE is a video-style autoencoder); PiD is a 2D image decoder, so we
drop the singleton temporal dim before decoding.

The latent stays in the Qwen VAE's per-channel normalized space
(``(raw - latents_mean) / latents_std``, what the upstream tokenizer encodes to):
PiD was trained on that, not on the raw latent ``qwen_image_l2i`` hands the VAE.
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
    assert_pid_decoder_matches_base,
    encode_caption_for_pid,
    estimate_pid_decode_working_memory,
)
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "qwen_image_pid_decode",
    title="Latents to Image - Qwen-Image + PiD (4x SR)",
    tags=["latents", "image", "pid", "qwen-image", "upscale"],
    category="latents",
    version="1.0.1",
    classification=Classification.Prototype,
)
class QwenImagePiDDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode a Qwen-Image latent with the PiD pixel-diffusion decoder.

    Produces a 4x super-resolved image in a single pass. The 5D Qwen latent is
    reduced to 2D and otherwise handed to PiD in its normalized form.
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
        description="Not used: PiD decodes the normalized latent directly. Kept so existing workflows "
        "that connect a VAE stay valid.",
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
        # the Nodes editor wire any PiD decoder into this Qwen-Image-specific node).
        assert_pid_decoder_matches_base(
            context.models.get_config(self.pid_decoder.decoder).base,
            BaseModelType.QwenImage,
            node_title="Qwen-Image PiD Decode",
        )

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

        context.logger.info(
            f"Qwen-Image PiD decode: latent shape={tuple(latents.shape)} (expect [B, 16, H/8, W/8]) "
            f"dtype={latents.dtype}"
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
        # Read once: the estimate and the decode must agree, or the cache reserves headroom for a
        # peak that will not happen (or too little for one that will).
        pid_memory_optimization = context.config.get().pid_memory_optimization
        estimated_working_memory = estimate_pid_decode_working_memory(
            latents,
            BaseModelType.QwenImage,
            pid_memory_optimization,
            pid_net=pid_info.model,
        )
        with pid_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, pid_net):
            if not isinstance(pid_net, PidNet):
                raise TypeError(f"Expected PidNet for PiD decoder, got {type(pid_net).__name__}.")
            device = TorchDevice.choose_torch_device()
            dtype = next(iter(pid_net.parameters())).dtype

            latent = latents.to(device=device, dtype=dtype)
            caption_embs = caption_embs.to(device=device, dtype=dtype)

            context.util.signal_progress("Running PiD decoder")
            decoder = PiDDecoder(pid_net, backbone=BaseModelType.QwenImage)
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
