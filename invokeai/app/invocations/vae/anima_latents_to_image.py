"""Anima latents-to-image invocation.

Decodes Anima latents with the Wan 2.1 VAE, whichever of the anima, qwen-image or 16-channel wan
registrations it was installed under and in either layout: the original-layout file loads as
AutoencoderKLWan, the diffusers-layout Qwen-Image export as AutoencoderKLQwenImage (same weights, same
latent statistics). Any other decoder is refused, and so is a Wan VAE of another geometry (Wan 2.2's
48-channel VAE).

Latents from the denoiser are in normalized space (zero-centered). Before
VAE decode, they must be denormalized using the Wan 2.1 per-channel
mean/std: latents = latents * std + mean (matching diffusers WanPipeline).

The VAE expects 5D latents [B, C, T, H, W] — for single images, T=1.
"""

import torch
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.qwen_image_vae import (
    QwenImageCompatibleVAE,
    as_qwen_image_vae,
    patch_qwen_image_vae_tiling,
)
from invokeai.backend.util.vae_decode_diagnostics import (
    allocator_state_summary,
    force_real_empty_cache,
    nonfinite_fraction,
    scan_module_for_nonfinite_weights,
)
from invokeai.backend.util.vae_working_memory import (
    estimate_vae_working_memory_anima,
    should_pretile_vae_decode,
)

# Tile geometry for tiled Wan VAE decode. 512px tiles with a 384px stride (128px blended
# overlap) cap peak decode working memory at ~1.7GB regardless of image size, while images
# <=512px still decode in a single pass. `patch_qwen_image_vae_tiling` derives that same stride
# from the tile size (the VAE's stock 3/4 ratio); the stride is named here because the
# working-memory constant was calibrated against this exact geometry.
ANIMA_VAE_TILE_SIZE = 512
ANIMA_VAE_TILE_STRIDE = 384

# Lower than the other decodes' `VAE_PRETILE_VRAM_FRACTION`, on purpose. A full 1024x1024 Wan VAE decode reserves ~6GB
# of working memory; on small-VRAM GPUs that evicts the (~4GB) Anima transformer from the model cache and thrashes the
# allocator near the VRAM ceiling (decode times of 7s+ observed on 8GB, vs ~1s tiled with the transformer left
# resident). Above this share of the device a tiled decode wins; below it a single pass is faster (~0.65s vs ~1.05s at
# 1024x1024) and exact.
ANIMA_PRETILE_VRAM_FRACTION = 0.7


@invocation(
    "anima_l2i",
    title="Latents to Image - Anima",
    tags=["latents", "image", "vae", "l2i", "anima"],
    category="latents",
    version="1.0.4",
    classification=Classification.Prototype,
)
class AnimaLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using the Anima VAE (the Wan 2.1 VAE, in either layout)."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        try:
            # Also raises ValueError for a Wan VAE of another geometry (Wan 2.2's 48-channel VAE).
            as_qwen_image_vae(vae_info.model)
        except TypeError as e:
            # A FLUX VAE reaches here with the right shape -- 16 channels at 8x, same as Anima --
            # and decodes without raising, which is why it was offered for a while. It is a
            # different basis: measured against the correct decode of the same latent, 8.67 dB
            # PSNR, a magenta moire in place of the subject, and the run still reports success.
            # Refuse until someone implements the change of basis; a wrong image is worse than
            # none, because nothing downstream can tell it apart from an intended one.
            raise TypeError(
                "Anima decodes in the 16-channel Wan 2.1 latent space, and "
                f"{type(vae_info.model).__name__} is not that decoder. Choose a VAE registered under "
                "'anima', 'qwen-image', or a 16-channel 'wan' VAE -- all three are the same "
                "194-tensor checkpoint. A FLUX VAE has the same channel count but a different "
                "basis, and converting between them is not implemented."
            ) from e

        full_decode_working_memory = estimate_vae_working_memory_anima(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
            tile_size=None,
        )
        # Not gated on `auto_tiled_decode`: this rule predates that setting and is a speed optimization, not a way
        # around an out-of-memory error. Turning the setting off here would hand an 8GB card the 7s decode above
        # instead of restoring anything.
        use_tiling = should_pretile_vae_decode(
            vae_info.compute_device, full_decode_working_memory, ANIMA_PRETILE_VRAM_FRACTION
        )
        estimated_working_memory = estimate_vae_working_memory_anima(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
            tile_size=ANIMA_VAE_TILE_SIZE if use_tiling else None,
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running Anima VAE decode")
            vae = as_qwen_image_vae(vae)

            vae_dtype = next(iter(vae.parameters())).dtype
            # Use the VAE's intended compute device (CUDA/MPS, or CPU if configured cpu_only). Do NOT infer it from
            # current param residency: partial loading may have temporarily offloaded all weights to RAM, which would
            # wrongly place the latents (and thus the whole decode) on the CPU (see #9373).
            latents = latents.to(device=vae_info.compute_device, dtype=vae_dtype)

            TorchDevice.empty_cache()

            with torch.inference_mode():
                # Expects 5D latents [B, C, T, H, W]
                if latents.ndim == 4:
                    latents = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                # Denormalize from denoiser space to raw VAE space
                # (same as diffusers WanPipeline and ComfyUI Wan21.process_out)
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                latents = latents * latents_std + latents_mean
                # Cheap (latents are small); the black-image triage below needs it to tell
                # upstream corruption from decode-side corruption.
                latents_finite = bool(torch.isfinite(latents).all())

                def decode_once() -> torch.Tensor:
                    out = vae.decode(latents, return_dict=False)[0]
                    if not bool(torch.isfinite(out).all()):
                        # NaN survives clamp(-1, 1) and quantizes to 0: without this, the
                        # failure renders as a silent black image. Diagnose and try to recover.
                        out = self._recover_nonfinite_decode(context, vae, latents, out, latents_finite)
                    return out

                # The cached VAE instance is shared across invocations and with the Qwen-Image
                # nodes, so the tile geometry is scoped: `enable_tiling` writes it onto the
                # module and `disable_tiling` restores only the flag, never the sizes.
                try:
                    with patch_qwen_image_vae_tiling(vae, ANIMA_VAE_TILE_SIZE if use_tiling else None):
                        decoded = decode_once()
                except RuntimeError as e:
                    if use_tiling or not is_oom_error(e):
                        raise
                    # The working-memory estimate was insufficient on this system;
                    # retry once with tiling, which caps the peak allocation.
                    context.util.signal_progress("VAE decode ran out of memory, retrying tiled")
                    context.logger.warning(
                        "VAE decode ran out of memory; retrying with tiling. The tiled result is not identical to an "
                        "untiled decode -- the decoder's normalisation and attention are global, so the difference is "
                        "spread over the image rather than confined to the seams."
                    )
                    # Drop the failed attempt's traceback before retrying. It pins that
                    # decode's frames, and their locals hold the full-resolution
                    # activations -- exception/traceback/frame is a reference cycle rooted
                    # on the stack, so `empty_cache()` frees nothing while `e` is bound and
                    # the retry has to fit on top of it. Measured at 1536px: 1.4 GiB held,
                    # and a retry that fails with it and succeeds without. Same reasoning as
                    # `ModelConfigFactory._detach_traceback`.
                    e.__traceback__ = None
                    TorchDevice.empty_cache()
                    with patch_qwen_image_vae_tiling(vae, ANIMA_VAE_TILE_SIZE):
                        decoded = decode_once()

                # Output is 5D [B, C, T, H, W] — squeeze temporal dim
                if decoded.ndim == 5:
                    decoded = decoded.squeeze(2)
                img = decoded

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)

    def _recover_nonfinite_decode(
        self,
        context: InvocationContext,
        vae: QwenImageCompatibleVAE,
        latents: torch.Tensor,
        decoded: torch.Tensor,
        latents_finite: bool,
    ) -> torch.Tensor:
        """Diagnose and, when possible, recover a decode that produced NaN/Inf (a black image).

        Seen intermittently on a dual-GPU ROCm rig while a long video generation runs on the
        other GPU. Logs a fingerprint that discriminates the candidate mechanisms — non-finite
        latents from upstream, corrupt cached VAE weights, or clean-inputs/clean-weights decode
        compute failure (the allocator-state-dependent kernel class) — then makes two bounded
        recovery attempts whose outcomes sharpen the diagnosis. Returns the best decode
        achieved; never raises, since a black image plus a diagnostic log beats a failed
        generation.
        """
        try:
            return self._recover_nonfinite_decode_impl(context, vae, latents, decoded, latents_finite)
        except Exception:
            # The triage itself must never convert a black image into a failed generation.
            context.logger.exception("VAE decode non-finite triage failed; returning the corrupt decode.")
            return decoded

    def _recover_nonfinite_decode_impl(
        self,
        context: InvocationContext,
        vae: QwenImageCompatibleVAE,
        latents: torch.Tensor,
        decoded: torch.Tensor,
        latents_finite: bool,
    ) -> torch.Tensor:
        device = latents.device
        scan = scan_module_for_nonfinite_weights(vae, device.type)

        if not latents_finite:
            assessment = (
                "the latents entering the decode already contain NaN/Inf — corruption happened "
                "UPSTREAM of the VAE (denoise output or tensor transfer); retrying the decode cannot help"
            )
        elif not scan.clean:
            assessment = (
                "NaN/Inf (or unreadable tensors) found among the cached VAE WEIGHTS — if genuinely "
                "corrupt, every decode will fail until the model is reloaded (clearing the model "
                "cache should recover it)"
            )
        else:
            assessment = (
                "latents and weights are finite but the decode COMPUTED NaN/Inf — consistent with an "
                "allocator-state-dependent kernel failure on this device"
            )
        context.logger.error(
            "VAE decode produced non-finite output (this renders as a black image).\n"
            f"  non-finite fraction of decode output: {nonfinite_fraction(decoded):.4f}\n"
            f"  latents finite: {latents_finite}\n"
            f"  weights: {scan.describe()}\n"
            f"  allocator: {allocator_state_summary(device)}\n"
            f"  assessment: {assessment}"
        )

        if not latents_finite or not scan.clean:
            return decoded

        try:
            # Attempt 1: plain retry, to rule out a transient.
            retry = vae.decode(latents, return_dict=False)[0]
            if bool(torch.isfinite(retry).all()):
                context.logger.warning("VAE decode recovered on a plain retry (transient non-finite decode).")
                return retry

            # Attempt 2: force a REAL global empty_cache — bypassing the peer-aware skip — and
            # retry. This stalls BOTH workers once: the peer's in-flight step, and this thread,
            # which waits inside hipFree until that step's kernel completes (up to ~a minute on
            # a long video step). Still a better trade than a black image; if it heals the
            # decode, the allocator-state mechanism is confirmed.
            context.logger.warning(
                "VAE decode still non-finite after a plain retry; forcing a global empty_cache "
                "(a peer GPU's in-flight step may stall once) and retrying."
            )
            force_real_empty_cache()
            retry = vae.decode(latents, return_dict=False)[0]
            if bool(torch.isfinite(retry).all()):
                context.logger.warning(
                    "VAE decode recovered after a forced global empty_cache — allocator-state-dependent "
                    "decode corruption on this device is CONFIRMED."
                )
                return retry
        except RuntimeError as e:
            context.logger.error(f"VAE decode recovery attempt raised {type(e).__name__}: {e}")
            return decoded

        context.logger.error(
            "VAE decode remained non-finite after both recovery attempts; returning the corrupt "
            "decode (the image will be black)."
        )
        return retry
