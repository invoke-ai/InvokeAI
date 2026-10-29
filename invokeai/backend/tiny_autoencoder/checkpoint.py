"""Which Tiny AutoEncoder decoder a checkpoint holds, and its weights in the decoder's own key layout.

The one place that reads TAE checkpoints, so identification and the loader cannot disagree about what
a file is. Two families are supported, TAESD (images) and TAEHV (video, decoded one frame at a time).

TAESD is published in two layouts:

- **A, native**: decoder-only, keyed by the upstream `nn.Sequential` (`1.weight`, `3.conv.0.weight`,
  ... `19.weight`). taesd, taesdxl, and ComfyUI's `vae_approx/*_decoder.pth`.
- **B, diffusers-style**: encoder and decoder in one file, `decoder.layers.N.*` and `encoder.layers.*`.
  diffusers' `DecoderTiny` keeps the `Clamp` out of its Sequential and applies it in `forward`, so its
  index N is native N+1. taesd3, taef1, taef2, taeqi2_1.

TAEHV files hold `encoder.N.*` and `decoder.N.*`, keyed by the upstream Sequentials. The decoder's
shapes give its latent width, output patch size and temporal strides, but not which model's latents it
was trained on: taehv (HunyuanVideo), taecvx (CogVideoX) and taew2_1 (Wan 2.1) share one network.
Identification settles that by name or base override (`configs/tiny_autoencoder.py`).

Everything is checked after normalising to the decoder's own keys. A file is a TAE decoder only if its
keys and shapes are exactly those of the decoder built for its layout, which is also what the loader's
strict load requires: identification never accepts a file the loader would refuse.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder
from invokeai.backend.tiny_autoencoder.taehv import TaehvDecoder
from invokeai.backend.tiny_autoencoder.taesd import Decoder, F16Decoder, TaesdDecoder

TaesdArch = Literal["default", "flux_2", "f16"]
TinyAutoencoderArch = Literal["default", "flux_2", "f16", "taehv"]

SAFETENSORS_SUFFIX = ".safetensors"
PICKLE_SUFFIXES = frozenset({".pth", ".pt"})
"""ComfyUI's `vae_approx/*_decoder.pth`: a plain tensor dict, read with `weights_only=True`."""
TINY_AUTOENCODER_CHECKPOINT_SUFFIXES = frozenset({SAFETENSORS_SUFFIX, *PICKLE_SUFFIXES})
"""The file types the loader reads. Identification refuses any other, though it could read their shapes."""

_LAYOUT_B_DECODER_PREFIX = "decoder.layers."
_LAYOUT_B_ENCODER_PREFIX = "encoder."


@dataclass(frozen=True)
class TaesdLayout:
    latent_channels: int
    arch: TaesdArch
    """`default` (8x), `flux_2` (8x, with mid-block pooling) or `f16` (16x, 2x2 output patchify)."""
    image_channels: int
    """3 for RGB, 4 for RGBA (TAEQI2.1)."""

    @property
    def spatial_compression(self) -> int:
        return 16 if self.arch == "f16" else 8


def taesd_decoder_state_dict(sd: Mapping[str, Any]) -> dict[str, Any]:
    """The decoder's weights keyed by the native Sequential (layout A), whichever layout `sd` is in.

    Layout B drops its encoder and shifts each decoder index by one. Keys that belong to neither half
    are kept, so a strict load refuses a file that carries something this decoder does not have.
    """
    if not any(isinstance(k, str) and k.startswith(_LAYOUT_B_DECODER_PREFIX) for k in sd):
        return dict(sd)
    converted: dict[str, Any] = {}
    for key, value in sd.items():
        if isinstance(key, str) and key.startswith(_LAYOUT_B_ENCODER_PREFIX):
            continue
        if isinstance(key, str) and key.startswith(_LAYOUT_B_DECODER_PREFIX):
            index, _, rest = key.removeprefix(_LAYOUT_B_DECODER_PREFIX).partition(".")
            if index.isdigit():
                key = f"{int(index) + 1}.{rest}"
        converted[key] = value
    return converted


def _may_be_taesd(sd: Mapping[str, Any]) -> bool:
    """Cheap pre-check, so identifying an unrelated multi-gigabyte checkpoint never copies its key set."""
    return "1.weight" in sd or f"{_LAYOUT_B_DECODER_PREFIX}0.weight" in sd


def build_taesd_decoder(layout: TaesdLayout) -> TaesdDecoder:
    if layout.arch == "f16":
        return F16Decoder(layout.latent_channels, layout.image_channels)
    return Decoder(
        layout.latent_channels, use_midblock_gn=layout.arch == "flux_2", image_channels=layout.image_channels
    )


def _shape(sd: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    shape = getattr(sd.get(key), "shape", None)
    return None if shape is None else tuple(shape)


def _candidate_layout(sd: Mapping[str, Any]) -> TaesdLayout | None:
    """The layout the fingerprint tensors point at, before the full key set is checked."""
    first, last = _shape(sd, "1.weight"), _shape(sd, "19.weight")
    if first is None or last is None or len(first) != 4 or len(last) != 4 or first[2:] != (3, 3):
        return None
    latent_channels = first[1]
    if first[0] == 64 and last[1:] == (64, 3, 3):
        arch: TaesdArch = "flux_2" if "3.pool.0.weight" in sd else "default"
        return TaesdLayout(latent_channels=latent_channels, arch=arch, image_channels=last[0])
    if first[0] == 256 and last[1:] == (64, 3, 3) and last[0] % 4 == 0:
        return TaesdLayout(latent_channels=latent_channels, arch="f16", image_channels=last[0] // 4)
    return None


def _decoder_shapes(layout: "TinyAutoencoderLayout") -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        decoder = build_tiny_autoencoder_decoder(layout)
    return {key: tuple(value.shape) for key, value in decoder.state_dict().items()}


def detect_taesd_layout(sd: Mapping[str, Any]) -> TaesdLayout | None:
    """The TAESD decoder `sd` holds, or None if it is not exactly one.

    Accepts either layout and a state dict on the `meta` device; only shapes are read. None for real
    VAEs (`decoder.conv_in`), TAEHV checkpoints (`decoder.N.*`), the tinier TAEQI2.1 (see
    `is_tinier_taesd`), the 32x Sana decoder, and a truncated or extended TAESD file.
    """
    if not _may_be_taesd(sd):
        return None
    normalised = taesd_decoder_state_dict(sd)
    layout = _candidate_layout(normalised)
    if layout is None:
        return None
    actual = {key: _shape(normalised, key) for key in normalised}
    return layout if actual == _decoder_shapes(layout) else None


def is_tinier_taesd(sd: Mapping[str, Any]) -> bool:
    """Whether `sd` is TAEQI2.1's "tinier" build, whose low-res blocks are ConvNeXt-style `SuperBlock`s.

    Recognised so identification can name the file to install instead; InvokeAI does not vendor
    `SuperBlock`. Its first block opens with a 7x7 depthwise convolution where `Block` has a 3x3 one.
    """
    if not _may_be_taesd(sd):
        return False
    normalised = taesd_decoder_state_dict(sd)
    first, depthwise = _shape(normalised, "1.weight"), _shape(normalised, "3.conv.0.weight")
    return (
        first is not None
        and depthwise is not None
        and len(first) == 4
        and depthwise == (first[0], 1, 7, 7)
        and "19.weight" in normalised
    )


@dataclass(frozen=True)
class TaehvLayout:
    latent_channels: int
    patch_size: int
    """Output pixel-shuffle patch: 1, 2 or 4, for 8x, 16x or 32x spatial compression."""
    time_upscale: tuple[bool, bool, bool]
    """Whether each of the three TGrow stages doubles the frame count."""

    @property
    def arch(self) -> Literal["taehv"]:
        return "taehv"

    @property
    def spatial_compression(self) -> int:
        return 8 * self.patch_size


TAEW2_1 = TaehvLayout(latent_channels=16, patch_size=1, time_upscale=(False, True, True))
"""taew2_1 (Wan 2.1 VAE). taehv and taecvx share this network for other models' latents."""
TAEW2_2 = TaehvLayout(latent_channels=48, patch_size=2, time_upscale=(False, True, True))
"""taew2_2 (Wan 2.2 TI2V-5B VAE)."""
TAELTX2_3 = TaehvLayout(latent_channels=128, patch_size=4, time_upscale=(True, True, True))
"""taeltx2_3 (LTX-2.3 and later VAEs). taeltx_2 (LTX-2) has the same network but decodes LTX-2.5 latents worse
than the linear preview; identification refuses it by name."""

SUPPORTED_TAEHV_LAYOUTS = frozenset({TAEW2_1, TAEW2_2, TAELTX2_3})
"""The TAEHV networks a supported base can use. taeh3 (24 channels) is left to MiniMax H3's own preview
path, and taehv1_5 (32 channels) decodes latents no InvokeAI architecture denoises in."""

_TAEHV_TGROW_INDICES = (7, 13, 19)


def taehv_decoder_state_dict(sd: Mapping[str, Any]) -> dict[str, Any]:
    """The decoder's weights: everything but the encoder, so a strict load refuses anything extra."""
    return {key: value for key, value in sd.items() if not (isinstance(key, str) and key.startswith("encoder."))}


def build_taehv_decoder(layout: TaehvLayout) -> TaehvDecoder:
    return TaehvDecoder(layout.latent_channels, patch_size=layout.patch_size, time_upscale=layout.time_upscale)


def _candidate_taehv_layout(sd: Mapping[str, Any]) -> TaehvLayout | None:
    first, last = _shape(sd, "decoder.1.weight"), _shape(sd, "decoder.22.weight")
    if first is None or last is None or len(first) != 4 or first[0] != 256 or first[2:] != (3, 3):
        return None
    patch_size = math.isqrt(last[0] // 3)
    if 3 * patch_size**2 != last[0]:
        return None
    strides = []
    for index in _TAEHV_TGROW_INDICES:
        grow = _shape(sd, f"decoder.{index}.conv.weight")
        if grow is None or len(grow) != 4 or grow[0] not in (grow[1], 2 * grow[1]):
            return None
        strides.append(grow[0] == 2 * grow[1])
    return TaehvLayout(
        latent_channels=first[1], patch_size=patch_size, time_upscale=(strides[0], strides[1], strides[2])
    )


def detect_taehv_layout(sd: Mapping[str, Any]) -> TaehvLayout | None:
    """The supported TAEHV decoder `sd` holds, or None if it is not exactly one.

    Accepts a state dict on the `meta` device. None for taeh3 and other TAEHV networks no supported base
    uses, the "super" decoders, TAESD checkpoints, and a truncated or extended TAEHV file.
    """
    if "decoder.1.weight" not in sd or "decoder.22.weight" not in sd:
        return None
    layout = _candidate_taehv_layout(sd)
    if layout not in SUPPORTED_TAEHV_LAYOUTS:
        return None
    normalised = taehv_decoder_state_dict(sd)
    actual = {key: _shape(normalised, key) for key in normalised}
    return layout if actual == _decoder_shapes(layout) else None


TinyAutoencoderLayout = TaesdLayout | TaehvLayout


def detect_tiny_autoencoder_layout(sd: Mapping[str, Any]) -> TinyAutoencoderLayout | None:
    """The supported TAE decoder `sd` holds, of either family, or None."""
    return detect_taesd_layout(sd) or detect_taehv_layout(sd)


def tiny_autoencoder_state_dict(sd: Mapping[str, Any], layout: TinyAutoencoderLayout) -> dict[str, Any]:
    """`sd` keyed as the decoder built for `layout` expects."""
    return taesd_decoder_state_dict(sd) if isinstance(layout, TaesdLayout) else taehv_decoder_state_dict(sd)


def build_tiny_autoencoder_decoder(layout: TinyAutoencoderLayout) -> TinyAutoencoderDecoder:
    return build_taesd_decoder(layout) if isinstance(layout, TaesdLayout) else build_taehv_decoder(layout)
