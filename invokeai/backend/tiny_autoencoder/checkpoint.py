"""Which TAESD decoder a checkpoint holds, and its weights in the decoder's own key layout.

The one place that reads TAESD checkpoints, so identification and the loader cannot disagree about
what a file is. Three layouts are published:

- **A, native**: decoder-only, keyed by the upstream `nn.Sequential` (`1.weight`, `3.conv.0.weight`,
  ... `19.weight`). taesd, taesdxl, and ComfyUI's `vae_approx/*_decoder.pth`.
- **B, diffusers-style**: encoder and decoder in one file, `decoder.layers.N.*` and `encoder.layers.*`.
  diffusers' `DecoderTiny` keeps the `Clamp` out of its Sequential and applies it in `forward`, so its
  index N is native N+1. taesd3, taef1, taef2, taeqi2_1.
- **C, TAEHV**: madebyollin's video decoders, encoder and decoder in one file, `decoder.N.*` with native
  indices (Clamp at 0). Only taew2_1's build is a decoder here (`taehv.ImageDecoder`, upstream's last decoded
  frame): a `TGrow` after the last `MemBlock` is cut to its last-timestep output channels, to load at stride 1.

Everything is checked after normalising to layout A. A file is a TAESD decoder only if its keys and
shapes are exactly those of the decoder `build_taesd_decoder` builds for it, which is also what the
loader's strict load requires: identification never accepts a file the loader would refuse.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from invokeai.backend.tiny_autoencoder.taehv import ImageDecoder
from invokeai.backend.tiny_autoencoder.taesd import Decoder, F16Decoder, TaesdDecoder

TaesdArch = Literal["default", "flux_2", "f16", "taehv"]

SAFETENSORS_SUFFIX = ".safetensors"
PICKLE_SUFFIXES = frozenset({".pth", ".pt"})
"""ComfyUI's `vae_approx/*_decoder.pth`: a plain tensor dict, read with `weights_only=True`."""
TAESD_CHECKPOINT_SUFFIXES = frozenset({SAFETENSORS_SUFFIX, *PICKLE_SUFFIXES})
"""The file types the loader reads. Identification refuses any other, though it could read their shapes."""

_LAYOUT_B_DECODER_PREFIX = "decoder.layers."
_LAYOUT_B_ENCODER_PREFIX = "encoder."
_LAYOUT_C_DECODER_KEY = re.compile(r"decoder\.\d+\.")
_LAYOUT_C_DECODER_PREFIX = "decoder."
_TGROW_WEIGHT_KEY = re.compile(r"(\d+)\.conv\.weight")
_LAYOUT_C_MEMBLOCK_KEY = re.compile(r"decoder\.(\d+)\.conv\.0\.weight")


@dataclass(frozen=True)
class TaesdLayout:
    latent_channels: int
    arch: TaesdArch
    """`default` (8x), `flux_2` (8x, with mid-block pooling), `f16` (16x, 2x2 output patchify) or `taehv` (8x, one
    frame of the TAEHV video decoder)."""
    image_channels: int
    """3 for RGB, 4 for RGBA (TAEQI2.1)."""


def taesd_decoder_state_dict(sd: Mapping[str, Any]) -> dict[str, Any]:
    """The decoder's weights keyed by the native Sequential (layout A), whichever layout `sd` is in.

    Layout B drops its encoder and shifts each decoder index by one; layout C drops its encoder and
    strips the `decoder.` prefix (see `_taehv_decoder_state_dict`). Keys that belong to neither half
    are kept, so a strict load refuses a file that carries something this decoder does not have.
    """
    if not any(isinstance(k, str) and k.startswith(_LAYOUT_B_DECODER_PREFIX) for k in sd):
        if any(isinstance(k, str) and _LAYOUT_C_DECODER_KEY.match(k) for k in sd):
            return _taehv_decoder_state_dict(sd)
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


def _taehv_decoder_state_dict(sd: Mapping[str, Any]) -> dict[str, Any]:
    """Layout C as layout A: the encoder dropped, `decoder.` stripped, a `TGrow` after the last `MemBlock` cut to
    stride 1.

    A `TGrow` weight is `(n_f * stride, n_f, 1, 1)`, one group of `n_f` output channels per output frame. The image
    decoder keeps only the last frame once no `MemBlock` follows, so such a `TGrow` keeps the last group, as upstream's
    `patch_tgrow_layers` does when it builds a `TGrow` at a smaller stride. A `TGrow` before the last `MemBlock` keeps
    its stride. Only the shape is read, so a weight that is not a widening 1x1 convolution passes through unchanged
    for the shape check to judge.
    """
    memblocks = [int(m[1]) for k in sd if isinstance(k, str) and (m := _LAYOUT_C_MEMBLOCK_KEY.fullmatch(k))]
    last_memblock = max(memblocks, default=-1)
    converted: dict[str, Any] = {}
    for key, value in sd.items():
        if isinstance(key, str) and key.startswith(_LAYOUT_B_ENCODER_PREFIX):
            continue
        if isinstance(key, str) and _LAYOUT_C_DECODER_KEY.match(key):
            key = key.removeprefix(_LAYOUT_C_DECODER_PREFIX)
            shape = getattr(value, "shape", ())
            tgrow = _TGROW_WEIGHT_KEY.fullmatch(key)
            if (
                tgrow
                and int(tgrow[1]) > last_memblock
                and len(shape) == 4
                and shape[2:] == (1, 1)
                and shape[0] > shape[1]
                and shape[0] % shape[1] == 0
            ):
                value = value[-shape[1] :]
        converted[key] = value
    return converted


def _may_be_taesd(sd: Mapping[str, Any]) -> bool:
    """Cheap pre-check, so identifying an unrelated multi-gigabyte checkpoint never copies its key set."""
    return (
        "1.weight" in sd or f"{_LAYOUT_B_DECODER_PREFIX}0.weight" in sd or f"{_LAYOUT_C_DECODER_PREFIX}1.weight" in sd
    )


def build_taesd_decoder(layout: TaesdLayout) -> TaesdDecoder:
    if layout.arch == "taehv":
        return ImageDecoder(layout.latent_channels)
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
    if first is None or len(first) != 4 or first[2:] != (3, 3):
        return None
    latent_channels = first[1]
    # A MemBlock's first convolution reads its input and the past frame: twice the width.
    if first[0] == 256 and _shape(sd, "3.conv.0.weight") == (256, 512, 3, 3):
        return TaesdLayout(latent_channels=latent_channels, arch="taehv", image_channels=3)
    if last is None or len(last) != 4:
        return None
    if first[0] == 64 and last[1:] == (64, 3, 3):
        arch: TaesdArch = "flux_2" if "3.pool.0.weight" in sd else "default"
        return TaesdLayout(latent_channels=latent_channels, arch=arch, image_channels=last[0])
    if first[0] == 256 and last[1:] == (64, 3, 3) and last[0] % 4 == 0:
        return TaesdLayout(latent_channels=latent_channels, arch="f16", image_channels=last[0] // 4)
    return None


def _decoder_shapes(layout: TaesdLayout) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        decoder = build_taesd_decoder(layout)
    return {key: tuple(value.shape) for key, value in decoder.state_dict().items()}


def detect_taesd_layout(sd: Mapping[str, Any]) -> TaesdLayout | None:
    """The TAESD decoder `sd` holds, or None if it is not exactly one.

    Accepts any layout and a state dict on the `meta` device; only shapes are read. None for real
    VAEs (`decoder.conv_in`), the TAEHV builds other than taew2_1's shapes (patch size 2 or 4, `_super`),
    the tinier TAEQI2.1 (see `is_tinier_taesd`), the 32x Sana decoder, and a truncated or extended file.
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
