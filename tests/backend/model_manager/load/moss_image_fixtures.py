"""The MOSS-Image single-file layouts the community publishes, built from a tiny transformer or from real key lists.

- fused bf16 (Comfy-Org `moss_image_base_bf16`): `layers.N.proj.qkv` instead of `proj.q/k/v`, 379 tensors;
- int8 convrot (Comfy-Org `moss_image_base_int8_convrot`): fused, `proj.{qkv,o,ffn_up_gate,ffn_down}` as I8 with a
  per-row F32 `weight_scale` and a `.comfy_quant` marker, plus a `comfy_attention.config` per layer;
- unsloth GGUF: split keys, `general.architecture = moss_image`, some tensors BF16, the rest quantized;
- leejet GGUF: fused qkv, no metadata at all.

The dequantized values each file means are returned beside it, so a test compares a load against a dense model
holding exactly those, not against the loader's own arithmetic.
"""

import re
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize

from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot

Keys = dict[str, tuple[list[int], str]]

TINY_GEOMETRY = {
    "dim": 64,
    "ffn_dim": 96,
    "num_heads": 2,
    "num_layers": 3,
    "text_dim": 32,
    "freq_dim": 32,
    "latent_channels": 32,
}

# A power of 4, as the Hadamard requires, that divides every quantized input (64 and 96 wide).
TINY_CONVROT_GROUP = 16

CONVROT_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": TINY_CONVROT_GROUP}

_QKV_PARTS = re.compile(r"^(layers\.\d+\.proj)\.[qkv]\.(weight)$")
_BLOCK_PROJECTIONS = re.compile(r"^layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.weight$")


def tiny_reference(seed: int = 0) -> MOSSImageTransformer:
    """A tiny transformer with every weight non-trivial, so a weight that does not arrive shows."""
    torch.manual_seed(seed)
    reference = MOSSImageTransformer(**TINY_GEOMETRY).eval()
    for name, parameter in reference.named_parameters():
        if "norm" in name:
            torch.nn.init.normal_(parameter, mean=1.0, std=0.1)
    return reference


def forward(model: MOSSImageTransformer) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, TINY_GEOMETRY["latent_channels"], 4, 4, generator=generator)
    text = torch.randn(1, 5, TINY_GEOMETRY["text_dim"], generator=generator)
    with torch.no_grad():
        return model(latents, torch.tensor(500.0), text).sample


def with_state_dict(state_dict: dict[str, torch.Tensor]) -> MOSSImageTransformer:
    """A tiny transformer holding exactly `state_dict`."""
    model = MOSSImageTransformer(**TINY_GEOMETRY).eval()
    model.load_state_dict(state_dict, strict=True)
    return model


def fuse_qkv_keys(keys: Keys) -> Keys:
    """A header key list in Comfy-Org's fused layout: `proj.q/k/v` replaced by one `proj.qkv` of three times the rows."""
    fused: Keys = {}
    for key, (shape, dtype) in keys.items():
        match = _QKV_PARTS.match(key)
        if match is None:
            fused[key] = (shape, dtype)
        elif key.endswith(".proj.q.weight"):
            fused[f"{match.group(1)}.qkv.weight"] = ([3 * shape[0], *shape[1:]], dtype)
    return fused


def fuse_qkv(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same, for tensors: q, k, v stacked along dim 0."""
    fused = {key: value for key, value in state_dict.items() if _QKV_PARTS.match(key) is None}
    for key in state_dict:
        match = _QKV_PARTS.match(key)
        if match is not None and key.endswith(".proj.q.weight"):
            block = match.group(1)
            fused[f"{block}.qkv.weight"] = torch.cat(
                [state_dict[f"{block}.{name}.weight"] for name in "qkv"], dim=0
            ).contiguous()
    return fused


def int8_convrot_keys(keys: Keys) -> tuple[Keys, dict[str, dict[str, Any]]]:
    """Comfy-Org's int8 convrot header for a (split) key list: fused, the four block projections I8 with a per-row F32
    scale, a marker per projection and a `comfy_attention.config` per layer. Returns the keys and the markers."""
    out: Keys = {}
    markers: dict[str, dict[str, Any]] = {}
    for key, (shape, dtype) in fuse_qkv_keys(keys).items():
        layer = re.match(r"^(layers\.(\d+)\.proj\.(qkv|o|ffn_up_gate|ffn_down))\.weight$", key)
        if layer is None:
            out[key] = (shape, dtype)
            continue
        out[key] = (shape, "I8")
        out[f"{layer.group(1)}.weight_scale"] = ([shape[0], 1], "F32")
        markers[layer.group(1)] = dict(CONVROT_MARKER)
        out[f"layers.{layer.group(2)}.comfy_attention.config"] = ([34], "U8")
    return out, markers


def int8_convrot_state_dict(
    reference: MOSSImageTransformer,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The reference as Comfy-Org's int8 convrot file stores it, and the dense weights that file means.

    Rows of the fused projection get their own scales, so a loader that cuts the scale at the wrong rows decodes a
    plausible-looking weight that is wrong.
    """
    dense = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    fused = fuse_qkv(dense)
    stored: dict[str, torch.Tensor] = {}
    meant: dict[str, torch.Tensor] = {}
    for key, value in fused.items():
        if re.match(r"^layers\.\d+\.proj\.(qkv|o|ffn_up_gate|ffn_down)\.weight$", key) is None:
            stored[key] = value
            continue
        payload = quantize_convrot(value.float(), group_size=TINY_CONVROT_GROUP)
        layer = key[: -len(".weight")]
        stored[key] = payload.codes
        stored[f"{layer}.weight_scale"] = payload.scale
        stored[f"{layer}.comfy_quant"] = comfy_quant_marker(CONVROT_MARKER)
        # W8A8 repacks ship an activation scale per layer; this decode computes in bf16 and has nothing to apply it to.
        stored[f"{layer}.input_scale"] = torch.tensor(0.5)
        index = key.split(".")[1]
        stored[f"layers.{index}.comfy_attention.config"] = comfy_quant_marker({"attention": "comfy_kitchen_int8"})
        meant[key] = payload.dequantized
    # What the dense reference holds once those layers are replaced: q, k and v are the fused tensor's thirds.
    dense_meant = dict(dense)
    for key, value in meant.items():
        if key.endswith(".proj.qkv.weight"):
            block = key[: -len(".qkv.weight")]
            for name, part in zip("qkv", value.chunk(3, dim=0), strict=True):
                dense_meant[f"{block}.{name}.weight"] = part
        else:
            dense_meant[key] = value
    return stored, dense_meant


def write_moss_image_gguf(
    path: Path, reference: MOSSImageTransformer, *, fused: bool, architecture: str | None, prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Write `reference` as a GGUF; return what each (split) tensor dequantizes to.

    unsloth's layout (split): Linears are Q8_0, except the ones its releases keep BF16 (the modulations, the head and
    the second time-embedding layer); `video_embedder`, `time_embedding.0` and `text_embedder.proj` are quantized, so
    the skip patterns have packed tensors to act on. leejet's layout (fused, `architecture=None`, no metadata at all):
    only `proj.qkv` is quantized and every other Linear is BF16. Norms and biases are F32 in both. `prefix` is
    prepended to every key, as some ComfyUI exports do.
    """
    bf16 = ("final_linear_video", "shared_block_modulation", "video_head_modulation", "time_embedding.2")
    state = {key: value.detach() for key, value in reference.state_dict().items()}
    writer = gguf.GGUFWriter(str(path), architecture or "unused")
    if architecture is None:
        # The writer always declares an architecture; leejet's files have no key/value pairs at all.
        writer.kv_data = [{}]
    meant: dict[str, torch.Tensor] = {}
    for key, value in (fuse_qkv(state) if fused else state).items():
        data = value.to(torch.float32).numpy()
        if data.ndim == 2 and (any(name in key for name in bf16) or (fused and not key.endswith(".proj.qkv.weight"))):
            stored_as = gguf.GGMLQuantizationType.BF16
        elif data.ndim == 2 and data.shape[1] % 32 == 0:
            stored_as = gguf.GGMLQuantizationType.Q8_0
        else:
            stored_as = gguf.GGMLQuantizationType.F32
        if stored_as is gguf.GGMLQuantizationType.F32:
            writer.add_tensor(f"{prefix}{key}", data)
            meant[key] = torch.from_numpy(data.copy())
        else:
            raw = quantize(data, stored_as)
            writer.add_tensor(f"{prefix}{key}", raw, raw_shape=raw.shape, raw_dtype=stored_as)
            meant[key] = torch.from_numpy(np.asarray(dequantize(raw, stored_as))).reshape(value.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if fused:
        # The values the fused tensor means, cut back into the split keys the model has.
        for block in {key[: -len(".qkv.weight")] for key in meant if key.endswith(".proj.qkv.weight")}:
            for name, part in zip("qkv", meant.pop(f"{block}.qkv.weight").chunk(3, dim=0), strict=True):
                meant[f"{block}.{name}.weight"] = part
    return meant
