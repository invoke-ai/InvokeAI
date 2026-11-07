"""A tiny FLUX.3 transformer, written out in Black Forest Labs' native key layout.

Shared by the conversion, loader and identification tests. The native names are enumerated here from the
layout every release uses (``tests/backend/model_manager/load/state_dicts/flux3_klein_keys.py`` is the
captured real one), not derived from the converter under test; ``test_conversion`` checks the two agree.

The head width is the released 128: the rotary axes are fixed at (32, 32, 32, 32) and the loader refuses any
other width. Everything else is small. Weights are random rather than the diffusers zero-init, so a forward
through a mis-wired parameter cannot pass by multiplying by zero. The widths are multiples of 256, so every
block linear can be quantized the way the releases quantize it: int8 rotates in groups of 256, and nvfp4 needs
whole 128-row, 64-column scale tiles.
"""

import json
from pathlib import Path
from typing import NamedTuple

import gguf
import torch
from diffusers import Flux3Transformer2DModel
from gguf.quants import dequantize, quantize

from invokeai.backend.flux3.conversion import convert_flux3_native_to_diffusers, flux3_native_key_to_diffusers
from tests.fixtures.quantized_payloads import (
    comfy_quant_marker,
    nvfp4_codes,
    nvfp4_modelopt_tensors,
    nvfp4_tensors,
    nvfp4_weight,
    quantize_convrot,
    quantize_per_row_fp8,
    stored_layout,
)

TINY_CONFIG = {
    "in_channels": 32,
    "cond_in_channels": 32,
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 128,
    "num_attention_heads": 2,
    "joint_attention_dim": 256,
    "timestep_channels": 32,
}

_BLOCK_PARAMS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "attn_out.weight",
    "mlp_in.weight",
    "mlp_out.weight",
    "norm.query_norm.scale",
    "norm.key_norm.scale",
)
_ADALN = "final_layer.image.adaLN_modulation.1.weight"


def native_flux3_keys(num_layers: int, num_single_layers: int, flow_map: bool) -> list[str]:
    keys = [
        "emb_in.image.weight",
        "emb_in.image_cond.weight",
        "txt_in.weight",
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
        "final_layer.image.linear.weight",
        _ADALN,
    ]
    if flow_map:
        keys += ["ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight"]
    for stage in ("early_stream_modulations", "single_stream_modulations"):
        keys += [f"{stage}.{mode}.lin.weight" for mode in ("image", "image_cond", "txt")]
    stacks = [f"content_mode_blocks.{mode}.{i}" for mode in ("image", "image_cond") for i in range(num_layers)]
    stacks += [f"txt_mode_blocks.{i}" for i in range(num_layers)]
    stacks += [f"single_blocks.{i}" for i in range(num_single_layers)]
    keys += [f"{stack}.{param}" for stack in stacks for param in _BLOCK_PARAMS]
    return keys


def tiny_flux3_model(flow_map: bool, seed: int = 0) -> Flux3Transformer2DModel:
    torch.manual_seed(seed)
    model = Flux3Transformer2DModel(**TINY_CONFIG, flow_map=flow_map).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model


def to_native(model: Flux3Transformer2DModel) -> dict[str, torch.Tensor]:
    """The model's weights under BFL's names, with the output adaLN in BFL's [shift, scale] order."""
    diffusers_sd = model.state_dict()
    native = {}
    for key in native_flux3_keys(TINY_CONFIG["num_layers"], TINY_CONFIG["num_single_layers"], model.config.flow_map):
        value = diffusers_sd[flux3_native_key_to_diffusers(key)].clone()
        if key == _ADALN:
            scale, shift = value.chunk(2, dim=0)
            value = torch.cat([shift, scale], dim=0)
        native[key] = value
    return native


def forward(model: torch.nn.Module, flow_map: bool, with_reference: bool = True) -> torch.Tensor:
    """One step over four image tokens, two reference tokens and three text tokens."""
    generator = torch.Generator().manual_seed(1)
    kwargs = {}
    if with_reference:
        kwargs = {
            "image_cond_hidden_states": torch.randn(1, 2, TINY_CONFIG["cond_in_channels"], generator=generator),
            "image_cond_ids": torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.5, -1.0]]),
        }
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, 4, TINY_CONFIG["in_channels"], generator=generator),
            encoder_hidden_states=torch.randn(1, 3, TINY_CONFIG["joint_attention_dim"], generator=generator),
            timestep=torch.tensor([0.7]),
            img_ids=torch.tensor([[0.0, h * 0.5, w * 0.5, 0.0] for h in range(2) for w in range(2)]),
            txt_ids=torch.tensor([[0.0, 0.0, 0.0, float(i)] for i in range(3)]),
            target_timestep=torch.tensor([0.4]) if flow_map else None,
            return_dict=False,
            **kwargs,
        )[0]


def write_flux3_gguf(
    path: Path,
    native: dict[str, torch.Tensor],
    *,
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Write ``native`` the way the published GGUFs do: 2D weights in ``qtype``, norms in F32, the rest BF16.

    The architecture string is the published one, ``flux``. Returns the values a reader dequantizes each
    tensor to, so a reference model can be built from what the file means.
    """
    writer = gguf.GGUFWriter(str(path), "flux")
    block_size = gguf.GGML_QUANT_SIZES[qtype][0]
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in native.items():
        data = tensor.detach().to(torch.float32).numpy()
        if name.endswith(".scale"):
            stored_as = gguf.GGMLQuantizationType.F32
        elif data.ndim == 2 and data.shape[-1] % block_size == 0 and not name.startswith(("final_layer", "time_in")):
            stored_as = qtype
        else:
            stored_as = gguf.GGMLQuantizationType.BF16
        raw = quantize(data, stored_as)
        writer.add_tensor(prefix + name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
        meant[name] = torch.from_numpy(dequantize(raw, stored_as)).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant


QUANTIZED_SCHEMES = ("int8_convrot", "nvfp4_comfy", "nvfp4_bfl", "fp8r")
"""The released builds, as `quantize_flux3_native` names them."""

ADALN = "final_layer.image.adaLN_modulation.1"
NVFP4_GLOBAL_SCALE = 1 / 32
"""With block scales of 1 and 2 an nvfp4 weight is +-1/32 or +-1/16: the tiny model's magnitude, exact in float32."""

_LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
_QKV = _LINEARS[:3]
_MODULATIONS = tuple(
    f"{stage}.{mode}.lin"
    for stage in ("early_stream_modulations", "single_stream_modulations")
    for mode in ("image", "image_cond", "txt")
)
_INT8_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
# Comfy-Org's int8 build marks the image and the joint blocks for its own int8 attention kernel.
_COMFY_ATTENTION_HINT = {"attention": "comfy_kitchen_int8"}

# Which blocks each release quantizes, on the tiny model's one block per mode stack and two joint blocks: int8 all of
# them; nvfp4 all but the text blocks, the first image block and the last joint blocks; fp8r all but the text blocks,
# plus the six modulations. The safetensors metadata is each release's own.
_BLOCKS = {
    "int8_convrot": (
        "content_mode_blocks.image.0",
        "content_mode_blocks.image_cond.0",
        "txt_mode_blocks.0",
        "single_blocks.0",
        "single_blocks.1",
    ),
    "nvfp4": ("content_mode_blocks.image_cond.0", "single_blocks.0"),
    "fp8r": ("content_mode_blocks.image.0", "content_mode_blocks.image_cond.0", "single_blocks.0", "single_blocks.1"),
}
_METADATA = {
    "int8_convrot": {"format": "pt", "model": "f3-image-small"},
    "nvfp4_comfy": {"format": "pt", "model": "f3-image-small"},
    "nvfp4_bfl": {"format": "pt", "model": "f3-image-small", "quantization": "nvfp4"},
    "fp8r": {"format": "pt", "model": "f3-image-small", "quantization": "fp8r", "scale": "per_row_fp32"},
}


class QuantizedFlux3(NamedTuple):
    state_dict: dict[str, torch.Tensor]
    """The native checkpoint as the build stores it."""
    meant: dict[str, torch.Tensor]
    """Every weight dense, as the build means it, a fused ``qkv`` as its three projections: a model converted from
    these is what a load must give."""
    metadata: dict[str, str]
    """The file's safetensors metadata."""


def quantize_flux3_native(
    native: dict[str, torch.Tensor], scheme: str, *, adaln: bool = False, fuse_qkv: bool = False, seed: int = 0
) -> QuantizedFlux3:
    """``native`` written the way the release ``scheme`` of `QUANTIZED_SCHEMES` writes it.

    ``int8_convrot`` is Comfy-Org's: a block's q, k and v fused into one ``qkv`` (as that build always does),
    ``int8_tensorwise`` + convrot codes with a ``[N, 1]`` scale and a ``.comfy_quant`` marker per layer. ``nvfp4_comfy``
    and ``nvfp4_bfl`` hold the same numbers, in ComfyUI's layout with the layers named in ``_quantization_metadata``,
    and in ModelOpt's with the metadata saying ``quantization: nvfp4``; their weights are drawn rather than quantized,
    random signs under a random grid of block scales 1 and 2, so the expected weight is exact. ``fp8r`` is BFL's per-row
    e4m3 with a ``[N]`` scale.

    ``adaln`` quantizes the output adaLN as well, which no release does; ``fuse_qkv`` fuses the nvfp4 ``qkv`` of
    ComfyUI's layout as the int8 build does.
    """
    generator = torch.Generator().manual_seed(seed)
    stored = dict(native)
    meant = dict(native)
    blocks = _BLOCKS["nvfp4" if scheme.startswith("nvfp4") else scheme]
    fused = scheme == "int8_convrot" or (scheme == "nvfp4_comfy" and fuse_qkv)
    layers: list[str] = []
    for block in blocks:
        if fused:
            weights = [stored.pop(f"{block}.{name}.weight") for name in _QKV]
            stored[f"{block}.qkv.weight"] = torch.cat(weights)
            layers += [f"{block}.qkv", *(f"{block}.{name}" for name in _LINEARS[3:])]
        else:
            layers += [f"{block}.{name}" for name in _LINEARS]
        if scheme == "int8_convrot" and not block.startswith(("content_mode_blocks.image_cond", "txt_mode_blocks")):
            stored[f"{block}.comfy_attention.config"] = comfy_quant_marker(_COMFY_ATTENTION_HINT)
    if scheme == "fp8r":
        layers += list(_MODULATIONS)
    if adaln:
        layers.append(ADALN)

    for layer in layers:
        weight = stored.pop(f"{layer}.weight")
        if scheme == "int8_convrot":
            payload = quantize_convrot(weight)
            stored[f"{layer}.weight"], stored[f"{layer}.weight_scale"] = payload.codes, payload.scale
            stored[f"{layer}.comfy_quant"] = comfy_quant_marker(_INT8_MARKER)
            meant[f"{layer}.weight"] = payload.dequantized
        elif scheme == "fp8r":
            fp8 = quantize_per_row_fp8(weight)
            stored[f"{layer}.weight"], stored[f"{layer}.weight_scale"] = fp8.codes, fp8.scale
            meant[f"{layer}.weight"] = fp8.dequantized
        else:
            positive = torch.randint(0, 2, weight.shape, generator=generator, dtype=torch.bool)
            grid = torch.randint(1, 3, (weight.shape[0], weight.shape[1] // 16), generator=generator).float()
            if scheme == "nvfp4_comfy":
                stored.update(
                    nvfp4_tensors(
                        layer, nvfp4_codes(positive), block_scale=stored_layout(grid), global_scale=NVFP4_GLOBAL_SCALE
                    )
                )
            else:
                stored.update(
                    nvfp4_modelopt_tensors(layer, nvfp4_codes(positive), grid=grid, global_scale=NVFP4_GLOBAL_SCALE)
                )
            # Both releases ship an activation scale for fp4 compute, which a dequantizing load has no use for.
            stored[f"{layer}.input_scale"] = torch.tensor(1.0)
            meant[f"{layer}.weight"] = nvfp4_weight(positive, grid, NVFP4_GLOBAL_SCALE)

    # A fused layer means its query, key and value rows, in that order: stated here rather than left to the
    # converter under test, so that a reference built from `meant` does not share its split.
    for layer in [layer for layer in layers if layer.endswith(".qkv")]:
        block = layer[: -len(".qkv")]
        for name, part in zip(_QKV, meant.pop(f"{layer}.weight").chunk(3), strict=True):
            meant[f"{block}.{name}.weight"] = part

    metadata = dict(_METADATA[scheme])
    if scheme == "nvfp4_comfy":
        metadata["_quantization_metadata"] = json.dumps({"layers": {layer: {"format": "nvfp4"} for layer in layers}})
    return QuantizedFlux3(stored, meant, metadata)


def flux3_model_from_native(native: dict[str, torch.Tensor], flow_map: bool) -> Flux3Transformer2DModel:
    """A float32 reference model holding ``native``'s values, converted to the diffusers layout."""
    model = Flux3Transformer2DModel(**TINY_CONFIG, flow_map=flow_map).eval()
    model.load_state_dict(convert_flux3_native_to_diffusers(dict(native)))
    return model
