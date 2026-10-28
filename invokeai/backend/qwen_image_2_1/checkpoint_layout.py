"""The single-file layouts Qwen-Image-2.1 ships in, and how they become diffusers' modules.

Transformer: ComfyUI's export (Comfy-Org, unsloth's GGUFs) uses diffusers' names behind a
`model.diffusion_model.` prefix, except that the gated MLP's two input projections are fused into one
`img_mlp.gate_up` tensor, gate rows first. diffusers' converter splits it the same way.

VAE: the diffusers export loads as is; ComfyUI's is the original Wan-style layout (`encoder.downsamples`,
`decoder.middle`, `head`), with every 2-D convolution stored 5-D with a temporal extent of 1.
"""

import re
from typing import Any

import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

GATE_UP = ".img_mlp.gate_up."
"""ComfyUI's fused gated-MLP projection; diffusers splits it into `gate_layer` (first half) and `proj`."""

# Sidecars that describe the whole tensor and so belong to both halves unchanged.
_SHARED_SIDECARS = ("input_scale", "scale_input", "comfy_quant")


def _split_rows(value: Any, rows: int) -> tuple[Any, Any]:
    """Split a tensor's leading (output) dimension in two at `rows`.

    GGML blocks run along a row, never across rows, so a GGML tensor splits by its quantized rows
    without dequantizing.
    """
    if isinstance(value, GGMLTensor):
        total, cols = value.tensor_shape
        data = value.quantized_data
        if data.shape[0] != total:
            raise ValueError(f"GGML tensor of shape {tuple(value.tensor_shape)} is not stored one row per output")
        halves = (data[:rows].clone(), data[rows:].clone())
        shapes = (torch.Size((rows, cols)), torch.Size((total - rows, cols)))
        return tuple(  # type: ignore[return-value]
            GGMLTensor(part, value._ggml_quantization_type, shape, value.compute_dtype)
            for part, shape in zip(halves, shapes, strict=True)
        )
    return value[:rows], value[rows:]


def split_fused_gate_up(sd: dict[str, Any]) -> dict[str, Any]:
    """Split every `img_mlp.gate_up` weight, with its quantization sidecars, into `gate_layer` and `proj`.

    Runs before any quantization handling, so the int8 and fp8 helpers see one module per diffusers
    Linear: a per-output-channel scale is split with its weight, a per-tensor one is shared, and the
    `comfy_quant` marker is copied to both halves. Keys without the fused tensor pass through.
    """
    fused = {k for k in sd if GATE_UP in k}
    if not fused:
        return sd
    out = {k: v for k, v in sd.items() if k not in fused}
    weights = {k for k in fused if k.endswith(".weight")}
    halves_by_stem: dict[str, int] = {}
    for key in weights:
        stem = key[: -len(".weight")]
        rows = sd[key].shape[0]
        if rows % 2:
            raise ValueError(f"{key} has {rows} rows, which do not split into gate and up halves")
        halves_by_stem[stem] = rows // 2
        gate, up = _split_rows(sd[key], rows // 2)
        out[f"{stem.replace(GATE_UP[:-1], '.img_mlp.gate_layer')}.weight"] = gate
        out[f"{stem.replace(GATE_UP[:-1], '.img_mlp.proj')}.weight"] = up
    for key in fused - weights:
        stem, suffix = key.rsplit(".", 1)
        if stem not in halves_by_stem:
            raise ValueError(f"{key} has no fused weight to split with")
        gate_key = f"{stem.replace(GATE_UP[:-1], '.img_mlp.gate_layer')}.{suffix}"
        proj_key = f"{stem.replace(GATE_UP[:-1], '.img_mlp.proj')}.{suffix}"
        value = sd[key]
        per_row = (
            suffix not in _SHARED_SIDECARS
            and isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == 2 * halves_by_stem[stem]
        )
        if per_row:
            out[gate_key], out[proj_key] = _split_rows(value, halves_by_stem[stem])
        else:
            out[gate_key], out[proj_key] = value, value
    return out


def split_fused_layer_names(layers: dict[str, Any]) -> dict[str, Any]:
    """Give a per-layer mapping keyed by module path (header quantization hints) entries for both halves."""
    out: dict[str, Any] = {}
    for name, value in layers.items():
        if name.endswith(GATE_UP[:-1]):
            out[name.replace(GATE_UP[:-1], ".img_mlp.gate_layer")] = value
            out[name.replace(GATE_UP[:-1], ".img_mlp.proj")] = value
        else:
            out[name] = value
    return out


def count_transformer_blocks(sd: dict[str, Any]) -> int:
    """The number of transformer blocks the state dict holds, refusing a gap in their indices."""
    indices = {int(m.group(1)) for k in sd if (m := re.match(r"transformer_blocks\.(\d+)\.", k))}
    if not indices:
        raise ValueError("state dict has no transformer_blocks")
    if indices != set(range(max(indices) + 1)):
        raise ValueError(
            f"transformer_blocks are not contiguous: missing {sorted(set(range(max(indices) + 1)) - indices)}"
        )
    return max(indices) + 1


# ComfyUI VAE layout -> diffusers. Derived by matching all 238 tensors of Comfy-Org's
# `qwen_image_2.1_vae_bf16.safetensors` to the diffusers export by value; every one matched exactly once.
_RESIDUAL = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}
_VAE_RULES: list[tuple[re.Pattern[str], Any]] = [
    (re.compile(r"^conv1\.(\w+)$"), r"quant_conv.\1"),
    (re.compile(r"^conv2\.(\w+)$"), r"post_quant_conv.\1"),
    (re.compile(r"^(encoder|decoder)\.conv1\.(\w+)$"), r"\1.conv_in.\2"),
    (re.compile(r"^(encoder|decoder)\.head\.0\.gamma$"), r"\1.norm_out.gamma"),
    (re.compile(r"^(encoder|decoder)\.head\.2\.(\w+)$"), r"\1.conv_out.\2"),
    (re.compile(r"^(encoder|decoder)\.middle\.1\.(\w+)\.(\w+)$"), r"\1.mid_block.attentions.0.\2.\3"),
    (
        re.compile(r"^(encoder|decoder)\.middle\.([02])\.residual\.(\d+)\.(\w+)$"),
        lambda m: f"{m[1]}.mid_block.resnets.{int(m[2]) // 2}.{_RESIDUAL[m[3]]}.{m[4]}",
    ),
    # The last entry of each down/up block is its resampler (index 2 in the encoder's two-resnet blocks,
    # 3 in the decoder's three-resnet blocks); the entries before it are resnets.
    (
        re.compile(r"^encoder\.downsamples\.(\d+)\.downsamples\.2\.(resample\.\d+|time_conv)\.(\w+)$"),
        r"encoder.down_blocks.\1.downsampler.\2.\3",
    ),
    (
        re.compile(r"^decoder\.upsamples\.(\d+)\.upsamples\.3\.(resample\.\d+|time_conv)\.(\w+)$"),
        r"decoder.up_blocks.\1.upsampler.\2.\3",
    ),
    (
        re.compile(r"^(encoder\.down|decoder\.up)samples\.(\d+)\.\w+\.(\d+)\.residual\.(\d+)\.(\w+)$"),
        lambda m: f"{m[1]}_blocks.{m[2]}.resnets.{m[3]}.{_RESIDUAL[m[4]]}.{m[5]}",
    ),
    (
        re.compile(r"^(encoder\.down|decoder\.up)samples\.(\d+)\.\w+\.(\d+)\.shortcut\.(\w+)$"),
        r"\1_blocks.\2.resnets.\3.conv_shortcut.\4",
    ),
]


def is_comfy_vae_layout(sd: dict[str, Any]) -> bool:
    return "decoder.conv1.weight" in sd and "decoder.conv_in.weight" not in sd


def convert_comfy_vae_to_diffusers(sd: dict[str, Any]) -> dict[str, Any]:
    """Rename a ComfyUI-layout Qwen-Image-2.1 VAE to diffusers keys. Shapes stay as stored.

    A key no rule covers raises: the export is exactly 238 tensors, and a silently dropped one would
    surface only as a meta tensor at the first decode.
    """
    out: dict[str, Any] = {}
    for key, value in sd.items():
        for pattern, repl in _VAE_RULES:
            new_key, n = pattern.subn(repl, key)
            if n:
                out[new_key] = value
                break
        else:
            raise ValueError(f"unrecognized key in a ComfyUI Qwen-Image-2.1 VAE: {key}")
    return out


def fit_to_module_shapes(sd: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    """Reshape tensors stored with singleton axes (ComfyUI's 5-D convs, `[C,1,1,1]` gammas) to the module's.

    Only an element-preserving reshape is accepted; anything else is left for `load_state_dict` to reject.
    """
    shapes = {name: tuple(t.shape) for name, t in [*model.named_parameters(), *model.named_buffers()]}
    for key, value in sd.items():
        target = shapes.get(key)
        if target is not None and tuple(value.shape) != target and value.numel() == torch.Size(target).numel():
            sd[key] = value.reshape(target)
