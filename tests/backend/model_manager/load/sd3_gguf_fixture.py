"""A real, tiny SD3 transformer GGUF, written in Stability's MMDiT layout the way the published builds pack it.

city96's builds hold the transformer alone under bare keys, quantize the Linears inside the joint blocks and keep
everything else -- biases, norms, embedders, the final layer, `pos_embed` -- unquantized. stable-diffusion.cpp's put
the same tensors behind `model.diffusion_model.` beside the VAE (`first_stage_model.`), quantize the qk norms too,
and store `pos_embed` without its batch dimension.

The model is SD 3.5's shape at toy size: qk norms, two MMDiT-X dual-attention blocks -- two, so a second
attention landing in the wrong block shows -- and a last block that is `context_pre_only`. The text widths stay
real (4096 and 2048) because identification reads them. Every tensor is random, norms around one, so no
zero-initialised modulation can make a forward comparison vacuous.
"""

from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize

# Every Linear input a multiple of 32, the Q8_0 block size; head_dim 32 is one block for a quantized qk norm.
TINY_CONFIG = {
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 3,
    "attention_head_dim": 32,
    "num_attention_heads": 2,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 64,
    "pooled_projection_dim": 2048,
    "out_channels": 4,
    "pos_embed_max_size": 8,
    "dual_attention_layers": (0, 1),
    "qk_norm": "rms_norm",
}

SD_CPP_PREFIX = "model.diffusion_model."


def mmdit_shapes(config: dict = TINY_CONFIG) -> dict[str, tuple[int, ...]]:
    """Every tensor of the original MMDiT layout for ``config``, with its shape."""
    d = config["attention_head_dim"] * config["num_attention_heads"]
    p = config["patch_size"]
    head = (config["attention_head_dim"],)
    shapes: dict[str, tuple[int, ...]] = {"pos_embed": (1, config["pos_embed_max_size"] ** 2, d)}

    def linear(name: str, out_features: int, in_features: int) -> None:
        shapes[f"{name}.weight"] = (out_features, in_features)
        shapes[f"{name}.bias"] = (out_features,)

    shapes["x_embedder.proj.weight"] = (d, config["in_channels"], p, p)
    shapes["x_embedder.proj.bias"] = (d,)
    linear("t_embedder.mlp.0", d, 256)
    linear("t_embedder.mlp.2", d, d)
    linear("y_embedder.mlp.0", d, config["pooled_projection_dim"])
    linear("y_embedder.mlp.2", d, d)
    linear("context_embedder", config["caption_projection_dim"], config["joint_attention_dim"])
    linear("final_layer.linear", p * p * config["out_channels"], d)
    linear("final_layer.adaLN_modulation.1", 2 * d, d)
    for i in range(config["num_layers"]):
        last = i == config["num_layers"] - 1
        dual = i in config["dual_attention_layers"]
        for stream in ("x_block", "context_block"):
            block = f"joint_blocks.{i}.{stream}"
            linear(f"{block}.attn.qkv", 3 * d, d)
            shapes[f"{block}.attn.ln_q.weight"] = head
            shapes[f"{block}.attn.ln_k.weight"] = head
            if stream == "context_block" and last:
                linear(f"{block}.adaLN_modulation.1", 2 * d, d)
                continue
            linear(f"{block}.attn.proj", d, d)
            linear(f"{block}.adaLN_modulation.1", (9 if dual and stream == "x_block" else 6) * d, d)
            linear(f"{block}.mlp.fc1", 4 * d, d)
            linear(f"{block}.mlp.fc2", d, 4 * d)
        if dual:
            block = f"joint_blocks.{i}.x_block.attn2"
            linear(f"{block}.qkv", 3 * d, d)
            shapes[f"{block}.ln_q.weight"] = head
            shapes[f"{block}.ln_k.weight"] = head
            linear(f"{block}.proj", d, d)
    return shapes


def tiny_mmdit_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for name, shape in mmdit_shapes().items():
        noise = torch.randn(shape, generator=generator)
        if ".ln_" in name:
            sd[name] = 1 + 0.2 * noise
        elif len(shape) >= 2 and name != "pos_embed":
            sd[name] = noise / np.sqrt(np.prod(shape[1:]))
        else:
            sd[name] = 0.1 * noise
    return sd


def _stored_as(name: str, ndim: int, sd_cpp: bool) -> gguf.GGMLQuantizationType:
    if name.startswith("joint_blocks.") and ndim == 2:
        return gguf.GGMLQuantizationType.Q8_0
    if ".ln_" in name:
        return gguf.GGMLQuantizationType.Q8_0 if sd_cpp else gguf.GGMLQuantizationType.F32
    return gguf.GGMLQuantizationType.F16


def write_sd3_gguf(path: Path, *, sd_cpp: bool = False, seed: int = 0) -> dict[str, torch.Tensor]:
    """Write the tiny transformer to ``path``; return what each tensor dequantizes to, under its bare MMDiT name.

    ``sd_cpp`` writes stable-diffusion.cpp's layout: prefixed keys, quantized qk norms, a 2-D `pos_embed` and a
    bundled VAE tensor. The returned dict is in city96's layout either way, `pos_embed` included.
    """
    writer = gguf.GGUFWriter(str(path), "sd3")
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in tiny_mmdit_state_dict(seed).items():
        data = tensor.numpy()
        if sd_cpp and name == "pos_embed":
            data = data[0]
        stored_as = _stored_as(name, data.ndim, sd_cpp)
        key = f"{SD_CPP_PREFIX}{name}" if sd_cpp else name
        if stored_as is gguf.GGMLQuantizationType.F32:
            writer.add_tensor(key, data)
            stored = data
        elif stored_as is gguf.GGMLQuantizationType.F16:
            writer.add_tensor(key, data.astype(np.float16))
            stored = data.astype(np.float16).astype(np.float32)
        else:
            raw = quantize(data, stored_as)
            # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
            writer.add_tensor(key, raw, raw_shape=raw.shape, raw_dtype=stored_as)
            stored = dequantize(raw, stored_as)
        meant[name] = torch.from_numpy(np.ascontiguousarray(stored)).reshape(tensor.shape)
    if sd_cpp:
        writer.add_tensor("first_stage_model.decoder.conv_in.bias", np.ones(8, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant
