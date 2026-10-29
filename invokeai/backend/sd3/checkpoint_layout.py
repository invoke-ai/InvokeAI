"""The MMDiT layout SD3 single files ship in, and how it becomes diffusers' `SD3Transformer2DModel`.

Stability's original layout -- `joint_blocks.N.{x_block,context_block}`, one fused `attn.qkv` per
stream, `adaLN_modulation` -- is what the GGUF builds carry: city96's hold the transformer alone under
bare keys; stable-diffusion.cpp's put it behind `model.diffusion_model.`, beside the VAE
(`first_stage_model.`) and possibly the text encoders, store the qk norms quantized and `pos_embed`
without its batch dimension. Only the transformer is read: `CheckpointPrefix.strip(drop_foreign=True)`
separates it, and everything left is validated in full.

Ported from diffusers' `convert_sd3_transformer_checkpoint_to_diffusers`, but on packed GGML tensors:
the fused projections are cut, and the final layers' shift/scale halves swapped, by quantized rows,
which GGML blocks never cross.
"""

import math
from collections.abc import Mapping
from typing import Any, Literal

import torch

from invokeai.backend.model_manager.checkpoint_prefix import COMFYUI_KEY_PREFIXES
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.row_ops import split_rows, take_rows

# The transformer's top-level modules, for skipping reads before the file's prefix is known.
_MMDIT_MODULES = frozenset(
    {"pos_embed", "x_embedder", "t_embedder", "y_embedder", "context_embedder", "joint_blocks", "final_layer"}
)

# Every released SD3 has 64-wide heads. Only the qk norms record the width, and SD 3.0 Medium has none.
_DEFAULT_HEAD_DIM = 64
# diffusers' default and every released config's. With `pos_embed_max_size` set it only seeds the buffer the
# checkpoint overwrites.
_SAMPLE_SIZE = 128

_Op = Literal["copy", "split_qkv", "swap_halves", "add_batch_dim"]


def is_mmdit_transformer_key(key: str) -> bool:
    """Whether to read a tensor, decided per name before the file's prefix is known: everything behind a ComfyUI
    prefix, which `CheckpointPrefix` then separates from a bundled VAE's or encoder's namespace, and the MMDiT's
    own top-level modules in a bare file. It only saves reads; what is read is validated in full."""
    return key.startswith(COMFYUI_KEY_PREFIXES) or key.split(".", 1)[0] in _MMDIT_MODULES


def is_sd3_mmdit(sd: Mapping[Any, Any]) -> bool:
    """Whether a bare-named state dict is an SD3 MMDiT: fused joint-block attention, a T5-wide (4096) context
    embedder and a pooled CLIP-L + CLIP-G (2048) embedder."""
    context = sd.get("context_embedder.weight")
    pooled = sd.get("y_embedder.mlp.0.weight")
    return (
        "joint_blocks.0.x_block.attn.qkv.weight" in sd
        and context is not None
        and tuple(context.shape)[1:] == (4096,)
        and pooled is not None
        and tuple(pooled.shape)[1:] == (2048,)
    )


def _shape(sd: Mapping[Any, Any], key: str) -> tuple[int, ...]:
    if key not in sd:
        raise ValueError(f"the file is incomplete: it has no '{key}'")
    return tuple(int(d) for d in sd[key].shape)


def sd3_transformer_config(sd: Mapping[Any, Any]) -> dict[str, Any]:
    """`SD3Transformer2DModel` keyword arguments for the bare-named MMDiT in `sd`, read off its shapes.

    Raises ValueError for a file that is incomplete or holds tensors the diffusers model has no place for.
    """
    inner_dim, in_channels, patch_size, _ = _shape(sd, "x_embedder.proj.weight")
    caption_projection_dim, joint_attention_dim = _shape(sd, "context_embedder.weight")
    pooled_projection_dim = _shape(sd, "y_embedder.mlp.0.weight")[1]
    out_features = _shape(sd, "final_layer.linear.weight")[0]

    blocks = {int(key.split(".")[1]) for key in sd if isinstance(key, str) and key.startswith("joint_blocks.")}
    num_layers = max(blocks) + 1
    qk_norm = "rms_norm" if "joint_blocks.0.x_block.attn.ln_q.weight" in sd else None
    head_dim = _shape(sd, "joint_blocks.0.x_block.attn.ln_q.weight")[0] if qk_norm else _DEFAULT_HEAD_DIM
    if inner_dim % head_dim:
        raise ValueError(f"a width of {inner_dim} does not split into {head_dim}-wide attention heads")

    positions, remainder = divmod(math.prod(_shape(sd, "pos_embed")), inner_dim)
    pos_embed_max_size = math.isqrt(positions)
    if remainder or pos_embed_max_size**2 != positions:
        raise ValueError(f"pos_embed of shape {_shape(sd, 'pos_embed')} is not a square grid of {inner_dim}-wide rows")

    config = {
        "sample_size": _SAMPLE_SIZE,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "num_layers": num_layers,
        "attention_head_dim": head_dim,
        "num_attention_heads": inner_dim // head_dim,
        "joint_attention_dim": joint_attention_dim,
        "caption_projection_dim": caption_projection_dim,
        "pooled_projection_dim": pooled_projection_dim,
        "out_channels": out_features // (patch_size * patch_size),
        "pos_embed_max_size": pos_embed_max_size,
        "dual_attention_layers": tuple(sorted(i for i in blocks if f"joint_blocks.{i}.x_block.attn2.qkv.weight" in sd)),
        "qk_norm": qk_norm,
    }

    expected = {source for source, _, _ in _plan(config)}
    missing = sorted(expected - sd.keys())
    if missing:
        raise ValueError(f"the file is incomplete: {len(missing)} tensor(s) are missing, e.g. '{missing[0]}'")
    unexpected = sorted(sd.keys() - expected, key=str)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} tensor(s) have no place in the SD3 transformer, e.g. '{unexpected[0]}'. "
            "This is a variant of the MMDiT layout that is not supported."
        )
    return config


def _plan(config: Mapping[str, Any]) -> list[tuple[str, _Op, tuple[str, ...]]]:
    """Every MMDiT tensor the config implies, what is done to it, and the diffusers name(s) it lands under."""

    def linear(source: str, target: str, op: _Op = "copy") -> list[tuple[str, _Op, tuple[str, ...]]]:
        return [(f"{source}.{p}", op, (f"{target}.{p}",)) for p in ("weight", "bias")]

    def qkv(source: str, q: str, k: str, v: str) -> list[tuple[str, _Op, tuple[str, ...]]]:
        return [(f"{source}.{p}", "split_qkv", (f"{q}.{p}", f"{k}.{p}", f"{v}.{p}")) for p in ("weight", "bias")]

    def norm(source: str, target: str) -> list[tuple[str, _Op, tuple[str, ...]]]:
        return [(f"{source}.weight", "copy", (f"{target}.weight",))] if config["qk_norm"] else []

    plan: list[tuple[str, _Op, tuple[str, ...]]] = [
        ("pos_embed", "add_batch_dim", ("pos_embed.pos_embed",)),
        *linear("x_embedder.proj", "pos_embed.proj"),
        *linear("t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1"),
        *linear("t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2"),
        *linear("y_embedder.mlp.0", "time_text_embed.text_embedder.linear_1"),
        *linear("y_embedder.mlp.2", "time_text_embed.text_embedder.linear_2"),
        *linear("context_embedder", "context_embedder"),
        *linear("final_layer.linear", "proj_out"),
        # The original AdaLayerNormContinuous splits its projection into (shift, scale), diffusers' into (scale, shift).
        *linear("final_layer.adaLN_modulation.1", "norm_out.linear", "swap_halves"),
    ]
    num_layers = config["num_layers"]
    for i in range(num_layers):
        x, c, t = f"joint_blocks.{i}.x_block", f"joint_blocks.{i}.context_block", f"transformer_blocks.{i}"
        # The last block is `context_pre_only`: its context stream only feeds the joint attention, so it has no
        # output projection and no MLP, and its modulation is the same continuous norm as the final layer's.
        last = i == num_layers - 1
        plan += [
            *qkv(f"{x}.attn.qkv", f"{t}.attn.to_q", f"{t}.attn.to_k", f"{t}.attn.to_v"),
            *qkv(f"{c}.attn.qkv", f"{t}.attn.add_q_proj", f"{t}.attn.add_k_proj", f"{t}.attn.add_v_proj"),
            *norm(f"{x}.attn.ln_q", f"{t}.attn.norm_q"),
            *norm(f"{x}.attn.ln_k", f"{t}.attn.norm_k"),
            *norm(f"{c}.attn.ln_q", f"{t}.attn.norm_added_q"),
            *norm(f"{c}.attn.ln_k", f"{t}.attn.norm_added_k"),
            *linear(f"{x}.attn.proj", f"{t}.attn.to_out.0"),
            *linear(f"{x}.adaLN_modulation.1", f"{t}.norm1.linear"),
            *linear(f"{c}.adaLN_modulation.1", f"{t}.norm1_context.linear", "swap_halves" if last else "copy"),
            *linear(f"{x}.mlp.fc1", f"{t}.ff.net.0.proj"),
            *linear(f"{x}.mlp.fc2", f"{t}.ff.net.2"),
        ]
        if not last:
            plan += [
                *linear(f"{c}.attn.proj", f"{t}.attn.to_add_out"),
                *linear(f"{c}.mlp.fc1", f"{t}.ff_context.net.0.proj"),
                *linear(f"{c}.mlp.fc2", f"{t}.ff_context.net.2"),
            ]
        # SD 3.5 Medium's MMDiT-X blocks add a second, image-only self-attention.
        if i in config["dual_attention_layers"]:
            plan += [
                *qkv(f"{x}.attn2.qkv", f"{t}.attn2.to_q", f"{t}.attn2.to_k", f"{t}.attn2.to_v"),
                *norm(f"{x}.attn2.ln_q", f"{t}.attn2.norm_q"),
                *norm(f"{x}.attn2.ln_k", f"{t}.attn2.norm_k"),
                *linear(f"{x}.attn2.proj", f"{t}.attn2.to_out.0"),
            ]
    return plan


def _split_qkv(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = value.shape[0]
    if rows % 3:
        raise ValueError(f"a fused qkv projection of {rows} rows does not split into q, k and v")
    q, kv = split_rows(value, rows // 3)
    k, v = split_rows(kv, rows // 3)
    return q, k, v


def _swap_halves(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[0] // 2
    return take_rows(value, torch.cat((torch.arange(half, 2 * half), torch.arange(half))))


def _add_batch_dim(value: torch.Tensor) -> torch.Tensor:
    """`pos_embed` as the (1, positions, width) buffer diffusers holds; stable-diffusion.cpp drops the 1."""
    if len(value.shape) == 3:
        return value
    if isinstance(value, GGMLTensor):
        return GGMLTensor(
            value.quantized_data.unsqueeze(0),
            value._ggml_quantization_type,
            torch.Size((1, *value.tensor_shape)),
            value.compute_dtype,
        )
    return value.unsqueeze(0)


def convert_mmdit_to_diffusers(sd: dict[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The bare-named MMDiT in `sd` under diffusers' names, for `config` from `sd3_transformer_config`.

    `sd` is emptied as it goes, so each fused tensor is released once its parts exist. GGML tensors stay packed.
    """
    converted: dict[str, torch.Tensor] = {}
    for source, op, targets in _plan(config):
        value = sd.pop(source)
        match op:
            case "copy":
                parts: tuple[torch.Tensor, ...] = (value,)
            case "split_qkv":
                parts = _split_qkv(value)
            case "swap_halves":
                parts = (_swap_halves(value),)
            case "add_batch_dim":
                parts = (_add_batch_dim(value),)
        converted.update(zip(targets, parts, strict=True))
    return converted
