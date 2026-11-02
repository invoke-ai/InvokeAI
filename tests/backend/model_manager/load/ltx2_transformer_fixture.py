"""Toy LTX-2 transformers in the official key layout: as a state dict, or as a GGUF packed like the releases.

Shared by the loader tests of both containers. The real ``LTX2VideoTransformer3DModel`` is built at toy size and
its weights are written under the *official* names -- the inverse of the renames the loaders apply -- so what a
test pins is the whole conversion chain landing every tensor back on the model.

The GGUF layout follows the published headers (realrebelai, Abiray, vantagewithai, molbal, ruygar, joeygambino):
every per-block Linear quantized, the attention gates F16 in realrebelai's builds; the norms, biases, scale-shift
tables, the keyframe embedding and the stem and adaLN Linears unquantized, in F32 (joeygambino: F16, molbal:
partly BF16); the two text connectors bundled beside the transformer. The departures, each a ``write_ltx2_gguf`` argument: ruygar
quantizes the stem and adaLN Linears too, molbal the keyframe embedding, and realrebelai and joeygambino store
that embedding flattened, with its real shape in ``comfy.gguf.orig_shape``.
"""

import json
from pathlib import Path
from typing import Callable

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize

from invokeai.backend.ltx2 import component_configs as cc
from tests.fixtures.quantized_payloads import q8_cr_marker, random_q4_k

# One block. A 128-wide video stream (so an nvfp4 layer's 128-row tiles fit) and a 32-wide audio stream. The
# cross-attention widths equal the stream widths, as the release's do (4096 and 2048), so the model runs a forward.
TINY_TRANSFORMER = {
    **cc.LTX2_5_TRANSFORMER_CONFIG,
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 64,
    "cross_attention_dim": 128,
    "audio_in_channels": 4,
    "audio_out_channels": 4,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 16,
    "audio_cross_attention_dim": 32,
    "num_layers": 1,
    "caption_channels": 8,
}

# What the published Q4_K_M builds mostly hold: K-quants, whose 256-element super-blocks need every quantized
# Linear input to be a multiple of 256, so both streams are 256 wide here.
K_QUANT_TRANSFORMER = {
    **TINY_TRANSFORMER,
    "attention_head_dim": 128,
    "cross_attention_dim": 256,
    "audio_attention_head_dim": 128,
    "audio_cross_attention_dim": 256,
}

PREFIX = "model.diffusion_model."

# diffusers attribute path -> official checkpoint spelling; ordered so a longer name is rewritten before the
# shorter name it contains. Written independently of the loaders' tables: the test is that their forward
# conversion lands every one of these back on the model.
_TO_OFFICIAL = (
    ("audio_time_embed.", "audio_adaln_single."),
    ("time_embed.", "adaln_single."),
    ("av_cross_attn_video_scale_shift", "av_ca_video_scale_shift_adaln_single"),
    ("av_cross_attn_video_a2v_gate", "av_ca_a2v_gate_adaln_single"),
    ("av_cross_attn_audio_scale_shift", "av_ca_audio_scale_shift_adaln_single"),
    ("av_cross_attn_audio_v2a_gate", "av_ca_v2a_gate_adaln_single"),
    ("video_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_video"),
    ("audio_a2v_cross_attn_scale_shift_table", "scale_shift_table_a2v_ca_audio"),
    ("audio_proj_in", "audio_patchify_proj"),
    ("proj_in", "patchify_proj"),
    ("norm_q", "q_norm"),
    ("norm_k", "k_norm"),
    ("audio_prompt_adaln.", "audio_prompt_adaln_single."),
    ("prompt_adaln.", "prompt_adaln_single."),
)
_EXCLUSIVE = {"audio_time_embed.", "audio_proj_in", "audio_prompt_adaln."}

# Bundled beside the transformer by every release; the component folder supplies them instead.
CONNECTOR_TENSORS = (
    "video_embeddings_connector.learnable_registers",
    "audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight",
)


def official_name(diffusers_name: str, prefix: str = PREFIX) -> str:
    name = diffusers_name
    for new, old in _TO_OFFICIAL:
        if new in name:
            name = name.replace(new, old, 1)
            if new in _EXCLUSIVE:
                break
    return prefix + name


def tiny_transformer(geometry: dict = TINY_TRANSFORMER, *, keyframe_embedding: bool = True, seed: int = 0):
    """The toy transformer with every parameter non-zero.

    diffusers starts several at zero (the output projections, the keyframe embedding), and a zero reads the
    same from any slot: a loader that dropped or swapped one would compare equal.
    """
    from diffusers import LTX2VideoTransformer3DModel

    torch.manual_seed(seed)
    model = LTX2VideoTransformer3DModel(**{**geometry, "use_keyframes_abs_pos_embedding": keyframe_embedding})
    with torch.no_grad():
        for param in model.parameters():
            if not param.any():
                param.normal_(std=0.02)
    return model.eval()


def tiny_transformer_checkpoint(*, keyframe_embedding: bool = True, prefix: str = PREFIX) -> dict[str, torch.Tensor]:
    model = tiny_transformer(keyframe_embedding=keyframe_embedding)
    return {official_name(k, prefix): v.detach().clone() for k, v in model.state_dict().items()}


Packs = Callable[[str, tuple[int, ...]], bool]


def packs_block_linears(name: str, shape: tuple[int, ...]) -> bool:
    """Every release: the per-block Linears, gates aside (realrebelai keeps those F16)."""
    return (
        name.startswith("transformer_blocks.")
        and name.endswith(".weight")
        and len(shape) == 2
        and not name.endswith("to_gate_logits.weight")
    )


def packs_every_linear(name: str, shape: tuple[int, ...]) -> bool:
    """ruygar's builds: the stem and adaLN Linears are quantized as well."""
    return name.endswith(".weight") and len(shape) == 2 and "scale_shift_table" not in name


def write_ltx2_gguf(
    path: Path,
    *,
    geometry: dict = TINY_TRANSFORMER,
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    packs: Packs = packs_block_linears,
    unquantized: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.F32,
    prefix: str = "",
    architecture: str = "ltxv",
    metadata: dict[str, str] | None = None,
    q8_cr_layer: str | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Write the toy transformer as a GGUF; return what each tensor dequantizes to, keyed by its diffusers name.

    A tensor ``packs`` selects is stored as ``qtype`` when its rows fit the type's blocks, and as ``unquantized``
    otherwise, as converters do. ``q8_cr_layer`` (a diffusers weight name) is written the way ComfyUI-GGUF stores a
    ``Q8_CR`` layer: int8 codes, a scale beside them, and its marker in the metadata.
    """
    model = tiny_transformer(geometry, seed=seed)
    rng = np.random.default_rng(seed)
    block_size = gguf.GGML_QUANT_SIZES[qtype][0]

    writer = gguf.GGUFWriter(str(path), architecture)
    for key, value in (metadata or {}).items():
        writer.add_string(key, value)

    meant: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().to(torch.float32).numpy()
        stored_name = official_name(name, prefix)
        if name == q8_cr_layer:
            writer.add_string(f"comfy.gguf.quant.{stored_name}", json.dumps(q8_cr_marker()))
            writer.add_tensor(stored_name, np.zeros(data.shape, np.int8), raw_dtype=gguf.GGMLQuantizationType.I8)
            writer.add_tensor(f"{stored_name}_scale", np.ones((1, data.shape[0]), np.float32))
            continue
        if packs(name, data.shape) and data.shape[-1] % block_size == 0:
            stored_as = qtype
        elif name.endswith("to_gate_logits.weight"):
            stored_as = gguf.GGMLQuantizationType.F16
        else:
            stored_as = unquantized
        if name == "keyframes_abs_pos_embedding" and stored_as is unquantized:
            writer.add_array(f"comfy.gguf.orig_shape.{stored_name}", list(data.shape))
            data = data.reshape(-1)
        meant[name] = _add(writer, stored_name, data, stored_as, rng).reshape(tensor.shape)

    connector_type = qtype if 32 % block_size == 0 else gguf.GGMLQuantizationType.F16
    for name in CONNECTOR_TENSORS:
        _add(writer, prefix + name, rng.standard_normal((32, 32)).astype(np.float32), connector_type, rng)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant


def _add(
    writer: gguf.GGUFWriter, name: str, data: np.ndarray, stored_as: gguf.GGMLQuantizationType, rng: np.random.Generator
) -> torch.Tensor:
    if stored_as is gguf.GGMLQuantizationType.F32:
        writer.add_tensor(name, data)
        return torch.from_numpy(data.copy())
    raw = random_q4_k(data.shape, rng) if stored_as is gguf.GGMLQuantizationType.Q4_K else quantize(data, stored_as)
    # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
    writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
    return torch.from_numpy(dequantize(raw, stored_as).astype(np.float32))


def run_forward(model: torch.nn.Module, geometry: dict = TINY_TRANSFORMER) -> tuple[torch.Tensor, torch.Tensor]:
    """One float32 forward over a 1x2x2 latent grid, three audio rows and a five-token prompt, called the way
    the denoise loop calls it (``invokeai.backend.ltx2.denoise``)."""
    generator = torch.Generator().manual_seed(1)
    frames, height, width, audio_rows, tokens = 1, 2, 2, 3, 5
    cpu = torch.device("cpu")
    timestep = torch.tensor([500.0])
    mask = torch.ones(1, tokens, dtype=torch.long)
    with torch.no_grad():
        video, audio = model(
            hidden_states=torch.randn(1, frames * height * width, geometry["in_channels"], generator=generator),
            audio_hidden_states=torch.randn(1, audio_rows, geometry["audio_in_channels"], generator=generator),
            timestep=timestep,
            audio_timestep=timestep,
            sigma=timestep,
            num_frames=frames,
            height=height,
            width=width,
            fps=24.0,
            audio_num_frames=audio_rows,
            video_coords=model.rope.prepare_video_coords(1, frames, height, width, cpu, fps=24.0),
            audio_coords=model.audio_rope.prepare_audio_coords(1, audio_rows, cpu),
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            use_cross_timestep=True,
            return_dict=False,
            encoder_hidden_states=torch.randn(1, tokens, geometry["cross_attention_dim"], generator=generator),
            audio_encoder_hidden_states=torch.randn(
                1, tokens, geometry["audio_cross_attention_dim"], generator=generator
            ),
            encoder_attention_mask=mask,
            audio_encoder_attention_mask=mask,
        )
    return video, audio
