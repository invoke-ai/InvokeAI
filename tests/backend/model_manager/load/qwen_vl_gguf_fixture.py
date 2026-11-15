"""A real, tiny llama.cpp Qwen2.5-VL language-model GGUF, the dense model it was written from, and the loader run.

Used by the loader's tests, which judge what the production loader builds against the dense model.

The file is written from a seeded dense ``Qwen2_5_VLForConditionalGeneration`` the way llama.cpp's converter writes
one: its own tensor names (the table below, kept independent of the loader's mapping), no Q/K permutation for this
architecture, matrices genuinely Q8_0 and norms and biases F32, the LM head as a separate ``output.weight``.

The dense matrices are chosen to be exactly representable in Q8_0 (every block's scale a power of two), so the file
and the dense model hold bit-identical weights and any difference in their hidden states is the loader's. Arbitrary
values would not do: InvokeAI's Q8_0 kernel multiplies in float16, which rounds them by up to half a float16 step.

Every weight is random, biases included: transformers initialises biases to zero and norms to one, which would let a
dropped bias or a swapped norm pass unnoticed.
"""

from pathlib import Path
from unittest.mock import MagicMock

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

import invokeai.backend.model_manager.load.model_loaders.qwen_image as qwen_image_loaders
from invokeai.backend.model_manager.configs.qwen_vl_encoder import QwenVLEncoder_GGUF_Config
from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenVLEncoderGGUFLoader
from invokeai.backend.model_manager.taxonomy import SubModelType

HIDDEN_SIZE = 64
NUM_LAYERS = 2

# transformers 5 module path within a decoder layer -> llama.cpp's block component, as convert_hf_to_gguf.py names it.
_LLAMACPP_BLOCK_NAMES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}
_LLAMACPP_TOP_LEVEL_NAMES = {
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "model.language_model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
_LAYER_PREFIX = "model.language_model.layers."


def tiny_qwen_vl_config() -> Qwen2_5_VLConfig:
    """A Qwen2.5-VL config small enough to build on the CPU, with the real architecture's module tree.

    Grouped-query attention as in the 7B (4 heads over 2 KV heads here, 28 over 4 there). Widths stay multiples of
    32, Q8_0's block size, so every matrix quantizes.
    """
    return Qwen2_5_VLConfig(
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 128,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
            # Sums to head_dim / 2 (16 / 2), as the real [16, 24, 24] does for head_dim 128.
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": HIDDEN_SIZE,
            "fullatt_block_indexes": [0],
        },
    )


def llamacpp_name(name: str) -> str | None:
    """The llama.cpp tensor name for a language-model parameter, or None for the vision tower's."""
    if name in _LLAMACPP_TOP_LEVEL_NAMES:
        return _LLAMACPP_TOP_LEVEL_NAMES[name]
    if not name.startswith(_LAYER_PREFIX):
        return None
    layer, rest = name[len(_LAYER_PREFIX) :].split(".", 1)
    module, suffix = rest.rsplit(".", 1)
    return f"blk.{layer}.{_LLAMACPP_BLOCK_NAMES[module]}.{suffix}"


_Q8_0_BLOCK = 32


def _q8_0_exact(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    """Random values Q8_0 stores exactly: int8 codes over a power-of-two scale per 32-element block.

    Each block holds a code of +-127, so its scale is that power of two exactly and the codes come back unchanged.
    """
    rows, cols = shape
    codes = torch.randint(-127, 128, (rows, cols // _Q8_0_BLOCK, _Q8_0_BLOCK), generator=generator)
    codes[..., 0] = 127 * torch.where(torch.rand(codes.shape[:-1], generator=generator) < 0.5, -1, 1)
    scales = 2.0 ** -torch.randint(8, 10, (rows, cols // _Q8_0_BLOCK, 1), generator=generator)
    return (codes * scales).reshape(rows, cols).to(torch.float32)


def build_dense_model(seed: int = 0) -> Qwen2_5_VLForConditionalGeneration:
    generator = torch.Generator().manual_seed(seed)
    model = Qwen2_5_VLForConditionalGeneration(tiny_qwen_vl_config()).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 2:
                param.copy_(_q8_0_exact(param.shape, generator))
            elif name.endswith("norm.weight"):
                param.copy_(1 + 0.2 * torch.randn(param.shape, generator=generator))
            else:
                param.copy_(0.5 * torch.randn(param.shape, generator=generator))
    return model


def write_qwen_vl_gguf(
    path: Path, dense: Qwen2_5_VLForConditionalGeneration, *, omit: frozenset[str] = frozenset()
) -> None:
    """Write `dense`'s language model as a llama.cpp qwen2vl GGUF.

    `omit` leaves tensors out, which is what an incomplete or truncated file looks like.
    """
    text = dense.config.text_config
    writer = gguf.GGUFWriter(str(path), "qwen2vl")
    writer.add_uint32("qwen2vl.block_count", text.num_hidden_layers)
    writer.add_uint32("qwen2vl.embedding_length", text.hidden_size)
    for name, param in dense.named_parameters():
        gguf_name = llamacpp_name(name)
        if gguf_name is None or gguf_name in omit:
            continue
        data = param.detach().to(torch.float32).numpy()
        if data.ndim == 2:
            packed = quantize(data, gguf.GGMLQuantizationType.Q8_0)
            # The premise of every comparison against `dense` (see `_q8_0_exact`).
            assert np.array_equal(dequantize(packed, gguf.GGMLQuantizationType.Q8_0), data)
            # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
            writer.add_tensor(gguf_name, packed, raw_shape=packed.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(gguf_name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def make_loader(monkeypatch, *, ram_cache=None) -> QwenVLEncoderGGUFLoader:
    """The production loader, on the CPU at float32, building the tiny config in place of the vendored 7B one.

    float32 rather than the bfloat16 a CUDA device would pick: torch's CPU bf16 GEMM faults on part of the hosted
    Windows fleet (see tests/AGENTS.md).
    """
    monkeypatch.setattr(
        qwen_image_loaders, "load_bundled_qwen2_5_vl_config_dict", lambda: tiny_qwen_vl_config().to_dict()
    )
    monkeypatch.setattr(qwen_image_loaders.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(qwen_image_loaders.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    loader = object.__new__(QwenVLEncoderGGUFLoader)
    loader._logger = MagicMock()
    loader._ram_cache = MagicMock() if ram_cache is None else ram_cache
    return loader


def gguf_config(path: Path) -> QwenVLEncoder_GGUF_Config:
    return QwenVLEncoder_GGUF_Config.model_construct(path=str(path), name="tiny")


def load_text_encoder(monkeypatch, path: Path, *, ram_cache=None) -> Qwen2_5_VLForConditionalGeneration:
    model = make_loader(monkeypatch, ram_cache=ram_cache)._load_model(gguf_config(path), SubModelType.TextEncoder)
    assert isinstance(model, Qwen2_5_VLForConditionalGeneration)
    return model
