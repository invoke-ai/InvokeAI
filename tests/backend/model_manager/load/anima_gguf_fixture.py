"""A real, tiny Anima transformer GGUF, written the way the published conversions pack theirs.

Read from the headers of vanes430's turbo v1.1, Bedovyy's preview, JusteLeo's preview2 and n-Arno's all-in-one
builds: the keys are Anima's own, under no prefix, `net.` or `model.diffusion_model.`; the Linear weights and the
LLM adapter's token embedding are quantized; norms and biases are F32 -- except in JusteLeo's build, which
quantizes the 1-D norms to Q8_0 and stores the biases and `t_embedding_norm` as BF16; the timestep MLP and the
output head stay BF16 in vanes430's build, and every build keeps the patch embedding unquantized, since no GGML
block divides its 68-wide rows.

Every width is shrunk to a multiple of 32, the Q8_0 block, so the forward runs in milliseconds. The module reads
its geometry from two places -- `ANIMA_TRANSFORMER_CONFIG` and the `LLMAdapter` it builds itself --
and `install_tiny_anima_geometry` patches both, for the model under test and for its dense reference alike.
"""

import functools
from collections.abc import Mapping
from pathlib import Path

import gguf
import pytest
import torch
from gguf.quants import dequantize, quantize

from invokeai.backend.anima import anima_transformer, checkpoint_layout
from invokeai.backend.anima.anima_transformer import AnimaTransformer, LLMAdapter

TINY_TRANSFORMER_CONFIG = {
    **checkpoint_layout.ANIMA_TRANSFORMER_CONFIG,
    "model_channels": 64,
    "num_heads": 2,
    "crossattn_emb_channels": 64,
    "adaln_lora_dim": 32,
}
# Its width is the DiT's cross-attention context width, as 1024 is in the release.
TINY_LLM_ADAPTER = {"vocab_size": 64, "dim": 64, "num_layers": 1, "num_heads": 2}

# The patch embedding and two of the small Linears vanes430's build keeps unquantized.
_BF16 = ("x_embedder.proj.1.weight", "t_embedder.1.linear_1.weight", "final_layer.linear.weight")
# 1-D yet Q8_0, as JusteLeo's preview2 stores every norm.
_Q8_0_1D = ("llm_adapter.norm.weight", "blocks.0.self_attn.q_norm.weight")


def install_tiny_anima_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_layout, "ANIMA_TRANSFORMER_CONFIG", TINY_TRANSFORMER_CONFIG)
    monkeypatch.setattr(anima_transformer, "LLMAdapter", functools.partial(LLMAdapter, **TINY_LLM_ADAPTER))


def tiny_anima_transformer(num_blocks: int) -> AnimaTransformer:
    """A dense transformer at the tiny geometry, which the caller must have installed."""
    model = AnimaTransformer(**{**TINY_TRANSFORMER_CONFIG, "num_blocks": num_blocks})
    assert model.llm_adapter.embed.weight.shape == (64, 64), "install_tiny_anima_geometry() first"
    return model


def write_anima_gguf(
    path: Path,
    *,
    prefix: str = "",
    num_blocks: int = 2,
    extra: Mapping[str, torch.Tensor] | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Write a randomly initialised tiny transformer under `prefix`; return what each tensor dequantizes to.

    `extra` tensors are written as they are, F32 or F16, beside the transformer and under their own names: an
    all-in-one's text encoder and VAE, or the `pos_embedder` buffers a depth-expanded export serializes.
    """
    torch.manual_seed(seed)
    model = tiny_anima_transformer(num_blocks)
    # The norms initialise to ones, which any scale reproduces; a dropped or misread norm must show.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 1 and "norm" in name:
                param.copy_(1.0 + 0.2 * torch.randn_like(param))

    writer = gguf.GGUFWriter(str(path), "cosmos")
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().to(torch.float32).numpy()
        if name in _BF16:
            stored_as = gguf.GGMLQuantizationType.BF16
        elif data.ndim == 2 or name in _Q8_0_1D:
            stored_as = gguf.GGMLQuantizationType.Q8_0
        else:
            stored_as = gguf.GGMLQuantizationType.F32
        if stored_as is gguf.GGMLQuantizationType.F32:
            writer.add_tensor(prefix + name, data)
            meant[name] = torch.from_numpy(data.copy())
        else:
            raw = quantize(data, stored_as)
            # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
            writer.add_tensor(prefix + name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
            meant[name] = torch.from_numpy(dequantize(raw, stored_as)).reshape(tensor.shape)
    for name, tensor in (extra or {}).items():
        writer.add_tensor(name, tensor.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant
