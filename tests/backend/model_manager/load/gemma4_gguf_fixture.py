"""Toy LTX-2 Gemma-4 encoder GGUFs, laid out like the community conversions of the release (elix3r, Rootport).

The tower under its Hugging Face names (``model.embed_tokens``, ``model.layers.N.*``, ``model.norm``) with its
Linear weights quantized and its norms and layer scalars in BF16; the embedding in BF16 (Rootport) or quantized
(elix3r). Beside it, as in the release: the text projection and the vision/audio stubs of the unified model, the
tokenizer files one byte per element (``I8`` in Rootport's builds, ``I16`` in elix3r's) and the unified
config.json as a metadata string, under the key each converter uses.
"""

import json
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import dequantize, quantize

from tests.fixtures.quantized_payloads import random_q4_k

# 256 wide so every Linear input fits K-quant super-blocks; the full-attention layer's 512-wide heads too.
TINY_GEMMA4 = {
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "intermediate_size": 512,
    "vocab_size": 64,
    "layer_types": ["sliding_attention", "full_attention"],
    "sliding_window": 8,
    "hidden_size_per_layer_input": 0,
    "vocab_size_per_layer_input": 64,
    "tie_word_embeddings": True,
}

# The two config keys the converters write; the first is elix3r's, the second Rootport's.
ELIX3R_CONFIG_KEY = "source.safetensors.gemma_config"
ROOTPORT_CONFIG_KEY = "config"


def tiny_gemma4(seed: int = 0):
    """The toy tower with its norms and layer scalars redrawn: constructed, each is one value throughout, so a
    norm loaded into its neighbour's slot would compute the same. The layer scalars are buffers, not
    parameters, and are read from the file all the same."""
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(seed)
    model = Gemma4TextModel(Gemma4TextConfig(**TINY_GEMMA4))
    scalars = [buffer for name, buffer in model.named_buffers() if name.endswith("layer_scalar")]
    with torch.no_grad():
        for tensor in [*model.parameters(), *scalars]:
            if tensor.ndim == 1:
                tensor.copy_(1 + 0.1 * torch.randn_like(tensor))
        # Halved so the sliding-attention scores stay moderate: at full size the softmax saturates, and a
        # query or key in the wrong slot changes nothing.
        for name, param in model.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                param.mul_(0.5)
    return model.eval()


def tokenizer_files() -> dict[str, bytes]:
    """A real, small tokenizer's two files: a WordLevel vocabulary with pad and EOS tokens.

    Some entries are not ASCII, as Gemma's ``▁`` word marker is not: their UTF-8 bytes run above 0x7F, which a
    signed ``I8`` asset stores as negative values.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["▁w1", "é", "日本", *(f"w{i}" for i in range(58))]
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{word: i + 3 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<pad>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
    }
    return {"tokenizer.json": tokenizer.to_str().encode(), "tokenizer_config.json": json.dumps(config).encode()}


def write_gemma4_gguf(
    path: Path,
    *,
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    quantized_embedding: bool = False,
    asset_type: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.I8,
    config_key: str | None = ROOTPORT_CONFIG_KEY,
    architecture: str = "ltxv",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Write the toy tower as a GGUF; return what each tower tensor reads as, keyed by its ``Gemma4TextModel`` name."""
    model = tiny_gemma4(seed)
    rng = np.random.default_rng(seed)
    writer = gguf.GGUFWriter(str(path), architecture)
    if config_key is not None:
        text_config = {**model.config.to_dict(), "model_type": "gemma4_text"}
        writer.add_string(
            config_key,
            json.dumps({"architectures": ["Gemma4UnifiedForConditionalGeneration"], "text_config": text_config}),
        )

    meant: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().float().numpy()
        packed = name.endswith("proj.weight") or (quantized_embedding and name == "embed_tokens.weight")
        meant[name] = _add(writer, f"model.{name}", data, qtype if packed else gguf.GGMLQuantizationType.BF16, rng)

    # Not the tower's: the connectors' projection, the unified model's other embedders, the packed files.
    _add(writer, "text_embedding_projection.video_aggregate_embed.weight", rng.standard_normal((8, 768)), qtype, rng)
    _add(
        writer,
        "text_embedding_projection.video_aggregate_embed.bias",
        rng.standard_normal(8),
        gguf.GGMLQuantizationType.BF16,
        rng,
    )
    _add(writer, "vision_model.patch_ln1.weight", rng.standard_normal(16), gguf.GGMLQuantizationType.BF16, rng)
    _add(writer, "multi_modal_projector.embedding_projection.weight", rng.standard_normal((256, 256)), qtype, rng)
    for name, content in {**tokenizer_files_as_assets(), "hf_asset__chat_template.jinja": b"{{ messages }}"}.items():
        _add_bytes(writer, name, content, asset_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant


def tokenizer_files_as_assets() -> dict[str, bytes]:
    files = tokenizer_files()
    return {
        "tokenizer_json": files["tokenizer.json"],
        "hf_asset__tokenizer_config.json": files["tokenizer_config.json"],
    }


def _add_bytes(writer: gguf.GGUFWriter, name: str, content: bytes, asset_type: gguf.GGMLQuantizationType) -> None:
    raw = np.frombuffer(content, dtype=np.uint8)
    if asset_type is gguf.GGMLQuantizationType.I8:
        writer.add_tensor(name, raw.view(np.int8), raw_dtype=asset_type)
    else:
        writer.add_tensor(name, raw.astype(np.int16), raw_dtype=asset_type)


def _add(
    writer: gguf.GGUFWriter, name: str, data: np.ndarray, stored_as: gguf.GGMLQuantizationType, rng: np.random.Generator
) -> torch.Tensor:
    data = np.asarray(data, dtype=np.float32)
    if stored_as is gguf.GGMLQuantizationType.Q4_K:
        raw = random_q4_k(data.shape, rng)
    else:
        raw = quantize(data, stored_as)
    # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
    writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
    return torch.from_numpy(dequantize(raw, stored_as).astype(np.float32)).reshape(data.shape)
