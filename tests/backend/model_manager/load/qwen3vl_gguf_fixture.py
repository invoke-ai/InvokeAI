"""A real, tiny llama.cpp Qwen3-VL encoder GGUF, and the loader run that turns it into a model.

Shared by the loader's own tests and by the consumer tests that have to answer "does what the loader
produces still satisfy me" — Krea-2 and Ideogram 4 drive the same module tree in different ways, and
neither should be tested against a hand-built stand-in for it.

The 2-D weights are genuinely Q8_0-quantized. That is the point of the feature and it is also the
harder path: an F32 fixture takes the ``TORCH_COMPATIBLE_QTYPES`` shortcut inside ``GGMLTensor`` and
never exercises packed storage at all.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_GGUF_Config
from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderGGUFLoader
from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType

FILL_VALUE = 0.5
# Q8_0 stores one fp16 scale per 32-element block, so a uniform tensor round-trips to within the
# scale's own representation error rather than exactly.
QUANT_TOLERANCE = 1e-3
HIDDEN_SIZE = 64


def tiny_qwen3vl_config(*, num_hidden_layers: int = 2):
    """A Qwen3-VL config small enough to build in a test, with the real architecture's module tree.

    Built locally rather than fetched, so the test needs neither network nor a warm HuggingFace
    cache. Only the dimensions shrink — and they stay multiples of 32 so the fixture can quantize to
    Q8_0, whose block size is 32. ``num_hidden_layers`` is a parameter because Ideogram 4 taps fixed
    layer indices up to 35 and so needs the real depth.
    """
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 128,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 64,
            "rope_scaling": {
                "mrope_interleaved": True,
                # Sums to head_dim / 2, as the real [24, 20, 20] does for head_dim 128.
                "mrope_section": [6, 5, 5],
                "rope_type": "default",
                "rope_theta": 5000000,
            },
        },
        vision_config={
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 128,
            "depth": 2,
            "num_heads": 2,
            "out_hidden_size": HIDDEN_SIZE,
            "deepstack_visual_indexes": [0],
        },
    )


def write_llamacpp_qwen3vl_gguf(path: Path, te_config) -> None:
    """Write the llama.cpp tensor set for ``te_config``'s language tower.

    Shapes are derived from the config rather than restated, so the fixture cannot drift out of
    agreement with the model the loader builds. No visual tensors: llama.cpp puts those in a
    separate ``mmproj-*.gguf``, which is the whole reason the loader drops that module. Matrices are
    Q8_0 and norms stay F32, which is how llama.cpp quantizes a real encoder.
    """
    import gguf
    from gguf.quants import quantize

    text = te_config.text_config
    q_out = text.num_attention_heads * text.head_dim
    kv_out = text.num_key_value_heads * text.head_dim
    hidden, ffn = text.hidden_size, text.intermediate_size

    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (text.vocab_size, hidden),
        "output_norm.weight": (hidden,),
    }
    for layer in range(text.num_hidden_layers):
        tensors.update(
            {
                f"blk.{layer}.attn_q.weight": (q_out, hidden),
                f"blk.{layer}.attn_k.weight": (kv_out, hidden),
                f"blk.{layer}.attn_v.weight": (kv_out, hidden),
                f"blk.{layer}.attn_output.weight": (hidden, q_out),
                f"blk.{layer}.attn_q_norm.weight": (text.head_dim,),
                f"blk.{layer}.attn_k_norm.weight": (text.head_dim,),
                f"blk.{layer}.ffn_gate.weight": (ffn, hidden),
                f"blk.{layer}.ffn_up.weight": (ffn, hidden),
                f"blk.{layer}.ffn_down.weight": (hidden, ffn),
                f"blk.{layer}.attn_norm.weight": (hidden,),
                f"blk.{layer}.ffn_norm.weight": (hidden,),
            }
        )

    writer = gguf.GGUFWriter(str(path), "qwen3vl")
    writer.add_uint32("qwen3vl.block_count", text.num_hidden_layers)
    writer.add_uint32("qwen3vl.embedding_length", hidden)
    for name, shape in tensors.items():
        data = np.full(shape, FILL_VALUE, dtype=np.float32)
        if len(shape) == 2:
            packed = quantize(data, gguf.GGMLQuantizationType.Q8_0)
            # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
            writer.add_tensor(name, packed, raw_shape=packed.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def load_tiny_gguf_encoder(monkeypatch, tmp_path: Path, *, num_hidden_layers: int = 2, ram_cache=None):
    """Run the production loader over a freshly written tiny GGUF, on the CPU and offline.

    Only the HuggingFace config lookup is substituted, for a locally built config of the same
    architecture — the module tree under test is the one transformers builds. Pass ``ram_cache`` to
    inspect what the loader reserved.
    """
    te_config = tiny_qwen3vl_config(num_hidden_layers=num_hidden_layers)
    gguf_path = tmp_path / "qwen3vl_tiny.gguf"
    write_llamacpp_qwen3vl_gguf(gguf_path, te_config)

    config = Qwen3VLEncoder_GGUF_Config.model_construct(
        path=str(gguf_path), variant=Qwen3VLVariantType.Qwen3VL_4B, name="tiny"
    )
    loader = object.__new__(Qwen3VLEncoderGGUFLoader)
    loader._logger = MagicMock()
    loader._ram_cache = MagicMock() if ram_cache is None else ram_cache

    monkeypatch.setattr(Qwen3VLEncoderGGUFLoader, "_load_te_config", lambda _self, _config: te_config)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.krea2.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    return loader._load_text_encoder(config)
