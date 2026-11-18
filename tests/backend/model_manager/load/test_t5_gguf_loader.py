"""The FLUX/SD3 T5 GGUF loader (``T5EncoderGGUFModel``).

Its llama.cpp key mapping and feed-forward patch are shared with the Wan UMT5 loader and tested in
``tests/backend/model_manager/util/test_llamacpp_t5.py``. Its config inference is shared with the
single-file loader and tested in ``tests/backend/t5/test_t5_encoder.py``.

Forwards run in float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
"""

from pathlib import Path
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch
from gguf.quants import dequantize, quantize
from transformers import T5Config, T5EncoderModel

from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_GGUF_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.flux import T5EncoderGGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.util.devices import TorchDevice
from tests.backend.model_manager.load.umt5_gguf_fixture import write_tiny_umt5_gguf


def test_a_umt5_gguf_recorded_as_a_flux_t5_encoder_is_refused_with_the_way_out(tmp_path) -> None:
    """Before identification knew UMT5, city96's UMT5 GGUFs installed as FLUX/SD3 T5 encoders. Loaded
    as T5 v1.1, such a file runs every block on block 0's relative attention bias and encodes garbage
    without an error, so the old record has to be refused, pointing at the Model Manager action that
    turns it into a Wan T5 encoder."""
    path = tmp_path / "umt5-xxl-encoder-Q8_0.gguf"
    write_tiny_umt5_gguf(path)
    loader = object.__new__(T5EncoderGGUFModel)
    loader._ram_cache = MagicMock()
    loader._torch_device = torch.device("cpu")
    config = T5Encoder_GGUF_Config.model_construct(path=str(path), name="umt5-xxl-encoder-Q8_0")

    with pytest.raises(ValueError, match="UMT5.*Re-identify model"):
        loader._load_model(config, SubModelType.TextEncoder2)


# A tiny T5 v1.1 in llama.cpp naming. T5 v1.1 learns one relative attention bias, in block 0 (the UMT5 fixture
# above has one per block and is refused). Q8_0 and F16 matrices, F32 norms and bias, as city96's builds mix them.
D_MODEL = 64  # Q8_0 packs blocks of 32 along a row.
NUM_LAYERS = 2
VOCAB_SIZE = 32


def _llamacpp_t5_names() -> dict[str, str]:
    names = {"shared.weight": "token_embd.weight", "encoder.final_layer_norm.weight": "enc.output_norm.weight"}
    for block in range(NUM_LAYERS):
        attn, ffn = f"encoder.block.{block}.layer.0", f"encoder.block.{block}.layer.1"
        names.update(
            {
                f"{attn}.SelfAttention.q.weight": f"enc.blk.{block}.attn_q.weight",
                f"{attn}.SelfAttention.k.weight": f"enc.blk.{block}.attn_k.weight",
                f"{attn}.SelfAttention.v.weight": f"enc.blk.{block}.attn_v.weight",
                f"{attn}.SelfAttention.o.weight": f"enc.blk.{block}.attn_o.weight",
                f"{attn}.layer_norm.weight": f"enc.blk.{block}.attn_norm.weight",
                f"{ffn}.DenseReluDense.wi_0.weight": f"enc.blk.{block}.ffn_gate.weight",
                f"{ffn}.DenseReluDense.wi_1.weight": f"enc.blk.{block}.ffn_up.weight",
                f"{ffn}.DenseReluDense.wo.weight": f"enc.blk.{block}.ffn_down.weight",
                f"{ffn}.layer_norm.weight": f"enc.blk.{block}.ffn_norm.weight",
            }
        )
    names["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = "enc.blk.0.attn_rel_b.weight"
    return names


def _write_tiny_t5_gguf(path: Path) -> tuple[T5EncoderModel, dict[str, np.ndarray]]:
    """Write the GGUF; return the dense float32 model it encodes and the packed bytes written per llama.cpp name.

    The reference weights are the stored ones run back through gguf-py's own dequantizer, so nothing the tests
    compare against comes from the loader.
    """
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_kv=32,
        d_ff=128,
        num_layers=NUM_LAYERS,
        num_heads=2,
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        dense_act_fn="gelu_new",
        tie_word_embeddings=False,
        use_cache=False,
    )
    reference = T5EncoderModel(config).eval()
    writer = gguf.GGUFWriter(str(path), "t5encoder")
    writer.add_uint32("t5encoder.block_count", NUM_LAYERS)
    writer.add_uint32("t5encoder.embedding_length", D_MODEL)
    packed_bytes: dict[str, np.ndarray] = {}
    state_dict = reference.state_dict()
    with torch.no_grad():
        for hf_name, gguf_name in _llamacpp_t5_names().items():
            tensor = state_dict[hf_name]
            data = tensor.numpy().astype(np.float32)
            if data.ndim == 2 and "relative_attention_bias" not in hf_name:
                qtype = gguf.GGMLQuantizationType.F16 if "wi_" in hf_name else gguf.GGMLQuantizationType.Q8_0
                packed = quantize(data, qtype)
                writer.add_tensor(gguf_name, packed, raw_shape=packed.shape, raw_dtype=qtype)
                packed_bytes[gguf_name] = packed
                tensor.copy_(torch.from_numpy(dequantize(packed, qtype).reshape(data.shape)))
            else:
                writer.add_tensor(gguf_name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return reference, packed_bytes


def _t5_loader(device: str = "cpu") -> T5EncoderGGUFModel:
    loader = object.__new__(T5EncoderGGUFModel)
    loader._ram_cache = MagicMock()
    loader._torch_device = torch.device(device)
    return loader


def _gguf_config(path: Path, cpu_only: bool | None = None) -> T5Encoder_GGUF_Config:
    return T5Encoder_GGUF_Config.model_construct(path=str(path), name="tiny t5", cpu_only=cpu_only)


def _assert_float32_throughout(model: T5EncoderModel, reference: T5EncoderModel, packed: dict[str, np.ndarray]) -> None:
    ggml_params = {n: p for n, p in model.named_parameters() if isinstance(p, GGMLTensor)}
    # The Linear weights and norms stay packed; the token embedding and relative bias are dequantized.
    assert ggml_params
    assert {p.compute_dtype for p in ggml_params.values()} == {torch.float32}
    assert model.shared.weight.dtype == torch.float32
    assert model.encoder.embed_tokens.weight is model.shared.weight
    rel_bias = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
    assert not isinstance(rel_bias, GGMLTensor) and rel_bias.dtype == torch.float32
    q_weight = model.encoder.block[0].layer[0].SelfAttention.q.weight
    assert isinstance(q_weight, GGMLTensor)
    assert np.array_equal(q_weight.quantized_data.numpy(), packed["enc.blk.0.attn_q.weight"])

    model.eval()  # `ModelLoader.load_model` does this for every load
    apply_custom_layers_to_model(model)
    input_ids = torch.tensor([[3, 7, 1, 30, 12, 0]])
    with torch.no_grad():
        actual = model(input_ids=input_ids).last_hidden_state
        expected = reference(input_ids=input_ids).last_hidden_state
    assert actual.dtype == torch.float32
    # The runtime's Q8_0 dequantizer scales in float16, gguf-py's reference in float32.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(model.shared.weight, reference.shared.weight, rtol=1e-2, atol=1e-2)


def test_a_cpu_load_dequantizes_and_computes_in_float32(tmp_path: Path) -> None:
    path = tmp_path / "t5-tiny.gguf"
    reference, packed = _write_tiny_t5_gguf(path)
    loader = _t5_loader()
    config = _gguf_config(path)

    assert loader._cache_key_variant(config, SubModelType.TextEncoder2) == "float32"
    _assert_float32_throughout(loader._load_model(config, SubModelType.TextEncoder2), reference, packed)


def test_a_cpu_only_t5_on_a_gpu_box_computes_in_float32(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The dtype follows the execution device `cpu_only` selects, not the loader's own `_torch_device`. The device's
    # dtype choice is patched to float16 so a GPU decision would show.
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device=None: torch.float16))
    path = tmp_path / "t5-tiny.gguf"
    reference, packed = _write_tiny_t5_gguf(path)
    loader = _t5_loader("cuda:0")
    config = _gguf_config(path, cpu_only=True)

    assert loader._cache_key_variant(config, SubModelType.TextEncoder2) == "float32"
    _assert_float32_throughout(loader._load_model(config, SubModelType.TextEncoder2), reference, packed)


def test_a_gpu_box_dequantizes_to_the_dtype_the_device_picks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device=None: torch.float16))
    path = tmp_path / "t5-tiny.gguf"
    _write_tiny_t5_gguf(path)
    loader = _t5_loader("cuda:0")
    config = _gguf_config(path)

    assert loader._cache_key_variant(config, SubModelType.TextEncoder2) == "float16"
    assert loader._cache_key_variant(config, SubModelType.Tokenizer2) == ""
    model = loader._load_model(config, SubModelType.TextEncoder2)
    assert model.shared.weight.dtype == torch.float16
    assert {p.compute_dtype for p in model.parameters() if isinstance(p, GGMLTensor)} == {torch.float16}
