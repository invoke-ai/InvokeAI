"""Loading a llama.cpp GGUF of the Qwen2.5-VL language model as a text-only Qwen-Image encoder.

Nothing here mocks the GGUF layer: each test writes a real tiny qwen2vl GGUF from a seeded dense model (see
``qwen_vl_gguf_fixture``) and the production loader reads it, so the tensors arrive as the ``GGMLTensor`` wrappers the
loader has to handle and the dense model is an independent reference for what it must build.
"""

from pathlib import Path

import pytest
import torch

import invokeai.backend.model_manager.load.model_loaders.qwen_image as qwen_image_loaders
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import load_bundled_qwen2_5_vl_tokenizer
from tests.backend.model_manager.load.qwen_vl_gguf_fixture import (
    build_dense_model,
    gguf_config,
    load_text_encoder,
    make_loader,
    write_qwen_vl_gguf,
)


@pytest.fixture
def dense():
    return build_dense_model()


@pytest.fixture
def gguf_path(tmp_path: Path, dense) -> Path:
    path = tmp_path / "Qwen2.5-VL-tiny-Q8_0.gguf"
    write_qwen_vl_gguf(path, dense)
    return path


@pytest.fixture
def encoder(monkeypatch, gguf_path: Path):
    return load_text_encoder(monkeypatch, gguf_path)


def test_the_hidden_states_match_the_dense_model_the_file_was_written_from(encoder, dense) -> None:
    """The decisive check of the key mapping: a swapped or mis-named projection, a dropped bias, or a Q/K permutation
    that llama.cpp does not apply would each change every position's hidden state.

    Runs the call the Qwen-Image prompt node makes (`model.model(...)`), through the model cache's GGUF-aware layers,
    over the packed weights, across several positions so the rotary embedding takes part.
    """
    apply_custom_layers_to_model(encoder)
    input_ids = torch.tensor([[3, 17, 99, 250, 7, 400, 12, 5]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        actual = encoder.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        expected = dense.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def test_the_linear_weights_stay_packed(encoder) -> None:
    """The point of the format: the matrices reach the cache at their packed size and dequantize per forward."""
    for layer in encoder.model.language_model.layers:
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            assert isinstance(projection.weight, GGMLTensor)
            assert projection.weight.shape == (projection.out_features, projection.in_features)


def test_what_modules_read_outside_a_matmul_is_unpacked(encoder) -> None:
    """The embedding lookup and the RMSNorms read their weights directly, which packed storage cannot serve; the F32
    biases are no smaller packed and would otherwise be converted on every forward."""
    language_model = encoder.model.language_model
    unpacked = [language_model.embed_tokens.weight, language_model.norm.weight]
    for layer in language_model.layers:
        unpacked += [layer.input_layernorm.weight, layer.post_attention_layernorm.weight]
        unpacked += [layer.self_attn.q_proj.bias, layer.self_attn.k_proj.bias, layer.self_attn.v_proj.bias]

    assert not any(isinstance(tensor, GGMLTensor) for tensor in unpacked)


def test_the_lm_head_is_never_read(monkeypatch, gguf_path: Path) -> None:
    """The file's `output.weight` is as large as the embedding and nothing runs it: conditioning takes hidden states
    from the inner model. Skipping it at read time is what keeps its RAM from being paid at all."""
    read: list[str] = []
    real_loader = qwen_image_loaders.gguf_sd_loader

    def recording_loader(*args, **kwargs):
        sd = real_loader(*args, **kwargs)
        read.extend(sd)
        return sd

    monkeypatch.setattr(qwen_image_loaders, "gguf_sd_loader", recording_loader)

    encoder = load_text_encoder(monkeypatch, gguf_path)

    assert "token_embd.weight" in read
    assert "output.weight" not in read
    assert list(encoder.lm_head.parameters()) == []
    # A caller that runs the full causal-LM forward learns why, instead of computing logits from nothing.
    apply_custom_layers_to_model(encoder)
    with pytest.raises(RuntimeError, match="without its LM head"):
        encoder(input_ids=torch.tensor([[1, 2]]))


def test_the_vision_tower_is_dropped(encoder) -> None:
    """llama.cpp keeps it in the mmproj file, so there is nothing to load into it; kept, it would stay on the meta
    device and the load would be refused as incomplete."""
    assert isinstance(encoder.model.visual, torch.nn.Identity)


def test_an_incomplete_file_is_refused_at_load(monkeypatch, tmp_path: Path, dense) -> None:
    """Identification refuses such a file at install, but a record made before then, or a file truncated since,
    reaches the loader; the load must fail naming the gap rather than leave a weight on the meta device."""
    path = tmp_path / "incomplete.gguf"
    write_qwen_vl_gguf(path, dense, omit=frozenset({"blk.1.attn_k.bias"}))

    with pytest.raises(RuntimeError, match=r"missing 1 parameter.*layers\.1\.self_attn\.k_proj\.bias"):
        load_text_encoder(monkeypatch, path)


def test_only_the_text_encoder_is_sized_at_the_file(monkeypatch, gguf_path: Path) -> None:
    """The cache makes this much room before a load. The tokenizer is vendored, so sizing it at the file would evict
    cached models for a load that reads nothing from it."""
    loader = make_loader(monkeypatch)
    config = gguf_config(gguf_path)

    assert loader.get_size_fs(config, gguf_path, SubModelType.Tokenizer) == 0
    assert loader.get_size_fs(config, gguf_path, SubModelType.TextEncoder) == gguf_path.stat().st_size


def test_the_tokenizer_is_the_vendored_one(monkeypatch, gguf_path: Path) -> None:
    """The GGUF carries a llama.cpp tokenizer in its metadata; the encoder uses the release's, like every other
    Qwen2.5-VL encoder format, so a prompt tokenizes the same whichever is installed."""
    tokenizer = make_loader(monkeypatch)._load_model(gguf_config(gguf_path), SubModelType.Tokenizer)

    assert tokenizer is load_bundled_qwen2_5_vl_tokenizer()
