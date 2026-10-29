"""Loading a llama.cpp UMT5 encoder GGUF for Wan 2.2 (``city96/umt5-xxl-encoder-gguf``).

Each test writes a real, tiny GGUF (see ``umt5_gguf_fixture``) and runs the production loader over
it, so the tensors arrive as the ``GGMLTensor`` wrappers the cache dequantizes at runtime.
"""

from pathlib import Path
from unittest.mock import MagicMock

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_GGUF_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.wan import WanT5EncoderGGUFLoader
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.util.devices import TorchDevice
from tests.backend.model_manager.load.umt5_gguf_fixture import (
    D_MODEL,
    EOS_ID,
    PAD_ID,
    PIECE_IDS,
    UNK_ID,
    VOCAB,
    write_tiny_umt5_gguf,
)


def _loader(ram_cache=None) -> WanT5EncoderGGUFLoader:
    loader = object.__new__(WanT5EncoderGGUFLoader)
    loader._ram_cache = MagicMock() if ram_cache is None else ram_cache
    return loader


def _config(path: Path) -> WanT5Encoder_GGUF_Config:
    return WanT5Encoder_GGUF_Config.model_construct(path=str(path), name="tiny umt5")


@pytest.fixture
def float32_on_cpu(monkeypatch):
    """Load in float32: CPU bfloat16 matmuls fault on part of the CI fleet (see tests/AGENTS.md)."""
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda _device=None: torch.float32)


@pytest.fixture
def written(tmp_path) -> tuple[Path, torch.nn.Module]:
    path = tmp_path / "umt5-tiny-Q8_0.gguf"
    return path, write_tiny_umt5_gguf(path)


def test_a_forward_matches_the_dense_model_built_from_the_same_weights(written, float32_on_cpu) -> None:
    """The acceptance test. It fails for each of the ways a UMT5 load goes silently wrong: a block
    running on block 0's relative attention bias (T5 v1.1's scheme), a key mapped to a same-shaped
    neighbour (q/k, gate/up), or a feed-forward that casts its activations to the packed ``wo``'s
    uint8 storage dtype. Run through the cache's GGUF-aware layers, with padding in the mask as the
    Wan node pads every prompt to 512 tokens."""
    path, reference = written
    ram_cache = MagicMock()
    model = _loader(ram_cache)._load_model(_config(path), SubModelType.TextEncoder).eval()
    apply_custom_layers_to_model(model)

    input_ids = torch.tensor([[5, 17, 9, 30, 6, EOS_ID, PAD_ID, PAD_ID]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        actual = model(input_ids, attention_mask).last_hidden_state
        expected = reference(input_ids, attention_mask).last_hidden_state

    assert actual.dtype == torch.float32
    # The runtime's Q8_0 dequantizer scales in float16, gguf-py's reference in float32: about 2e-3 apart
    # here. Running every block on block 0's bias moves the output by about 0.8.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_the_linear_weights_stay_packed(written, float32_on_cpu) -> None:
    """The point of a Q8_0 encoder is its size. UMT5 declares ``_keep_in_fp32_modules = ["wo"]``, and
    feeding that to the unpack-at-load helper other GGUF loaders use would materialize every
    feed-forward output projection; every other test here would still pass."""
    path, _ = written
    model = _loader()._load_model(_config(path), SubModelType.TextEncoder)

    linears = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert len(linears) == 7 * len(model.encoder.block)
    for linear in linears:
        assert isinstance(linear.weight, GGMLTensor)
        assert linear.weight.dtype is torch.uint8
        assert linear.weight.shape == (linear.out_features, linear.in_features)


@pytest.mark.parametrize(
    "matrix_type",
    [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16],
    ids=lambda matrix_type: matrix_type.name,
)
def test_unquantized_matrices_are_unpacked_to_the_compute_dtype(tmp_path, float32_on_cpu, matrix_type) -> None:
    """city96 also ships F16 (10.6 GiB) and F32 (21.2 GiB) builds. Kept packed, the F32 one would stay
    that size in memory and every unquantized matrix would be converted on every forward; unpacked, they
    are the compute dtype's plain weights, and the encoding is the same."""
    path = tmp_path / f"umt5-tiny-{matrix_type.name}.gguf"
    reference = write_tiny_umt5_gguf(path, matrix_type=matrix_type)
    model = _loader()._load_model(_config(path), SubModelType.TextEncoder).eval()

    packed = [name for name, param in model.named_parameters() if isinstance(param, GGMLTensor)]
    assert packed == []
    assert {param.dtype for param in model.parameters()} == {torch.float32}
    input_ids = torch.tensor([[5, 17, 9, 30, 6, EOS_ID]])
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).last_hidden_state, reference(input_ids).last_hidden_state)


def test_a_device_without_bfloat16_computes_in_float32_not_float16(written, monkeypatch) -> None:
    """UMT5's feed-forward activations overflow float16, which transformers guards against by keeping
    ``wo`` in float32 -- not possible for a packed ``wo``, so the loader never picks float16."""
    path, _ = written
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda _device=None: torch.float16)

    model = _loader()._load_model(_config(path), SubModelType.TextEncoder)

    assert model.shared.weight.dtype == torch.float32
    assert {module.weight.compute_dtype for module in model.modules() if isinstance(module, torch.nn.Linear)} == {
        torch.float32
    }


def test_what_a_module_cannot_read_packed_is_unpacked_and_the_embedding_stays_tied(written, float32_on_cpu) -> None:
    """``nn.Embedding`` indexes its weight, so the token embedding and each block's own relative
    attention bias must be plain tensors; so are the norms. ``shared`` and ``encoder.embed_tokens``
    stay one tensor, and the embedding's decode is reserved with the cache first."""
    path, _ = written
    ram_cache = MagicMock()
    model = _loader(ram_cache)._load_model(_config(path), SubModelType.TextEncoder)

    assert model.encoder.embed_tokens.weight is model.shared.weight
    packed_outside_linears = [
        f"{module_name}.{name}"
        for module_name, module in model.named_modules()
        if not isinstance(module, torch.nn.Linear)
        for name, param in module.named_parameters(recurse=False)
        if isinstance(param, GGMLTensor)
    ]
    assert packed_outside_linears == []
    reserved = max(call.args[0] for call in ram_cache.make_room.call_args_list)
    assert reserved >= len(VOCAB) * D_MODEL * torch.float32.itemsize


def test_the_tokenizer_comes_from_the_gguf_metadata(written) -> None:
    """What the Wan node asks of it: runs of spaces collapse, ``</s>`` ends the prompt, padding uses
    ``<pad>``. A character outside the vocabulary becomes ``<unk>`` -- found by its token type, since
    the real file has no ``unknown_token_id`` -- and a sentinel typed into the prompt stays whole."""
    path, _ = written
    tokenizer = _loader()._load_model(_config(path), SubModelType.Tokenizer)
    words = [PIECE_IDS[piece] for piece in ("▁a", "▁cat", "▁on", "▁the", "▁mat")]

    assert tokenizer("a cat  on   the mat").input_ids == [*words, EOS_ID]
    assert tokenizer("a cat on the mat").input_ids == [*words, EOS_ID]
    padded = tokenizer(["a cat"], padding="max_length", max_length=6, truncation=True, return_tensors="pt")
    assert padded.input_ids.tolist() == [[PIECE_IDS["▁a"], PIECE_IDS["▁cat"], EOS_ID, PAD_ID, PAD_ID, PAD_ID]]
    assert padded.attention_mask.tolist() == [[1, 1, 1, 0, 0, 0]]
    assert tokenizer("a 猫").input_ids == [PIECE_IDS["▁a"], PIECE_IDS["▁"], UNK_ID, EOS_ID]
    # The `▁` prefix goes on every word run, also one that follows a sentinel without a space.
    assert tokenizer("<extra_id_0>a").input_ids == [PIECE_IDS["<extra_id_0>"], PIECE_IDS["▁a"], EOS_ID]
    # A prompt over the length is cut, and `</s>` still ends it.
    truncated = tokenizer(["a cat on the mat"], padding="max_length", max_length=4, truncation=True)
    assert truncated.input_ids == [[PIECE_IDS["▁a"], PIECE_IDS["▁cat"], PIECE_IDS["▁on"], EOS_ID]]
    assert truncated.attention_mask == [[1, 1, 1, 1]]


def test_a_gguf_without_tokenizer_fields_fails_naming_the_field(tmp_path) -> None:
    path = tmp_path / "no-tokenizer.gguf"
    write_tiny_umt5_gguf(path, with_tokenizer=False)

    with pytest.raises(ValueError, match="tokenizer.ggml"):
        _loader()._load_model(_config(path), SubModelType.Tokenizer)


def test_only_the_text_encoder_is_sized_by_the_file(written) -> None:
    """The cache makes room for this size before a load; the tokenizer reads only the header, and
    sizing it by the file would evict gigabytes to load a few megabytes."""
    path, _ = written
    loader = _loader()

    assert loader.get_size_fs(_config(path), path, SubModelType.Tokenizer) == 0
    assert loader.get_size_fs(_config(path), path, SubModelType.TextEncoder) == path.stat().st_size
