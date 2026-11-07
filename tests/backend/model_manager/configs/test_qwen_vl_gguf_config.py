"""Identification of llama.cpp Qwen2.5-VL GGUFs as text-only Qwen-Image encoders.

llama.cpp splits Qwen2.5-VL into the language model and a companion ``mmproj`` file holding the vision tower. The
language model's tensors are a plain llama.cpp decoder, the same names a Qwen3 or Llama GGUF uses, so only the
``general.architecture`` metadata identifies it, and the decisive tests go through ``ModelConfigFactory`` and count
the matches: identification iterates a *set* of config classes, so a file two configs accept is resolved by arbitrary
order.

Each file's header is real, written with gguf-py and read by the configs as they read any file: string metadata
through ``ModelOnDisk.metadata()``, the numeric context length through a reader of its own. Its tensors are
shape-faithful stand-ins without data (``_HeaderOnDisk``), because a 7B encoder's embedding alone is gigabytes. The
real unsloth header, tensor shapes included, is pinned by the stripped Q4_K_M fixture in
``tests/model_identification/stripped_models``, which the first test reads. The last tests write a whole encoder's
tensor set at a small width and identify it through ``ModelOnDisk``, which reads the shapes from the header.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import pytest
import torch

from invokeai.backend.model_manager.configs import qwen_vl_encoder
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.qwen_vl_encoder import QwenVLEncoder_GGUF_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.model_identification.stripped_model_on_disk import StrippedModelOnDisk

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

_UNSLOTH_Q4_K_M = (
    Path(__file__).parents[3]
    / "model_identification"
    / "stripped_models"
    / "d9c23cab-6868-4839-a884-125f23feb374"
    / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
)

_BLOCK_TENSORS = (
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_output.weight",
    "attn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_norm.weight",
)


class _HeaderOnDisk(ModelOnDisk):
    """A real GGUF file whose tensors are replaced by shape-faithful stand-ins without data."""

    def __init__(self, path: Path, tensors: dict[str, GGMLTensor]):
        super().__init__(path)
        self._tensors = tensors

    def load_state_dict(self, path: Path | None = None) -> dict[str, GGMLTensor]:
        return self._tensors


def _write_header(path: Path, *, architecture: str = "qwen2vl", context_length: int | None = 128000) -> Path:
    """The metadata the real unsloth file carries that identification reads."""
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_uint32(f"{architecture}.block_count", 28)
    writer.add_uint32(f"{architecture}.embedding_length", 3584)
    if context_length is not None:
        writer.add_uint32(f"{architecture}.context_length", context_length)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _encoder_tensors(
    *,
    hidden_size: int = 3584,
    vocab_size: int = 152064,
    num_layers: int = 28,
    omit: Iterable[str] = (),
    lm_head: bool = True,
) -> dict[str, GGMLTensor]:
    """A llama.cpp Qwen2-VL language model's tensor set. `omit` drops tensors, as an incomplete file does."""
    names = ["token_embd.weight", "output_norm.weight", *(["output.weight"] if lm_head else [])]
    names += [f"blk.{layer}.{name}" for layer in range(num_layers) for name in _BLOCK_TENSORS]
    omitted = set(omit)

    def stand_in(name: str) -> GGMLTensor:
        shape = (vocab_size, hidden_size) if name in ("token_embd.weight", "output.weight") else (hidden_size,)
        packed = torch.empty(0, dtype=torch.uint8, device="meta")
        return GGMLTensor(packed, gguf.GGMLQuantizationType.Q4_K, torch.Size(shape), torch.float32)

    return {name: stand_in(name) for name in names if name not in omitted}


def _encoder(tmp_path: Path, name: str = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", **kwargs: Any) -> _HeaderOnDisk:
    header = {key: kwargs.pop(key) for key in ("architecture", "context_length") if key in kwargs}
    return _HeaderOnDisk(_write_header(tmp_path / name, **header), _encoder_tensors(**kwargs))


def _write_mmproj(path: Path, projector_type: str) -> Path:
    """The companion vision tower, with the metadata the real `mmproj-F16.gguf` carries."""
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("general.type", "clip-vision")
    writer.add_string("clip.projector_type", projector_type)
    writer.add_uint32("clip.vision.block_count", 1)
    for name in ("v.blk.0.attn_q.weight", "v.blk.0.attn_q.bias", "v.blk.0.ln1.weight", "mm.0.weight"):
        writer.add_tensor(name, np.zeros((4,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _identify(mod: ModelOnDisk | Path, *, allow_unknown: bool = False) -> Any:
    return ModelConfigFactory.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS), allow_unknown=allow_unknown)


def _config_from(mod: ModelOnDisk) -> QwenVLEncoder_GGUF_Config:
    return QwenVLEncoder_GGUF_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_the_real_unsloth_header_identifies_unambiguously_as_a_qwen_vl_encoder() -> None:
    """One match: the text-only Qwen3 and the Qwen3-VL GGUF configs must both decline it, or which encoder type the
    file installs as would depend on set-iteration order."""
    result = _identify(StrippedModelOnDisk(_UNSLOTH_Q4_K_M))

    assert result.match_count == 1
    assert isinstance(result.config, QwenVLEncoder_GGUF_Config)
    assert isinstance(result.details["Qwen3Encoder_GGUF_Config"], NotAMatchError)
    assert isinstance(result.details["Qwen3VLEncoder_GGUF_Config"], NotAMatchError)


def test_a_qwen2_5_vl_header_identifies_unambiguously_as_a_qwen_vl_encoder(tmp_path: Path) -> None:
    result = _identify(_encoder(tmp_path))

    assert result.match_count == 1
    assert isinstance(result.config, QwenVLEncoder_GGUF_Config)


def test_qwen2_vl_is_refused_by_its_context_length(tmp_path: Path) -> None:
    """Qwen2-VL-7B shares Qwen2.5-VL-7B's architecture name, geometry and tensor set (checked against
    `bartowski/Qwen2-VL-7B-Instruct-GGUF`), so it would install and condition Qwen-Image with the wrong model. Its
    32768-token context is the difference. Through the factory at the default `allow_unknown`, so the refusal is
    final rather than an Unknown model."""
    mod = _encoder(tmp_path, "Qwen2-VL-7B-Instruct-Q4_K_M.gguf", context_length=32768)

    result = _identify(mod, allow_unknown=True)

    assert result.config is None
    [reason] = result.invalid_matches
    assert "Qwen2-VL, not Qwen2.5-VL, which Qwen-Image uses" in str(reason)


@pytest.mark.parametrize("context_length", [None, 32000, 262144], ids=["absent", "trimmed", "extended"])
def test_a_context_length_other_than_qwen2_vls_is_accepted(tmp_path: Path, context_length: int | None) -> None:
    """Only Qwen2-VL's value is refused: a conversion may leave the field out, and a finetune may change it."""
    assert _config_from(_encoder(tmp_path, context_length=context_length)).type.value == "qwen_vl_encoder"


def test_a_file_without_the_lm_head_is_still_complete(tmp_path: Path) -> None:
    """The loader never reads `output.weight`, so a conversion that leaves it out loses nothing."""
    assert _config_from(_encoder(tmp_path, lm_head=False)).type.value == "qwen_vl_encoder"


@pytest.mark.parametrize("projector_type", ["qwen2.5vl_merger", "qwen2vl_merger"])
def test_the_mmproj_vision_tower_is_refused_with_its_reason(tmp_path: Path, projector_type: str) -> None:
    """The GGUF repos ship it beside the language model. Run at the default `allow_unknown`, as an install does: a
    plain non-match would register it as an Unknown model, report success, and leave the reason in the server log."""
    result = _identify(_write_mmproj(tmp_path / "mmproj-F16.gguf", projector_type), allow_unknown=True)

    assert result.config is None
    [reason] = result.invalid_matches
    assert "Qwen2-VL / Qwen2.5-VL vision tower ('mmproj')" in str(reason)
    assert "Qwen-Image Edit with reference images" in str(reason)
    assert "fp8 scaled" in str(reason)


def test_the_qwen3_vl_mmproj_gets_only_its_own_refusal(tmp_path: Path) -> None:
    """Qwen3-VL's tower has its own refusal, which names the Krea-2 encoder to install instead; a second one pointing
    at Qwen-Image would contradict it."""
    result = _identify(_write_mmproj(tmp_path / "mmproj-Qwen3VL-4B.gguf", "qwen3vl_merger"), allow_unknown=True)

    [reason] = result.invalid_matches
    assert "Qwen3-VL visual tower" in str(reason)


@pytest.mark.parametrize(
    ("omit", "reason"),
    [
        # A later part of a multi-part quant: complete metadata, only some of the blocks.
        ([f"blk.27.{name}" for name in _BLOCK_TENSORS], "incomplete"),
        (["blk.5.attn_q.bias"], "incomplete"),
        (["token_embd.weight"], "must contain 'token_embd.weight'"),
    ],
    ids=["missing-last-block", "missing-q-bias", "missing-embedding"],
)
def test_an_incomplete_file_is_refused_at_install(tmp_path: Path, omit: list[str], reason: str) -> None:
    """Through the factory at the default `allow_unknown`, because that decides what the user sees: the refusal has
    to be final, or the file installs as an Unknown model and fails nowhere the user looks."""
    result = _identify(_encoder(tmp_path, omit=omit), allow_unknown=True)

    assert result.config is None
    assert any(reason in str(invalid) for invalid in result.invalid_matches)


@pytest.mark.parametrize(
    "geometry",
    [
        {"hidden_size": 2048},
        {"hidden_size": 5120},
        {"num_layers": 29},
        # Qwen2-7B's text vocabulary, and a finetune that added tokens: neither embedding would load.
        {"vocab_size": 151936},
        {"vocab_size": 152128},
    ],
    ids=["3b-width", "32b-width", "extra-layer", "qwen2-vocab", "resized-vocab"],
)
def test_other_qwen2_vl_geometries_are_refused(tmp_path: Path, geometry: dict[str, int]) -> None:
    """Qwen-Image's transformer reads a 3584-wide prompt, and the loader builds the 7B's 28 layers and vocabulary."""
    with pytest.raises(InvalidMatchError, match="7B encoder"):
        _config_from(_encoder(tmp_path, **geometry))


def test_the_text_only_qwen2_5_7b_is_not_a_qwen_vl_encoder(tmp_path: Path) -> None:
    """Qwen2.5-7B-Instruct has the same geometry and tensor names but other weights; llama.cpp calls it "qwen2"."""
    mod = _encoder(tmp_path, "Qwen2.5-7B-Instruct-Q4_K_M.gguf", architecture="qwen2")

    with pytest.raises(NotAMatchError, match="'qwen2'"):
        _config_from(mod)
    assert _identify(mod).config is None


def test_a_safetensors_file_is_not_a_gguf_encoder(tmp_path: Path) -> None:
    with pytest.raises(NotAMatchError, match="expected a .gguf file"):
        _config_from(_encoder(tmp_path, "encoder.safetensors"))


def test_a_file_that_is_not_a_gguf_is_reported_as_a_non_match(tmp_path: Path) -> None:
    """Identification probes every config in turn, so a corrupt file must not abort the sweep."""
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(b"not a gguf at all")

    with pytest.raises(NotAMatchError, match="general.architecture"):
        _config_from(ModelOnDisk(path))


# A real file read as identification reads it: `ModelOnDisk` takes the tensors' names, shapes and quantization types
# from the header and none of their data.

_SMALL_WIDTH = 32  # One Q8_0 block per row.
_SMALL_VOCAB_SIZE = 64


def _write_small_encoder(path: Path) -> Path:
    """A Qwen2.5-VL language model's complete tensor set at a small width: an unsloth header otherwise, 28 blocks.

    The embedding is Q8_0, whose packed bytes have another shape than the matrix they decode to, so a probe that
    read the storage shape would see the wrong width.
    """
    writer = gguf.GGUFWriter(str(path), "qwen2vl")
    writer.add_uint32("qwen2vl.block_count", 28)
    writer.add_uint32("qwen2vl.embedding_length", _SMALL_WIDTH)
    writer.add_uint32("qwen2vl.context_length", 128000)
    embedding = gguf.quants.quantize(
        np.zeros((_SMALL_VOCAB_SIZE, _SMALL_WIDTH), dtype=np.float32), gguf.GGMLQuantizationType.Q8_0
    )
    writer.add_tensor(
        "token_embd.weight", embedding, raw_shape=embedding.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0
    )
    writer.add_tensor("output_norm.weight", np.zeros(_SMALL_WIDTH, dtype=np.float32))
    for layer in range(28):
        for name in _BLOCK_TENSORS:
            writer.add_tensor(f"blk.{layer}.{name}", np.zeros(_SMALL_WIDTH, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_a_real_file_is_measured_by_its_logical_shapes_read_from_the_header(tmp_path: Path) -> None:
    mod = ModelOnDisk(_write_small_encoder(tmp_path / "Qwen2.5-VL-small-Q8_0.gguf"))

    result = _identify(mod, allow_unknown=True)

    assert result.config is None
    [reason] = result.invalid_matches
    assert f"{_SMALL_WIDTH} wide with 28 layers and a {_SMALL_VOCAB_SIZE}-token vocabulary" in str(reason)
    assert all(tensor.quantized_data.is_meta for tensor in mod.load_state_dict().values())


def test_a_real_file_of_the_expected_geometry_identifies_from_its_header_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 7B's own embedding is gigabytes even as Q4_K, so the expected geometry is scaled down to the file's."""
    monkeypatch.setattr(qwen_vl_encoder, "_QWEN_IMAGE_ENCODER_HIDDEN_SIZE", _SMALL_WIDTH)
    monkeypatch.setattr(qwen_vl_encoder, "_QWEN_IMAGE_ENCODER_VOCAB_SIZE", _SMALL_VOCAB_SIZE)

    result = _identify(ModelOnDisk(_write_small_encoder(tmp_path / "Qwen2.5-VL-small-Q8_0.gguf")))

    assert result.match_count == 1
    assert isinstance(result.config, QwenVLEncoder_GGUF_Config)
