"""Identifying city96's UMT5-XXL GGUF (Wan 2.2) apart from the T5 v1.1 GGUFs FLUX and SD3 use.

Both are llama.cpp ``t5encoder`` files with the same tensor names; a relative attention bias in every
block makes one UMT5, and its 256k vocabulary keeps it out of the T5 v1.1 config. Picking wrong is
silent: the UMT5 file installs as a FLUX T5 encoder and encodes garbage. Nothing orders two matching
configs of these types (both sort alike and ``Config_Base.CONFIG_CLASSES`` is a set), so the decisive
tests go through ``ModelConfigFactory`` and assert ``match_count``.

The fixtures are the stripped headers of the real files in ``tests/model_identification``; the
broken variants are those headers with tensors or metadata changed.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import pytest
from gguf.quants import quantize

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_GGUF_Config
from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_GGUF_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from tests.model_identification.stripped_model_on_disk import StrippedModelOnDisk

_STRIPPED = Path(__file__).parents[3] / "model_identification" / "stripped_models"
UMT5_Q8_0 = _STRIPPED / "f831a21c-4a99-4f89-ab7b-de4eb5e0708f" / "umt5-xxl-encoder-Q8_0.gguf"
T5_V1_1_Q3_K_S = _STRIPPED / "98e3f524-8018-4c25-8920-6db3951415d8" / "t5-v1_1-xxl-encoder-Q3_K_S.gguf"

_OVERRIDE_FIELDS: dict[str, Any] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/encoder.gguf",
    "file_size": 1000,
    "name": "encoder",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _identify(path: Path, *, allow_unknown: bool = False):
    return ModelConfigFactory.from_model_on_disk(
        StrippedModelOnDisk(path), dict(_OVERRIDE_FIELDS), allow_unknown=allow_unknown
    )


def _altered(tmp_path: Path, source: Path, edit: Callable[[dict[str, Any], dict[str, Any]], None]) -> Path:
    """A copy of a stripped header after ``edit(tensors, metadata)``."""
    contents = json.loads(source.read_text())
    metadata = contents.pop(StrippedModelOnDisk.METADATA_KEY)
    edit(contents, metadata)
    path = tmp_path / source.name
    path.write_text(json.dumps({**contents, StrippedModelOnDisk.METADATA_KEY: metadata}))
    return path


def _drop_tensors(prefix: str) -> Callable[[dict[str, Any], dict[str, Any]], None]:
    def edit(tensors: dict[str, Any], _metadata: dict[str, Any]) -> None:
        for key in [key for key in tensors if key.startswith(prefix)]:
            del tensors[key]

    return edit


def _drop_metadata(*keys: str) -> Callable[[dict[str, Any], dict[str, Any]], None]:
    def edit(_tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
        for key in keys:
            del metadata[key]

    return edit


def _refusals(result) -> list[str]:
    assert result.config is None
    return [str(reason) for reason in result.invalid_matches]


def test_the_umt5_gguf_identifies_only_as_a_wan_t5_encoder() -> None:
    result = _identify(UMT5_Q8_0)

    assert result.match_count == 1
    assert isinstance(result.config, WanT5Encoder_GGUF_Config)


def test_the_t5_v1_1_gguf_still_identifies_only_as_a_flux_t5_encoder() -> None:
    """The other direction: the Wan config must not take FLUX's files."""
    result = _identify(T5_V1_1_Q3_K_S)

    assert result.match_count == 1
    assert isinstance(result.config, T5Encoder_GGUF_Config)


def test_an_mt5_gguf_is_not_taken_for_a_flux_t5_encoder(tmp_path: Path) -> None:
    """mT5 has T5 v1.1's block-0-only bias in the same layout, but a 250112-token vocabulary the
    FLUX/SD3 loader's bundled tokenizer does not encode."""

    def mt5_vocabulary(tensors: dict[str, Any], _metadata: dict[str, Any]) -> None:
        tensors["token_embd.weight"]["tensor_shape"][0] = 250112

    result = _identify(_altered(tmp_path, T5_V1_1_Q3_K_S, mt5_vocabulary))

    assert result.match_count == 0
    assert "vocabulary of 250112" in str(result.details[T5Encoder_GGUF_Config.__name__])


def test_an_extended_vocabulary_t5_v1_1_gguf_is_still_a_flux_t5_encoder(tmp_path: Path) -> None:
    """Derivatives such as T5XXL-Unchained grow T5 v1.1's vocabulary to ~69k tokens and run on the
    bundled 32128-token tokenizer; only a multilingual vocabulary means another tokenizer."""

    def extended_vocabulary(tensors: dict[str, Any], _metadata: dict[str, Any]) -> None:
        tensors["token_embd.weight"]["tensor_shape"][0] = 69328

    result = _identify(_altered(tmp_path, T5_V1_1_Q3_K_S, extended_vocabulary))

    assert result.match_count == 1
    assert isinstance(result.config, T5Encoder_GGUF_Config)


# Refused at install, at the default `allow_unknown`: a plain non-match would register the file as an
# Unknown model, with the reason only in the server log.


@pytest.mark.parametrize(
    ("dropped", "named"),
    [("enc.blk.23.", "enc.blk.23."), ("token_embd.", "token_embd.weight")],
    ids=["short_of_its_block_count", "no_token_embedding"],
)
def test_an_incomplete_umt5_gguf_is_refused_naming_what_is_missing(tmp_path: Path, dropped: str, named: str) -> None:
    refusals = _refusals(_identify(_altered(tmp_path, UMT5_Q8_0, _drop_tensors(dropped)), allow_unknown=True))

    assert any("incomplete" in reason and named in reason for reason in refusals)


@pytest.mark.parametrize(
    "dropped",
    [("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id"), ("tokenizer.ggml.add_eos_token",)],
    ids=["no_tokenizer", "partial_tokenizer"],
)
def test_a_umt5_gguf_without_its_tokenizer_fields_is_refused(tmp_path: Path, dropped) -> None:
    """A GGUF ships no tokenizer files beside it, so the loader builds the tokenizer from these."""
    refusals = _refusals(_identify(_altered(tmp_path, UMT5_Q8_0, _drop_metadata(*dropped)), allow_unknown=True))

    assert any("tokenizer" in reason and dropped[-1] in reason for reason in refusals)


def test_a_umt5_gguf_declaring_another_tokenizer_is_refused(tmp_path: Path) -> None:
    def llama_tokenizer(_tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
        metadata["tokenizer.ggml.model"] = "llama"

    refusals = _refusals(_identify(_altered(tmp_path, UMT5_Q8_0, llama_tokenizer), allow_unknown=True))

    assert any("'llama' tokenizer" in reason for reason in refusals)


def test_an_encoder_decoder_umt5_gguf_is_refused_as_such(tmp_path: Path) -> None:
    """llama.cpp's full ``t5`` conversion keeps its sizes under ``t5.*``; calling it incomplete for a
    missing ``t5encoder.block_count`` would send the user looking for a damaged download."""

    def encoder_decoder(_tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
        metadata["general.architecture"] = "t5"
        metadata["t5.block_count"] = metadata.pop("t5encoder.block_count")

    refusals = _refusals(_identify(_altered(tmp_path, UMT5_Q8_0, encoder_decoder), allow_unknown=True))

    assert any("'t5'" in reason and "encoder-only" in reason for reason in refusals)
    assert not any("incomplete" in reason for reason in refusals)


# Identification of a real file, through `ModelOnDisk`: it reads the GGUF header only, so every probe the two
# configs make must hold on tensors that have names, shapes and quantization types but no values.

_T5_V1_1_VOCAB_SIZE = 32128
_UMT5_VOCAB_SIZE = 256384
_WIDTH = 32  # One Q8_0 block per row.


def _write_llamacpp_t5_encoder_gguf(path: Path, *, vocab_size: int, bias_in_every_block: bool) -> None:
    """A two-block llama.cpp ``t5encoder`` GGUF with the tensor names and metadata city96's files carry.

    The values are zeros, since identification never reads them. The token embedding is one element wide:
    only its row count, the vocabulary, tells T5 v1.1 from UMT5, and at full width a 256k vocabulary would
    take hundreds of MB to write.
    """
    writer = gguf.GGUFWriter(str(path), "t5encoder")
    writer.add_uint32("t5encoder.block_count", 2)
    writer.add_tokenizer_model("t5")
    writer.add_token_list(["<pad>", "</s>", "<unk>"])
    writer.add_add_space_prefix(True)
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(0)
    writer.add_add_eos_token(True)

    matrix = quantize(np.zeros((_WIDTH, _WIDTH), dtype=np.float32), gguf.GGMLQuantizationType.Q8_0)
    norm = np.zeros(_WIDTH, dtype=np.float32)
    writer.add_tensor("token_embd.weight", np.zeros((vocab_size, 1), dtype=np.float16))
    writer.add_tensor("enc.output_norm.weight", norm)
    for block in range(2):
        prefix = f"enc.blk.{block}"
        for name in ("attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down"):
            writer.add_tensor(
                f"{prefix}.{name}.weight", matrix, raw_shape=matrix.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0
            )
        writer.add_tensor(f"{prefix}.attn_norm.weight", norm)
        writer.add_tensor(f"{prefix}.ffn_norm.weight", norm)
        if block == 0 or bias_in_every_block:
            writer.add_tensor(f"{prefix}.attn_rel_b.weight", np.zeros((32, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(
    ("vocab_size", "bias_in_every_block", "expected"),
    [(_UMT5_VOCAB_SIZE, True, WanT5Encoder_GGUF_Config), (_T5_V1_1_VOCAB_SIZE, False, T5Encoder_GGUF_Config)],
    ids=["umt5", "t5_v1_1"],
)
def test_a_real_gguf_identifies_from_its_header_alone(
    tmp_path: Path, vocab_size: int, bias_in_every_block: bool, expected: type
) -> None:
    path = tmp_path / "encoder.gguf"
    _write_llamacpp_t5_encoder_gguf(path, vocab_size=vocab_size, bias_in_every_block=bias_in_every_block)
    mod = ModelOnDisk(path)

    result = ModelConfigFactory.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    assert result.match_count == 1
    assert isinstance(result.config, expected)
    # What the configs probed is what production identification reads: the header, not the tensor data.
    assert all(tensor.quantized_data.is_meta for tensor in mod.load_state_dict().values())
