"""Identification tests for single-file GGUF Qwen3-VL encoders.

llama.cpp splits Qwen3-VL into a language tower and a companion ``mmproj`` file, so a Qwen3-VL GGUF
carries no visual tower -- the structural signal every other Qwen3-VL config relies on. What is left
is indistinguishable from a text-only Qwen3 of the same width (the 4B of each is 36 layers at hidden
2560), and picking wrong is silent: the file installs under the wrong type and is simply absent from
Krea-2's encoder picker.

Nothing orders the two configs. Identification iterates ``Config_Base.CONFIG_CLASSES``, a *set*, and
``matches_sort_key`` puts both encoders in the same bucket -- so a file matching both would be
resolved by arbitrary set-iteration order, differing between processes. The architecture-metadata
checks on both sides are what make that unreachable, which is why the decisive tests here go through
``ModelConfigFactory`` and assert ``match_count``.

Fixtures are real GGUF files of a few hundred bytes, so the metadata and tensor reads under test are
the real ones.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_GGUF_Config
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_GGUF_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import Qwen3VLVariantType

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

_QWEN3_VL_4B_HIDDEN_SIZE = 2560
_QWEN3_VL_8B_HIDDEN_SIZE = 4096
_LAST_LAYER = 35
_VOCAB_STUB = 8


def _write_gguf(
    path: Path,
    *,
    architecture: str = "qwen3vl",
    hidden_size: int | None = _QWEN3_VL_4B_HIDDEN_SIZE,
    last_layer: int | None = _LAST_LAYER,
    projector_type: str | None = None,
) -> Path:
    """Write a minimal but structurally honest GGUF.

    `hidden_size=None` omits the embedding entirely; `last_layer=None` omits the final block, which
    is what a multi-part quant's later shard or an interrupted download looks like.
    """
    import gguf

    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_uint32(f"{architecture}.block_count", 36)
    if projector_type is not None:
        writer.add_string("clip.projector_type", projector_type)
    if hidden_size is not None:
        writer.add_uint32(f"{architecture}.embedding_length", hidden_size)
        # Loaded shape keeps this orientation, so [1] is the hidden size, as in a real file.
        writer.add_tensor("token_embd.weight", np.zeros((_VOCAB_STUB, hidden_size), dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((4, 4), dtype=np.float32))
    if last_layer is not None:
        writer.add_tensor(f"blk.{last_layer}.attn_q.weight", np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _identify(path: Path, *, allow_unknown: bool = False) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=allow_unknown)


def _config_from(path: Path) -> Qwen3VLEncoder_GGUF_Config:
    return Qwen3VLEncoder_GGUF_Config.from_model_on_disk(ModelOnDisk(path), dict(_OVERRIDE_FIELDS))


# --- the two acceptance criteria, at the interface that decides them ------------------------------


def test_a_qwen3_vl_gguf_identifies_unambiguously_as_a_vl_encoder(tmp_path: Path) -> None:
    """One match, so no set-iteration tiebreak is involved. This is the regression that matters:
    with two matches the same file installs as a Krea-2 encoder or a Z-Image one depending on the
    process, and on the bad runs it never appears in Krea-2's picker."""
    result = _identify(_write_gguf(tmp_path / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"))

    assert result.match_count == 1
    assert isinstance(result.config, Qwen3VLEncoder_GGUF_Config)
    assert result.config.variant is Qwen3VLVariantType.Qwen3VL_4B


def test_a_text_only_qwen3_gguf_still_identifies_as_a_qwen3_encoder(tmp_path: Path) -> None:
    """The other direction: the new Qwen3-VL config must not steal Z-Image's files."""
    result = _identify(_write_gguf(tmp_path / "qwen_3_4b.gguf", architecture="qwen3"))

    assert result.match_count == 1
    assert isinstance(result.config, Qwen3Encoder_GGUF_Config)


def test_the_companion_mmproj_file_is_turned_away_with_its_reason(tmp_path: Path) -> None:
    """Installing the whole HuggingFace GGUF repo hands the user the visual tower too. Run at the
    *default* `allow_unknown`, because that is what decides what the user sees: an unrecognised file
    registers as an Unknown model and the install reports success, so a plain non-match here would
    leave a dead entry in the model list with the explanation only in the server log."""
    result = _identify(
        _write_gguf(
            tmp_path / "mmproj-Qwen3VL-4B-Instruct-F16.gguf",
            architecture="clip",
            hidden_size=None,
            projector_type="qwen3vl_merger",
        ),
        allow_unknown=True,
    )

    assert result.config is None
    assert any("visual tower" in str(reason) for reason in result.invalid_matches)


# --- variant and completeness ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_size, expected_variant",
    [
        (_QWEN3_VL_4B_HIDDEN_SIZE, Qwen3VLVariantType.Qwen3VL_4B),
        (_QWEN3_VL_8B_HIDDEN_SIZE, Qwen3VLVariantType.Qwen3VL_8B),
    ],
)
def test_variant_comes_from_the_embedding_tensor(
    tmp_path: Path, hidden_size: int, expected_variant: Qwen3VLVariantType
) -> None:
    """Krea-2 needs the 4B and Ideogram 4 the 8B; the loader builds from whichever this records."""
    config = _config_from(_write_gguf(tmp_path / "encoder.gguf", hidden_size=hidden_size))

    assert config.variant is expected_variant


def test_a_partial_quant_is_rejected_at_install_time(tmp_path: Path) -> None:
    """A multi-part quant's later shard carries a complete KV block and only some tensors. Matching
    on metadata alone would install it, offer it in the picker, and fail at the first generation.

    Asserted through the factory at the default `allow_unknown`: once the architecture metadata has
    identified the file, the rejection has to be final (`InvalidMatchError`), or the file still
    installs -- as an Unknown model, with the reason nowhere the user looks."""
    path = _write_gguf(tmp_path / "Qwen3VL-8B-Instruct-Q8_0-00002-of-00003.gguf", last_layer=None)

    result = _identify(path, allow_unknown=True)

    assert result.config is None
    assert any("incomplete" in str(reason) for reason in result.invalid_matches)


def test_a_variant_override_does_not_skip_the_completeness_check(tmp_path: Path) -> None:
    """`variant` is an install-API override. It says which Qwen3-VL the file is, not that all of its
    tensors are present, so it must not buy the file past the one check that looks."""
    path = _write_gguf(tmp_path / "Qwen3VL-8B-Instruct-Q8_0-00002-of-00003.gguf", last_layer=None)

    with pytest.raises(InvalidMatchError, match="incomplete"):
        Qwen3VLEncoder_GGUF_Config.from_model_on_disk(
            ModelOnDisk(path), {**_OVERRIDE_FIELDS, "variant": Qwen3VLVariantType.Qwen3VL_8B}
        )


def test_a_gguf_without_an_embedding_tensor_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidMatchError, match="token_embd.weight"):
        _config_from(_write_gguf(tmp_path / "encoder.gguf", hidden_size=None))


def test_unsupported_width_is_rejected(tmp_path: Path) -> None:
    """The 32B (5120) has no loader path here; accepting it would fail thousands of tensors in."""
    with pytest.raises(InvalidMatchError, match="hidden size"):
        _config_from(_write_gguf(tmp_path / "encoder.gguf", hidden_size=5120))


# --- architecture gating ---------------------------------------------------------------------------


def test_text_only_qwen3_gguf_is_not_a_vl_encoder(tmp_path: Path) -> None:
    with pytest.raises(NotAMatchError, match="qwen3vl"):
        _config_from(_write_gguf(tmp_path / "qwen3.gguf", architecture="qwen3"))


def test_moe_qwen3_vl_gguf_is_rejected(tmp_path: Path) -> None:
    """Qwen3-VL MoE needs a different transformers architecture than the loader builds."""
    with pytest.raises(NotAMatchError, match="qwen3vlmoe"):
        _config_from(_write_gguf(tmp_path / "moe.gguf", architecture="qwen3vlmoe"))


def test_non_gguf_suffix_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NotAMatchError, match="expected a .gguf file"):
        _config_from(_write_gguf(tmp_path / "encoder.safetensors"))


def test_a_file_that_is_not_a_gguf_is_reported_as_a_non_match(tmp_path: Path) -> None:
    """Identification probes every config in turn, so a corrupt file must not abort the sweep."""
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(b"not a gguf at all")

    with pytest.raises(NotAMatchError, match="general.architecture"):
        _config_from(path)
