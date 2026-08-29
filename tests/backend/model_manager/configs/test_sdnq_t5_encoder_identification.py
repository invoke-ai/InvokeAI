"""Tests for SDNQ T5 encoder identification and tokenizer/encoder directory resolution.

T5Encoder_SDNQ_Config accepts two layouts:

1. **Standalone bundle** — ``path`` is the pipeline root; T5 lives under ``text_encoder_2/`` and the
   tokenizer under a sibling ``tokenizer_2/``.
2. **Inline submodel** — ``path`` *is* the ``text_encoder_2`` folder; the tokenizer is a *sibling*
   ``tokenizer_2/`` of that folder.

The encoder loader already resolves the encoder dir by layout, but the tokenizer must be resolved
the same way (``path / "tokenizer_2"`` is wrong for the inline layout). And an install with no
resolvable ``tokenizer_2/`` must be rejected at identification, so it never registers as a selectable
T5 that then fails mid-workflow when a Tokenizer2 is requested.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_SDNQ_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "file_size": 1000,
    "name": "sdnq-t5",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _write_t5_encoder_dir(te_dir: Path) -> None:
    te_dir.mkdir(parents=True, exist_ok=True)
    (te_dir / "config.json").write_text(json.dumps({"architectures": ["T5EncoderModel"]}), encoding="utf-8")
    (te_dir / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    save_file(
        {
            "encoder.block.0.layer.0.SelfAttention.q.weight": torch.zeros(64, 32, dtype=torch.uint8),
            "encoder.block.0.layer.0.SelfAttention.q.scale": torch.zeros(64, 1, dtype=torch.float32),
        },
        str(te_dir / "model.safetensors"),
    )


def _write_tokenizer_dir(tok_dir: Path) -> None:
    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _make_standalone_bundle(root: Path, *, with_tokenizer: bool = True) -> Path:
    """path is the pipeline root; T5 under text_encoder_2/, tokenizer under sibling tokenizer_2/."""
    root.mkdir(parents=True, exist_ok=True)
    _write_t5_encoder_dir(root / "text_encoder_2")
    if with_tokenizer:
        _write_tokenizer_dir(root / "tokenizer_2")
    return root


def _make_inline_submodel(root: Path, *, with_tokenizer: bool = True) -> Path:
    """path IS the text_encoder_2 folder; tokenizer is a sibling tokenizer_2/ of that folder."""
    te_dir = root / "text_encoder_2"
    _write_t5_encoder_dir(te_dir)
    if with_tokenizer:
        _write_tokenizer_dir(root / "tokenizer_2")
    return te_dir


def test_standalone_bundle_identifies_and_resolves_dirs(tmp_path: Path):
    root = _make_standalone_bundle(tmp_path / "bundle")
    mod = ModelOnDisk(root)

    config = T5Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": root.as_posix()})
    assert config.format.value == "sdnq_quantized"
    # tokenizer_2 is a child of the root; encoder is under text_encoder_2/.
    assert T5Encoder_SDNQ_Config.resolve_tokenizer_dir(root) == root / "tokenizer_2"
    assert T5Encoder_SDNQ_Config.resolve_text_encoder_dir(root) == root / "text_encoder_2"


def test_inline_submodel_identifies_and_resolves_sibling_tokenizer(tmp_path: Path):
    te_dir = _make_inline_submodel(tmp_path / "inline")
    mod = ModelOnDisk(te_dir)

    config = T5Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": te_dir.as_posix()})
    assert config.format.value == "sdnq_quantized"
    # Inline: the tokenizer is the *sibling* tokenizer_2/, not a child of the text_encoder_2 folder.
    assert T5Encoder_SDNQ_Config.resolve_tokenizer_dir(te_dir) == te_dir.parent / "tokenizer_2"
    assert not (te_dir / "tokenizer_2").exists()  # the path the old loader used doesn't exist
    assert T5Encoder_SDNQ_Config.resolve_text_encoder_dir(te_dir) == te_dir


def test_inline_submodel_without_tokenizer_is_rejected(tmp_path: Path):
    te_dir = _make_inline_submodel(tmp_path / "inline-no-tok", with_tokenizer=False)
    mod = ModelOnDisk(te_dir)
    with pytest.raises(NotAMatchError):
        T5Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": te_dir.as_posix()})
    assert T5Encoder_SDNQ_Config.resolve_tokenizer_dir(te_dir) is None


def test_standalone_bundle_without_tokenizer_is_rejected(tmp_path: Path):
    root = _make_standalone_bundle(tmp_path / "bundle-no-tok", with_tokenizer=False)
    mod = ModelOnDisk(root)
    with pytest.raises(NotAMatchError):
        T5Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": root.as_posix()})
    assert T5Encoder_SDNQ_Config.resolve_tokenizer_dir(root) is None


def _write_sharded_t5_encoder_dir(te_dir: Path, *, split_the_pair: bool) -> None:
    """A sharded SDNQ T5 encoder with **no** quantization marker, so identification has to rely on the
    weight/scale key heuristic alone.

    With ``split_the_pair`` the weight and its scale land in different shards, which is what sharding
    by tensor order actually does — nothing keeps a pair together.
    """
    te_dir.mkdir(parents=True, exist_ok=True)
    (te_dir / "config.json").write_text(json.dumps({"architectures": ["T5EncoderModel"]}), encoding="utf-8")

    weight = {"encoder.block.0.layer.0.SelfAttention.q.weight": torch.zeros(64, 32, dtype=torch.uint8)}
    scale = {"encoder.block.0.layer.0.SelfAttention.q.scale": torch.zeros(64, 1, dtype=torch.float32)}
    other = {"encoder.block.1.layer.0.SelfAttention.q.weight": torch.zeros(64, 32, dtype=torch.uint8)}

    if split_the_pair:
        save_file(weight, str(te_dir / "model-00001-of-00002.safetensors"))
        save_file(scale, str(te_dir / "model-00002-of-00002.safetensors"))
    else:
        save_file({**weight, **scale}, str(te_dir / "model-00001-of-00002.safetensors"))
        save_file(other, str(te_dir / "model-00002-of-00002.safetensors"))


@pytest.mark.parametrize("split_the_pair", [False, True])
def test_sharded_sdnq_encoder_is_identified_even_when_the_pair_straddles_shards(tmp_path: Path, split_the_pair: bool):
    """Matching weight/scale per file rejected a valid checkpoint whenever sharding separated them."""
    root = tmp_path / f"sharded-{'split' if split_the_pair else 'together'}"
    root.mkdir(parents=True, exist_ok=True)
    _write_sharded_t5_encoder_dir(root / "text_encoder_2", split_the_pair=split_the_pair)
    _write_tokenizer_dir(root / "tokenizer_2")

    config = T5Encoder_SDNQ_Config.from_model_on_disk(ModelOnDisk(root), {**_REQUIRED_FIELDS, "path": root.as_posix()})

    assert config is not None
