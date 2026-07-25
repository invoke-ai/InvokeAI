"""Regression tests for Gemma2 encoder (PiD caption encoder) config probing.

PiD's caption projection is hard-wired to Gemma-2-2b's 2304-dim hidden state. The classifier used
to accept every ``Gemma2ForCausalLM`` directory, so larger variants (9B → 3584, 27B → 4608) were
offered as compatible PiD encoders and then failed with a matrix-shape error deep inside inference.
The config now rejects any hidden size other than 2304 up front.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.gemma2_encoder import Gemma2Encoder_Gemma2Encoder_Config
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError

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


def _make_gemma_dir(root: Path, *, hidden_size: int) -> MagicMock:
    root.joinpath("config.json").write_text(
        json.dumps({"architectures": ["Gemma2ForCausalLM"], "hidden_size": hidden_size})
    )
    root.joinpath("tokenizer.json").write_text("{}")
    mod = MagicMock()
    mod.path = root
    return mod


def test_gemma_2_2b_matches() -> None:
    """The reference Gemma-2-2b encoder (2304-dim) is a valid PiD caption encoder."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_gemma_dir(Path(tmpdir), hidden_size=2304)
        config = Gemma2Encoder_Gemma2Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "gemma2_encoder"


@pytest.mark.parametrize("hidden_size", [3584, 4608])
def test_incompatible_gemma_sizes_are_rejected(hidden_size: int) -> None:
    """Gemma 2 9B (3584) and 27B (4608) share the architecture but are incompatible with PiD."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_gemma_dir(Path(tmpdir), hidden_size=hidden_size)
        with pytest.raises(NotAMatchError, match="hidden_size"):
            Gemma2Encoder_Gemma2Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
