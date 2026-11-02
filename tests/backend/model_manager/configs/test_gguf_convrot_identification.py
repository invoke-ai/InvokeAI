"""Install-time decisions for ComfyUI-GGUF ``Q8_CR`` and ``Q4_CR`` files, made through ``ModelConfigFactory``.

These run at the *default* ``allow_unknown``, because that is what decides what the user sees: a file a
config recognises but cannot load must come back as a refusal with its reason, not register as an Unknown
model whose explanation is only in the server log. Fixtures are real GGUF files of a few kilobytes.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.main import Main_GGUF_Krea2_Config
from tests.fixtures.quantized_payloads import (
    q4_cr_marker,
    q8_cr_marker,
    quantize_convrot,
    quantize_convrot_int4,
    write_gguf,
)

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


def _krea2_convrot(path: Path, marker: dict[str, Any], *, packed_int4: bool = False) -> Path:
    """A native-named Krea-2 GGUF with one quantized layer, as ComfyUI-GGUF's converter writes one: int8 codes and
    an F32 scale for Q8_CR, packed codes and an F16 scale for Q4_CR."""
    if packed_int4:
        payload = quantize_convrot_int4(torch.randn(4, 256))
        layer = {"blocks.0.attn.wq.weight": payload.packed, "blocks.0.attn.wq.weight_scale": payload.scale.half()}
    else:
        int8 = quantize_convrot(torch.randn(4, 256))
        layer = {"blocks.0.attn.wq.weight": int8.codes, "blocks.0.attn.wq.weight_scale": int8.scale}
    write_gguf(
        path,
        {
            "txtfusion.projector.weight": torch.ones(12, 1),
            "first.weight": torch.ones(8, 4),
            "tproj.0.weight": torch.ones(8, 4),
            **layer,
        },
        quant={"blocks.0.attn.wq.weight": marker},
    )
    return path


def _identify(path: Path) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=True)


def test_a_q8_cr_krea2_gguf_installs_as_a_krea2_gguf(tmp_path: Path) -> None:
    result = _identify(_krea2_convrot(tmp_path / "krea2_turbo-Q8_CR.gguf", q8_cr_marker()))

    assert isinstance(result.config, Main_GGUF_Krea2_Config)
    assert not result.invalid_matches


def test_a_q8_cr_gguf_is_refused_with_its_reason_where_the_loader_cannot_decode_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate every other GGUF config sits behind. Driven through the Krea-2 config with its
    capability switched off, so the fixture is one that is otherwise known to install."""
    monkeypatch.setattr(Main_GGUF_Krea2_Config, "DECODES_GGUF_CONVROT", frozenset())

    result = _identify(_krea2_convrot(tmp_path / "krea2_turbo-Q8_CR.gguf", q8_cr_marker()))

    assert result.config is None
    assert any("Q8_CR" in str(reason) for reason in result.invalid_matches)


def test_a_loader_that_decodes_q8_cr_but_not_int4_refuses_a_q4_cr_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q8_CR and Q4_CR are opted into separately, as the safetensors loaders opt into int4 separately: a loader wired
    for int8 alone would otherwise install a Q4_CR file and refuse it at its first load."""
    monkeypatch.setattr(Main_GGUF_Krea2_Config, "DECODES_GGUF_CONVROT", frozenset({"int8_tensorwise"}))

    result = _identify(_krea2_convrot(tmp_path / "krea2_turbo-Q4_CR.gguf", q4_cr_marker(4, 256), packed_int4=True))

    assert result.config is None
    assert any("(Q4_CR)" in str(reason) for reason in result.invalid_matches)


def test_a_q4_cr_krea2_gguf_installs_as_a_krea2_gguf(tmp_path: Path) -> None:
    result = _identify(_krea2_convrot(tmp_path / "krea2_turbo-Q4_CR.gguf", q4_cr_marker(4, 256), packed_int4=True))

    assert isinstance(result.config, Main_GGUF_Krea2_Config)
    assert not result.invalid_matches


def test_the_retired_q4_cr_backing_is_refused_with_its_reason(tmp_path: Path) -> None:
    """ComfyUI-GGUF's first Q4_CR was W4A16 with group-wise scales and no `backing` entry. Its own loader no longer
    reads it either, and it stores its codes as I8 just the same."""
    marker = q4_cr_marker(4, 256)
    del marker["backing"]

    result = _identify(_krea2_convrot(tmp_path / "krea2_turbo-Q4_CR.gguf", marker, packed_int4=True))

    assert result.config is None
    assert any("'int4_cr' without a backing" in str(reason) for reason in result.invalid_matches)
