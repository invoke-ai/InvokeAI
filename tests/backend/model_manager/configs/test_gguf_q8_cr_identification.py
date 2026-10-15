"""Install-time decisions for ComfyUI-GGUF ``Q8_CR`` files, made through ``ModelConfigFactory``.

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
from tests.fixtures.quantized_payloads import q8_cr_marker, quantize_convrot, write_gguf

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


def _krea2_q8_cr(path: Path, marker: dict[str, Any]) -> Path:
    """A native-named Krea-2 GGUF with one quantized layer, as ComfyUI-GGUF's converter writes one."""
    payload = quantize_convrot(torch.randn(4, 256))
    write_gguf(
        path,
        {
            "txtfusion.projector.weight": torch.ones(12, 1),
            "first.weight": torch.ones(8, 4),
            "tproj.0.weight": torch.ones(8, 4),
            "blocks.0.attn.wq.weight": payload.codes,
            "blocks.0.attn.wq.weight_scale": payload.scale,
        },
        quant={"blocks.0.attn.wq.weight": marker},
    )
    return path


def _identify(path: Path) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=True)


def test_a_q8_cr_krea2_gguf_installs_as_a_krea2_gguf(tmp_path: Path) -> None:
    result = _identify(_krea2_q8_cr(tmp_path / "krea2_turbo-Q8_CR.gguf", q8_cr_marker()))

    assert isinstance(result.config, Main_GGUF_Krea2_Config)
    assert not result.invalid_matches


def test_a_q8_cr_gguf_is_refused_with_its_reason_where_the_loader_cannot_decode_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate every other GGUF config sits behind. Driven through the Krea-2 config with its
    capability switched off, so the fixture is one that is otherwise known to install."""
    monkeypatch.setattr(Main_GGUF_Krea2_Config, "DECODES_GGUF_Q8_CR", False)

    result = _identify(_krea2_q8_cr(tmp_path / "krea2_turbo-Q8_CR.gguf", q8_cr_marker()))

    assert result.config is None
    assert any("Q8_CR" in str(reason) for reason in result.invalid_matches)


def test_a_q4_cr_gguf_is_refused_with_its_reason(tmp_path: Path) -> None:
    """ComfyUI-GGUF's Q4_CR is advertised beside Q8_CR and stores its codes as I8 as well."""
    result = _identify(_krea2_q8_cr(tmp_path / "krea2_turbo-Q4_CR.gguf", {"format": "int4_cr", "backing": "w4a4"}))

    assert result.config is None
    assert any("int4_cr" in str(reason) for reason in result.invalid_matches)
