"""Tests for malformed dynamic Diffusers loader configuration."""

import json
from pathlib import Path

import pytest

from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"_class_name": "NotARealDiffusersClass"},
        {"architectures": ["NotARealTransformersClass"]},
        {"_class_name": 123},
        {"architectures": []},
        {"architectures": "AutoModel"},
        {"architectures": [123]},
        {"architectures": {}},
        {"_class_name": "ConfigMixin"},
    ],
)
def test_generic_diffusers_loader_rejects_invalid_class_metadata(tmp_path: Path, config: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    with pytest.raises(ValueError, match="class|config|model"):
        loader.get_hf_load_class(tmp_path)


def test_generic_diffusers_loader_rejects_invalid_json(tmp_path: Path) -> None:
    """An interrupted write or a manually edited config must fail before model loading."""
    (tmp_path / "config.json").write_text('{"_class_name":', encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    with pytest.raises(OSError, match="config|JSON|json"):
        loader.get_hf_load_class(tmp_path)


def test_generic_diffusers_loader_accepts_a_real_loadable_class(tmp_path: Path) -> None:
    from diffusers import AutoencoderKL

    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL"}), encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    assert loader.get_hf_load_class(tmp_path) is AutoencoderKL


def test_generic_diffusers_loader_rejects_non_object_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('["AutoencoderKL"]', encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    with pytest.raises(ValueError, match="config|object"):
        loader.get_hf_load_class(tmp_path)
