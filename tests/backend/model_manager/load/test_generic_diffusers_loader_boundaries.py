"""Tests for malformed dynamic Diffusers loader configuration."""

import json
from pathlib import Path

import pytest

from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import SubModelType


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


def test_generic_diffusers_loader_accepts_a_real_transformers_class(tmp_path: Path) -> None:
    from transformers import T5EncoderModel

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["T5EncoderModel"]}), encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    assert loader.get_hf_load_class(tmp_path) is T5EncoderModel


def test_generic_diffusers_loader_accepts_a_diffusers_submodel_class(tmp_path: Path) -> None:
    from diffusers import EulerDiscreteScheduler

    (tmp_path / "model_index.json").write_text(
        json.dumps({"scheduler": ["diffusers", "EulerDiscreteScheduler"]}), encoding="utf-8"
    )
    loader = object.__new__(GenericDiffusersLoader)

    assert loader.get_hf_load_class(tmp_path, SubModelType.Scheduler) is EulerDiscreteScheduler


def test_generic_diffusers_loader_accepts_a_pipeline_namespace_class(tmp_path: Path) -> None:
    from diffusers import StableDiffusionPipeline

    loader = object.__new__(GenericDiffusersLoader)

    # Some model indexes name pipeline submodules instead of the top-level Diffusers namespace.
    assert loader._hf_definition_to_type("diffusers.pipelines.stable_diffusion", "StableDiffusionPipeline") is (
        StableDiffusionPipeline
    )


def test_generic_diffusers_loader_accepts_short_pipeline_namespace_in_model_index(tmp_path: Path) -> None:
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

    (tmp_path / "model_index.json").write_text(
        json.dumps({"safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"]}), encoding="utf-8"
    )
    loader = object.__new__(GenericDiffusersLoader)

    assert loader.get_hf_load_class(tmp_path, SubModelType.SafetyChecker) is StableDiffusionSafetyChecker


@pytest.mark.parametrize(
    "model_index",
    [
        [],
        {},
        {"scheduler": []},
        {"scheduler": ["diffusers"]},
        {"scheduler": [123, "EulerDiscreteScheduler"]},
        {"scheduler": ["diffusers", 123]},
        {"scheduler": ["untrusted.module", "EulerDiscreteScheduler"]},
        {"scheduler": ["diffusers", "NotARealScheduler"]},
        {"scheduler": ["diffusers", "ConfigMixin"]},
    ],
)
def test_generic_diffusers_loader_rejects_invalid_submodel_metadata(tmp_path: Path, model_index: object) -> None:
    (tmp_path / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    with pytest.raises(ValueError):
        loader.get_hf_load_class(tmp_path, SubModelType.Scheduler)


def test_generic_diffusers_loader_preserves_missing_config_oserror(tmp_path: Path) -> None:
    loader = object.__new__(GenericDiffusersLoader)

    with pytest.raises(OSError):
        loader.get_hf_load_class(tmp_path)


def test_generic_diffusers_loader_rejects_non_object_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('["AutoencoderKL"]', encoding="utf-8")
    loader = object.__new__(GenericDiffusersLoader)

    with pytest.raises(ValueError, match="config|object"):
        loader.get_hf_load_class(tmp_path)
