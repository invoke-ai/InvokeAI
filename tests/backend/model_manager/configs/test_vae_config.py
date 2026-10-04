"""Tests for four-channel Diffusers VAE identification heuristics."""

import json
from pathlib import Path

import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType


@pytest.mark.parametrize(
    ("config", "name", "expected_base"),
    [
        (
            {"_class_name": "AutoencoderKL", "scaling_factor": 0.18215, "sample_size": 512},
            "vae",
            BaseModelType.StableDiffusion1,
        ),
        (
            {"_class_name": "AutoencoderKL", "scaling_factor": 0.13025, "sample_size": 1024},
            "vae",
            BaseModelType.StableDiffusionXL,
        ),
        (
            {"_class_name": "AutoencoderKL", "scaling_factor": 0.5, "sample_size": 768},
            "my-xl-vae",
            BaseModelType.StableDiffusionXL,
        ),
    ],
)
def test_diffusers_vae_known_base_detection(
    tmp_path: Path, config: dict, name: str, expected_base: BaseModelType
) -> None:
    model_path = tmp_path / name
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "diffusion_pytorch_model.safetensors").touch()

    result = ModelConfigFactory.from_model_on_disk(model_path, allow_unknown=False)

    assert result.config is not None
    assert result.config.base is expected_base
