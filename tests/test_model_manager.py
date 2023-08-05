from pathlib import Path

import pytest

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend import ModelManager, BaseModelType, ModelType, SubModelType


@pytest.fixture
def model_manager(datadir) -> ModelManager:
    InvokeAIAppConfig.get_config(root=datadir)
    return ModelManager(datadir / "configs" / "relative_sub.models.yaml")


def test_get_model_names(model_manager: ModelManager):
    names = model_manager.model_names()
    assert names[:2] == [
        ("SDXL base", BaseModelType.StableDiffusionXL, ModelType.Main),
        ("SDXL with VAE", BaseModelType.StableDiffusionXL, ModelType.Main),
    ]


def test_get_model_path_for_diffusers(model_manager: ModelManager, datadir: Path):
    model_config = model_manager._get_model_config(BaseModelType.StableDiffusionXL, "SDXL base", ModelType.Main)
    top_model_path, is_override = model_manager._get_model_path(model_config)
    expected_model_path = datadir / "models" / "sdxl" / "main" / "SDXL base 1_0"
    assert top_model_path == expected_model_path
    assert not is_override


def test_get_model_path_for_overridden_vae(model_manager: ModelManager, datadir: Path):
    model_config = model_manager._get_model_config(BaseModelType.StableDiffusionXL, "SDXL with VAE", ModelType.Main)
    vae_model_path, is_override = model_manager._get_model_path(model_config, SubModelType.Vae)
    expected_vae_path = datadir / "models" / "sdxl" / "vae" / "sdxl-vae-fp16-fix"
    assert vae_model_path == expected_vae_path
    assert is_override
