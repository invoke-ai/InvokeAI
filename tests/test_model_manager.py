from pathlib import Path

import pytest

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend import ModelManager, BaseModelType, ModelType, SubModelType

BASIC_MODEL_NAME = ("SDXL base", BaseModelType.StableDiffusionXL, ModelType.Main)
VAE_OVERRIDE_MODEL_NAME = ("SDXL with VAE", BaseModelType.StableDiffusionXL, ModelType.Main)
VAE_NULL_OVERRIDE_MODEL_NAME = ("SDXL with empty VAE", BaseModelType.StableDiffusionXL, ModelType.Main)


@pytest.fixture
def model_manager(datadir) -> ModelManager:
    InvokeAIAppConfig.get_config(root=datadir)
    return ModelManager(datadir / "configs" / "relative_sub.models.yaml")


def test_get_model_names(model_manager: ModelManager):
    names = model_manager.model_names()
    assert names[:2] == [BASIC_MODEL_NAME, VAE_OVERRIDE_MODEL_NAME]


def test_get_model_path_for_diffusers(model_manager: ModelManager, datadir: Path):
    model_config = model_manager._get_model_config(BASIC_MODEL_NAME[1], BASIC_MODEL_NAME[0], BASIC_MODEL_NAME[2])
    top_model_path, is_override = model_manager._get_model_path(model_config)
    expected_model_path = datadir / "models" / "sdxl" / "main" / "SDXL base 1_0"
    assert top_model_path == expected_model_path
    assert not is_override


def test_get_model_path_for_overridden_vae(model_manager: ModelManager, datadir: Path):
    model_config = model_manager._get_model_config(
        VAE_OVERRIDE_MODEL_NAME[1], VAE_OVERRIDE_MODEL_NAME[0], VAE_OVERRIDE_MODEL_NAME[2]
    )
    vae_model_path, is_override = model_manager._get_model_path(model_config, SubModelType.Vae)
    expected_vae_path = datadir / "models" / "sdxl" / "vae" / "sdxl-vae-fp16-fix"
    assert vae_model_path == expected_vae_path
    assert is_override


def test_get_model_path_for_null_overridden_vae(model_manager: ModelManager, datadir: Path):
    model_config = model_manager._get_model_config(
        VAE_NULL_OVERRIDE_MODEL_NAME[1], VAE_NULL_OVERRIDE_MODEL_NAME[0], VAE_NULL_OVERRIDE_MODEL_NAME[2]
    )
    vae_model_path, is_override = model_manager._get_model_path(model_config, SubModelType.Vae)
    assert not is_override
