from pathlib import Path

import pytest

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend import SubModelType
from invokeai.backend.model_manager.loader import ModelLoad

BASIC_MODEL_NAME = "sdxl-base-1-0"
VAE_OVERRIDE_MODEL_NAME = "sdxl-base-with-custom-vae-1-0"
VAE_NULL_OVERRIDE_MODEL_NAME = "sdxl-base-with-empty-vae-1-0"


@pytest.fixture
def model_manager(datadir) -> ModelLoad:
    config = InvokeAIAppConfig(root=datadir, conf_path="configs/relative_sub.models.yaml")
    return ModelLoad(config=config)


def test_get_model_names(model_manager: ModelLoad):
    store = model_manager.store
    names = [x.name for x in store.all_models()]
    assert names[:2] == [BASIC_MODEL_NAME, VAE_OVERRIDE_MODEL_NAME]


def test_get_model_path_for_diffusers(model_manager: ModelLoad, datadir: Path):
    models = model_manager.store.search_by_name(model_name=BASIC_MODEL_NAME)
    assert len(models) == 1
    model_config = models[0]
    top_model_path, is_override = model_manager._get_model_path(model_config)
    expected_model_path = datadir / "models" / "sdxl" / "main" / "SDXL base 1_0"
    assert top_model_path == expected_model_path
    assert not is_override


def test_get_model_path_for_overridden_vae(model_manager: ModelLoad, datadir: Path):
    models = model_manager.store.search_by_name(model_name=VAE_OVERRIDE_MODEL_NAME)
    assert len(models) == 1
    model_config = models[0]
    vae_model_path, is_override = model_manager._get_model_path(model_config, SubModelType.Vae)
    expected_vae_path = datadir / "models" / "sdxl" / "vae" / "sdxl-vae-fp16-fix"
    assert vae_model_path == expected_vae_path
    assert is_override


def test_get_model_path_for_null_overridden_vae(model_manager: ModelLoad, datadir: Path):
    model_config = model_manager.store.search_by_name(model_name=VAE_NULL_OVERRIDE_MODEL_NAME)[0]
    vae_model_path, is_override = model_manager._get_model_path(model_config, SubModelType.Vae)
    assert not is_override
