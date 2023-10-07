import contextlib
from pathlib import Path
from typing import Optional, Union

import pytest
import torch

from invokeai.app.services.config.invokeai_config import InvokeAIAppConfig
from invokeai.app.services.model_record_service import ModelRecordServiceBase
from invokeai.backend.model_manager import BaseModelType, ModelType, SubModelType, UnknownModelException
from invokeai.backend.model_manager.install import ModelInstall
from invokeai.backend.model_manager.loader import ModelInfo, ModelLoad


@pytest.fixture(scope="session")
def torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_installer():
    """A global ModelInstall pytest fixture to be used by many tests."""
    # HACK(ryand): InvokeAIAppConfig.get_config() returns a singleton config object. This can lead to weird interactions
    # between tests that need to alter the config. For example, some tests change the 'root' directory in the config,
    # which can cause `install_and_load_model(...)` to re-download the model unnecessarily. As a temporary workaround,
    # we pass a kwarg to get_config, which causes the config to be re-loaded. To fix this properly, we should stop using
    # a singleton.
    #
    # REPLY(lstein): Don't use get_config() here. Just use the regular pydantic constructor.
    #
    config = InvokeAIAppConfig(log_level="info")
    model_store = ModelRecordServiceBase.get_impl(config)
    return ModelInstall(model_store, config)


def install_and_load_model(
    model_installer: ModelInstall,
    model_path_id_or_url: Union[str, Path],
    model_name: str,
    base_model: BaseModelType,
    model_type: ModelType,
    submodel_type: Optional[SubModelType] = None,
) -> ModelInfo:
    """Install a model if it is not already installed, then get the ModelInfo for that model.

    This is intended as a utility function for tests.

    Args:
        model_installer (ModelInstall): The model installer.
        model_path_id_or_url (Union[str, Path]): The path, HF ID, URL, etc. where the model can be installed from if it
            is not already installed.
        model_name (str): The model name, forwarded to ModelManager.get_model(...).
        base_model (BaseModelType): The base model, forwarded to ModelManager.get_model(...).
        model_type (ModelType): The model type, forwarded to ModelManager.get_model(...).
        submodel_type (Optional[SubModelType]): The submodel type, forwarded to ModelManager.get_model(...).

    Returns:
        ModelInfo
    """
    # If the requested model is already installed, return its ModelInfo.
    loader = ModelLoad(config=model_installer.config, store=model_installer.store)
    with contextlib.suppress(UnknownModelException):
        model = model_installer.store.model_info_by_name(model_name, base_model, model_type)
        return loader.get_model(model.key, submodel_type)

    # Install the requested model.
    model_installer.install(model_path_id_or_url)
    model_installer.wait_for_installs()

    try:
        model = model_installer.store.model_info_by_name(model_name, base_model, model_type)
        return loader.get_model(model.key, submodel_type)
    except UnknownModelException as e:
        raise Exception(
            "Failed to get model info after installing it. There could be a mismatch between the requested model and"
            f" the installation id ('{model_path_id_or_url}'). Error: {e}"
        )
