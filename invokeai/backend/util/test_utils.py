import contextlib
from pathlib import Path
from typing import Optional, Union

import pytest
import torch

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.install.model_install_backend import ModelInstall
from invokeai.backend.model_management.model_manager import LoadedModelInfo
from invokeai.backend.model_management.models.base import BaseModelType, ModelNotFoundException, ModelType, SubModelType


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
    return ModelInstall(InvokeAIAppConfig.get_config(log_level="info"))


def install_and_load_model(
    model_installer: ModelInstall,
    model_path_id_or_url: Union[str, Path],
    model_name: str,
    base_model: BaseModelType,
    model_type: ModelType,
    submodel_type: Optional[SubModelType] = None,
) -> LoadedModelInfo:
    """Install a model if it is not already installed, then get the LoadedModelInfo for that model.

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
        LoadedModelInfo
    """
    # If the requested model is already installed, return its LoadedModelInfo.
    with contextlib.suppress(ModelNotFoundException):
        return model_installer.mgr.get_model(model_name, base_model, model_type, submodel_type)

    # Install the requested model.
    model_installer.heuristic_import(model_path_id_or_url)

    try:
        return model_installer.mgr.get_model(model_name, base_model, model_type, submodel_type)
    except ModelNotFoundException as e:
        raise Exception(
            "Failed to get model info after installing it. There could be a mismatch between the requested model and"
            f" the installation id ('{model_path_id_or_url}'). Error: {e}"
        )
