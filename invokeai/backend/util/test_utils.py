import contextlib
from pathlib import Path
from typing import Optional, Union

import pytest
import torch

from invokeai.app.services.model_manager import ModelManagerServiceBase
from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager import BaseModelType, LoadedModel, ModelType, SubModelType


@pytest.fixture(scope="session")
def torch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def install_and_load_model(
    model_manager: ModelManagerServiceBase,
    model_path_id_or_url: Union[str, Path],
    model_name: str,
    base_model: BaseModelType,
    model_type: ModelType,
    submodel_type: Optional[SubModelType] = None,
) -> LoadedModel:
    """Install a model if it is not already installed, then get the LoadedModel for that model.

    This is intended as a utility function for tests.

    Args:
        mm2_model_manager (ModelManagerServiceBase): The model manager
        model_path_id_or_url (Union[str, Path]): The path, HF ID, URL, etc. where the model can be installed from if it
            is not already installed.
        model_name (str): The model name, forwarded to ModelManager.get_model(...).
        base_model (BaseModelType): The base model, forwarded to ModelManager.get_model(...).
        model_type (ModelType): The model type, forwarded to ModelManager.get_model(...).
        submodel_type (Optional[SubModelType]): The submodel type, forwarded to ModelManager.get_model(...).

    Returns:
        LoadedModelInfo
    """
    # If the requested model is already installed, return its LoadedModel
    with contextlib.suppress(UnknownModelException):
        # TODO: Replace with wrapper call
        loaded_model: LoadedModel = model_manager.load.load_model_by_attr(
            model_name=model_name, base_model=base_model, model_type=model_type
        )
        return loaded_model

    # Install the requested model.
    job = model_manager.install.heuristic_import(model_path_id_or_url)
    model_manager.install.wait_for_job(job, timeout=10)
    assert job.complete

    try:
        loaded_model = model_manager.load.load_model_by_config(job.config_out)
        return loaded_model
    except UnknownModelException as e:
        raise Exception(
            "Failed to get model info after installing it. There could be a mismatch between the requested model and"
            f" the installation id ('{model_path_id_or_url}'). Error: {e}"
        )
