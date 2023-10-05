"""Initialization file for invokeai.backend.model_manager.storage."""
import pathlib

from .base import (  # noqa F401
    ConfigFileVersionMismatchException,
    DuplicateModelException,
    ModelConfigStore,
    UnknownModelException,
)
from .migrate import migrate_models_store  # noqa F401
from .sql import ModelConfigStoreSQL  # noqa F401
from .yaml import ModelConfigStoreYAML  # noqa F401
from ..config import AnyModelConfig  # noqa F401

def get_config_store(location: pathlib.Path) -> ModelConfigStore:
    """Return the type of ModelConfigStore appropriate to the path."""
    location = pathlib.Path(location)
    if location.suffix == ".yaml":
        return ModelConfigStoreYAML(location)
    elif location.suffix == ".db":
        return ModelConfigStoreSQL(location)
    else:
        raise Exception("Unable to determine type of configuration file '{location}'")
