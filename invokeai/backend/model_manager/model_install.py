# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Install/delete models.

Typical usage:

  from invokeai.app.services.config import InvokeAIAppConfig
  from invokeai.backend.model_manager import ModelInstall
  from invokeai.backend.model_manager.storage import ModelConfigStoreSQL

  config = InvokeAIAppConfig.get_config()
  store = ModelConfigStoreSQL(config.db_path)
  installer = ModelInstall(store=store, config=config)

  # register config, don't move path
  id: str = installer.register_model('/path/to/model')

  # register config, and install model in `models`
  id: str = installer.install_model('/path/to/model')

  # unregister, don't delete
  installer.forget(id)

  # unregister and delete model from disk
  installer.delete_model(id)

  # scan directory recursively and install all new models found
  ids: List[str] = installer.scan_directory('/path/to/directory')

  # unregister any model whose path is no longer valid
  ids: List[str] = installer.garbage_collect()

  hash: str = installer.hash('/path/to/model')  # should be same as id above

The following exceptions may be raised:
  DuplicateModelException
  UnknownModelTypeException
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger
from .storage import ModelConfigStore, UnknownModelException

class ModelInstallBase(ABC):
    """Abstract base class for InvokeAI model installation"""

    @abstractmethod
    def __init__(self,
                 store: Optional[ModelConfigStore] = None,
                 config: Optional[InvokeAIAppConfig] = None,
                 logger: Optional[InvokeAILogger] = None
                 ):
        """
        Create ModelInstall object.

        :param store: Optional ModelConfigStore. If None passed,
        defaults to `configs/models.yaml`.
        :param config: Optional InvokeAIAppConfig. If None passed,
        uses the system-wide default app config.
        :param logger: Optional InvokeAILogger. If None passed,
        uses the system-wide default logger.
        """
        pass

    @abstractmethod
    def register(self, model_path: Path) -> str:
        """
        Probe and register the model at model_path.

        :param model_path: Filesystem Path to the model.
        :returns id: The string ID of the registered model.
        """
        pass

    @abstractmethod
    def install(self, model_path: Path) -> str:
        """
        Probe, register and install the model in the models directory.

        This involves moving the model from its current location into
        the models directory handled by InvokeAI.

        :param model_path: Filesystem Path to the model.
        :returns id: The string ID of the installed model.
        """
        pass

    @abstractmethod
    def forget(self, id: str):
        """
        Unregister the model identified by id.

        This removes the model from the registry without
        deleting the underlying model from disk.

        :param id: The string ID of the model to forget.
        :raises UnknownModelException: In the event the ID is unknown.
        """
        pass

    @abstractmethod
    def delete(self, id: str) -> str:
        """
        Unregister and delete the model identified by id.

        This removes the model from the registry and
        deletes the underlying model from disk.

        :param id: The string ID of the model to forget.
        :raises UnknownModelException: In the event the ID is unknown.
        :raises OSError: In the event the model cannot be deleted from disk.
        """
        pass

    @abstractmethod
    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:
        """
        Recursively scan directory for new models and register or install them.

        :param scan_dir: Path to the directory to scan.
        :param install: Install if True, otherwise register in place.
        :returns list of IDs: Returns list of IDs of models registered/installed
        """
        pass

    @abstractmethod
    def garbage_collect(self) -> List[str]:
        """
        Unregister any models whose paths are no longer valid.

        This checks each registered model's path. Models with paths that are
        no longer found on disk will be unregistered.

        :return List[str]: Return the list of model IDs that were unregistered.
        """
        pass

    @abstractmethod
    def hash(self, model_path: Path) -> str:
        """
        Compute and return the fast hash of the model.

        :param model_path: Path to the model on disk.
        :return str: FastHash of the model for use as an ID.
        """
        pass


class ModelInstall(ModelInstallBase):
    """Model installer class handles installation from a local path."""

    _config: InvokeAIAppConfig
    _logger: InvokeAILogger
    _store: ModelConfigStore

    def __init__(self,  
                 store: Optional[ModelConfigStore] = None,
                 config: Optional[InvokeAIAppConfig] = None,
                 logger: Optional[InvokeAILogger] = None
                 ):                                             # noqa D107 - use base class docstrings
        self._config = config or InvokeAIAppConfig.get_config()
        self._logger = logger or InvokeAILogger.getLogger()
        if store is None:
            from .storage import ModelConfigStoreYAML
            store = ModelConfigStoreYAML(config.model_conf_path)
        self._store = store

    def register(self, model_path: Path) -> str:  # noqa D102
        pass

    def install(self, model_path: Path) -> str:    # noqa D102
        pass

    def forget(self, id: str) -> str:    # noqa D102
        pass

    def delete(self, id: str) -> str:    # noqa D102
        pass

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:     # noqa D102
        pass

    def garbage_collect(self) -> List[str]:      # noqa D102
        pass

    def hash(self, model_path: Path) -> str:     # noqa D102
        pass
