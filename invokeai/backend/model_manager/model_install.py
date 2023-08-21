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
from pathlib import Path
from typing import Optional, List
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger
from .storage import ModelConfigStore


class ModelInstall(object):
    """Model installer class handles installation from a local path."""

    _config: InvokeAIAppConfig
    _logger: InvokeAILogger
    _store: ModelConfigStore

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
        self._config = config or InvokeAIAppConfig.get_config()
        self._logger = logger or InvokeAILogger.getLogger()
        if store is None:
            from .storage import ModelConfigStoreYAML
            store = ModelConfigStoreYAML(config.model_conf_path)
        self._store = store


    def register(self, model_path: Path) -> str:
        """Probe and register the model at model_path."""
        pass

    def install(self, model_path: Path) -> str:
        """Probe, register and Install the model in the models directory."""
        pass

    def forget(self, id: str) -> str:
        """Unregister the model identified by id."""
        pass

    def delete(self, id: str) -> str:
        """
        Unregister and delete the model identified by id.
        Note that this deletes the model unconditionally.
        """
        pass

    def scan_directory(self, scan_dir: Path, install: bool=False) -> List[str]:
        """Scan directory for new models and register or install them."""
        pass

    def garbage_collect(self):
        """Unregister any models whose paths are no longer valid."""
        pass

    def hash(self, model_path: Path) -> str:
        """Compute the fast hash of the model."""
        pass

