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

  id: str = installer.install_model('/path/to/model')
  installer.delete_model(id)

  hash: str = installer.hash('/path/to/model')  # should be same as id above

The following exceptions may be raised:
  DuplicateModelException
  UnknownModelTypeException
"""


