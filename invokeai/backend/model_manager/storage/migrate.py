# Copyright (c) 2023 The InvokeAI Development Team

import shutil
from pathlib import Path

from omegaconf import OmegaConf

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger

from .base import CONFIG_FILE_VERSION


def migrate_models_store(config: InvokeAIAppConfig):
    # avoid circular import
    from invokeai.backend.model_manager import DuplicateModelException, InvalidModelException, ModelInstall
    from invokeai.backend.model_manager.storage import get_config_store

    app_config = InvokeAIAppConfig.get_config()
    logger = InvokeAILogger.getLogger()
    old_file: Path = app_config.model_conf_path
    new_file: Path = old_file.with_name("models3_2.yaml")

    old_conf = OmegaConf.load(old_file)
    store = get_config_store(new_file)
    installer = ModelInstall(store=store)
    logger.info(f"Migrating old models file at {old_file} to new {CONFIG_FILE_VERSION} format")

    for model_key, stanza in old_conf.items():
        if model_key == "__metadata__":
            assert (
                stanza["version"] == "3.0.0"
            ), f"This script works on version 3.0.0 yaml files, but your configuration points to a {stanza['version']} version"
            continue

        base_type, model_type, model_name = model_key.split("/")

        try:
            path = app_config.models_path / stanza["path"]
            new_key = installer.register_path(path)
        except DuplicateModelException:
            # if model already installed, then we just update its info
            models = store.search_by_name(model_name=model_name, base_model=base_type, model_type=model_type)
            if len(models) != 1:
                continue
            new_key = models[0].key
        except Exception as excp:
            print(str(excp))

        model_info = store.get_model(new_key)
        if vae := stanza.get("vae"):
            model_info.vae = (app_config.models_path / vae).as_posix()
        if model_config := stanza.get("config"):
            model_info.config = (app_config.root_path / model_config).as_posix()
        model_info.description = stanza.get("description")
        store.update_model(new_key, model_info)
        store.update_model(new_key, model_info)

    logger.info(f"Original version of models config file saved as {str(old_file) + '.orig'}")
    shutil.move(old_file, str(old_file) + ".orig")
    shutil.move(new_file, old_file)
