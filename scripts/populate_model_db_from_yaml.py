#!/bin/env python

from argparse import ArgumentParser, Namespace
from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig, get_config
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.services.model_install import ModelInstallService
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger


def get_args() -> Namespace:
    parser = ArgumentParser(description="Update models database from yaml file")
    parser.add_argument("--root", type=Path, required=False, default=None)
    parser.add_argument("--yaml_file", type=Path, required=False, default=None)
    return parser.parse_args()


def populate_config() -> InvokeAIAppConfig:
    args = get_args()
    config = get_config()
    if args.root:
        config._root = args.root
    if args.yaml_file:
        config.legacy_models_yaml_path = args.yaml_file
    else:
        config.legacy_models_yaml_path = config.root_path / "configs/models.yaml"
    return config


def initialize_installer(config: InvokeAIAppConfig) -> ModelInstallService:
    logger = InvokeAILogger.get_logger(config=config)
    db = SqliteDatabase(config.db_path, logger)
    record_store = ModelRecordServiceSQL(db)
    queue = DownloadQueueService()
    queue.start()
    installer = ModelInstallService(app_config=config, record_store=record_store, download_queue=queue)
    return installer


def main() -> None:
    config = populate_config()
    installer = initialize_installer(config)
    installer._migrate_yaml(rename_yaml=False, overwrite_db=True)
    print("\n<INSTALLED MODELS>")
    print("\t".join(["key", "name", "type", "path"]))
    for model in installer.record_store.all_models():
        print("\t".join([model.key, model.name, model.type, (config.models_path / model.path).as_posix()]))


if __name__ == "__main__":
    main()
