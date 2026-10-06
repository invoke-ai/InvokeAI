import asyncio
from logging import getLogger
from unittest.mock import MagicMock

from invokeai.app.api import dependencies
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


def test_initialize_does_not_vacuum_database(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dependencies, "validate_architectures", lambda: None)
    monkeypatch.setattr(dependencies, "warm_up_attention", lambda *_: None)
    monkeypatch.setattr(dependencies, "set_jwt_secret", lambda _: None)
    monkeypatch.setattr(
        dependencies.ModelManagerService,
        "build_model_manager",
        MagicMock(return_value=MagicMock()),
    )

    original_init_db = dependencies.init_db
    initialized_databases: list[SqliteDatabase] = []

    def initialize_database(*args, **kwargs) -> SqliteDatabase:
        database = original_init_db(*args, **kwargs)
        initialized_databases.append(database)
        return database

    monkeypatch.setattr(dependencies, "init_db", initialize_database)
    clean_calls: list[SqliteDatabase] = []
    monkeypatch.setattr(SqliteDatabase, "clean", lambda database: clean_calls.append(database))

    config = InvokeAIAppConfig(
        use_memory_db=True,
        image_index_enabled=False,
        node_cache_size=0,
        scan_models_on_startup=False,
    )
    config._root = tmp_path
    loop = asyncio.new_event_loop()
    original_invoker = getattr(ApiDependencies, "invoker", None)

    try:
        ApiDependencies.initialize(config=config, event_handler_id=0, loop=loop, logger=getLogger("test"))

        assert clean_calls == []
        assert len(initialized_databases) == 1
        assert ApiDependencies.invoker.services.database is initialized_databases[0]
    finally:
        invoker = getattr(ApiDependencies, "invoker", None)
        if invoker is not None and invoker is not original_invoker:
            invoker.stop()
            if original_invoker is None:
                delattr(ApiDependencies, "invoker")
            else:
                ApiDependencies.invoker = original_invoker
        loop.close()
