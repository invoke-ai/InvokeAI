import importlib
from logging import Logger
from pathlib import Path

import pytest

from invokeai.app.services.shared.sqlite_migrator.migration_loader import (
    MigrationBuildContext,
    MigrationLoaderError,
    build_migrations,
)


def _write_package(tmp_path: Path, package_name: str, modules: dict[str, str]) -> str:
    package_path = tmp_path / package_name
    package_path.mkdir()
    (package_path / "__init__.py").write_text("", encoding="utf-8")
    for module_name, module_source in modules.items():
        (package_path / f"{module_name}.py").write_text(module_source, encoding="utf-8")
    importlib.invalidate_caches()
    return package_name


def test_build_migrations_discovers_modules_in_numeric_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_order",
        {
            "migration_2": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration_2():
    return Migration(from_version=1, to_version=2, callback=lambda cursor: None)
""",
            "migration_1": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration_1():
    return Migration(from_version=0, to_version=1, callback=lambda cursor: None)
""",
            "not_a_migration": "VALUE = 1",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    migrations = build_migrations(
        MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
        package_name=package_name,
    )

    assert [migration.id for migration in migrations] == ["migration_1", "migration_2"]


def test_build_migrations_injects_requested_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_injection",
        {
            "migration_1": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration_1(app_config, logger, image_files):
    assert app_config == "config"
    assert logger.name == "test"
    assert image_files == "images"
    return Migration(from_version=0, to_version=1, callback=lambda cursor: None)
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    migrations = build_migrations(
        MigrationBuildContext(app_config="config", logger=Logger("test"), image_files="images"),
        package_name=package_name,
    )

    assert [migration.id for migration in migrations] == ["migration_1"]


def test_build_migrations_supports_dated_descriptive_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_dated",
        {
            "migration_2026_06_30_add_example_table": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration():
    return Migration(
        id="2026_06_30_add_example_table",
        depends_on="migration_1",
        callback=lambda cursor: None,
    )
""",
            "migration_1": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration_1():
    return Migration(from_version=0, to_version=1, callback=lambda cursor: None)
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    migrations = build_migrations(
        MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
        package_name=package_name,
    )

    assert [migration.id for migration in migrations] == ["migration_1", "2026_06_30_add_example_table"]


def test_build_migrations_rejects_dated_module_id_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_dated_id_mismatch",
        {
            "migration_2026_06_30_add_example_table": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration():
    return Migration(
        id="2026_06_30_typo",
        depends_on="migration_1",
        callback=lambda cursor: None,
    )
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(MigrationLoaderError, match="must return migration id"):
        build_migrations(
            MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
            package_name=package_name,
        )


def test_build_migrations_rejects_numeric_module_id_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_numeric_id_mismatch",
        {
            "migration_1": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration_1():
    return Migration(id="wrong", from_version=0, to_version=1, callback=lambda cursor: None)
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(MigrationLoaderError, match="must return migration id"):
        build_migrations(
            MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
            package_name=package_name,
        )


def test_build_migrations_rejects_unknown_builder_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_unknown_dependency",
        {
            "migration_1": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_migration_1(unknown_service):
    return Migration(from_version=0, to_version=1, callback=lambda cursor: None)
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(MigrationLoaderError, match="unknown dependency"):
        build_migrations(
            MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
            package_name=package_name,
        )


def test_build_migrations_rejects_missing_expected_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_missing_builder",
        {
            "migration_1": """
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
def build_something_else():
    return Migration(from_version=0, to_version=1, callback=lambda cursor: None)
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(MigrationLoaderError, match="build_migration_1"):
        build_migrations(
            MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
            package_name=package_name,
        )


def test_build_migrations_rejects_non_migration_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_bad_return",
        {
            "migration_1": """
def build_migration_1():
    return object()
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(MigrationLoaderError, match="must return Migration"):
        build_migrations(
            MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
            package_name=package_name,
        )


def test_build_migrations_rejects_malformed_migration_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_name = _write_package(
        tmp_path,
        "test_migrations_malformed",
        {
            "migration_latest": "VALUE = 1",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(MigrationLoaderError, match="Malformed migration module"):
        build_migrations(
            MigrationBuildContext(app_config=object(), logger=Logger("test"), image_files=object()),
            package_name=package_name,
        )


def test_build_migrations_discovers_production_migrations(tmp_path: Path) -> None:
    class FakeConfig:
        root_path = tmp_path
        models_path = tmp_path / "models"
        convert_cache_path = tmp_path / "models" / ".cache"
        legacy_conf_path = tmp_path / "models.yaml"
        legacy_conf_dir = tmp_path

    migrations = build_migrations(
        MigrationBuildContext(app_config=FakeConfig(), logger=Logger("test"), image_files=object())
    )

    assert [migration.id for migration in migrations] == [f"migration_{i}" for i in range(1, 33)]
    assert [migration.depends_on for migration in migrations] == [None] + [f"migration_{i}" for i in range(1, 32)]
