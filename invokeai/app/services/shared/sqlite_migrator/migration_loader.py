import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass
from logging import Logger
from types import ModuleType
from typing import Any

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

DEFAULT_MIGRATIONS_PACKAGE = "invokeai.app.services.shared.sqlite_migrator.migrations"
_MIGRATION_MODULE_RE = re.compile(r"^migration_(\d+)$")
_DATED_MIGRATION_MODULE_RE = re.compile(r"^migration_(\d{4}_\d{2}_\d{2}_[a-z0-9][a-z0-9_]*)$")


class MigrationLoaderError(RuntimeError):
    """Raised when migration discovery or construction fails."""


@dataclass(frozen=True)
class MigrationBuildContext:
    app_config: InvokeAIAppConfig
    logger: Logger
    image_files: ImageFileStorageBase

    def get_dependency(self, name: str) -> Any:
        if name == "app_config":
            return self.app_config
        if name == "config":
            return self.app_config
        if name == "logger":
            return self.logger
        if name == "image_files":
            return self.image_files
        raise MigrationLoaderError(f"Migration builder requested unknown dependency '{name}'")


MigrationBuilder = Callable[..., Migration]


@dataclass(frozen=True)
class DiscoveredMigrationBuilder:
    module_name: str
    sort_key: tuple[int, int | str]
    expected_migration_id: str
    builder: MigrationBuilder


def discover_migration_builders(package_name: str = DEFAULT_MIGRATIONS_PACKAGE) -> list[DiscoveredMigrationBuilder]:
    try:
        package = importlib.import_module(package_name)
    except Exception as e:
        raise MigrationLoaderError(f"Unable to import migrations package '{package_name}': {e}") from e

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise MigrationLoaderError(f"Migrations package '{package_name}' is not a package")

    discovered: list[DiscoveredMigrationBuilder] = []
    for module_info in pkgutil.iter_modules(package_path):
        module_short_name = module_info.name
        if not module_short_name.startswith("migration_"):
            continue
        numeric_match = _MIGRATION_MODULE_RE.match(module_short_name)
        dated_match = _DATED_MIGRATION_MODULE_RE.match(module_short_name)
        if numeric_match is None and dated_match is None:
            raise MigrationLoaderError(f"Malformed migration module name '{module_short_name}'")

        if numeric_match is not None:
            sort_key: tuple[int, int | str] = (0, int(numeric_match.group(1)))
            expected_builder_name = f"build_migration_{numeric_match.group(1)}"
            expected_migration_id = f"migration_{numeric_match.group(1)}"
        else:
            sort_key = (1, module_short_name)
            expected_builder_name = "build_migration"
            expected_migration_id = module_short_name.removeprefix("migration_")

        full_module_name = f"{package_name}.{module_short_name}"
        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            raise MigrationLoaderError(f"Unable to import migration module '{full_module_name}': {e}") from e

        builder = _get_builder(module, expected_builder_name)
        discovered.append(DiscoveredMigrationBuilder(full_module_name, sort_key, expected_migration_id, builder))

    return sorted(discovered, key=lambda discovered_builder: discovered_builder.sort_key)


def build_migrations(context: MigrationBuildContext, package_name: str = DEFAULT_MIGRATIONS_PACKAGE) -> list[Migration]:
    migrations: list[Migration] = []
    for discovered_builder in discover_migration_builders(package_name):
        kwargs = _get_builder_kwargs(discovered_builder.builder, context, discovered_builder.module_name)
        migration = discovered_builder.builder(**kwargs)
        if not isinstance(migration, Migration):
            raise MigrationLoaderError(
                f"Migration builder '{discovered_builder.builder.__name__}' in "
                f"'{discovered_builder.module_name}' must return Migration"
            )
        if migration.id != discovered_builder.expected_migration_id:
            raise MigrationLoaderError(
                f"Migration builder '{discovered_builder.builder.__name__}' in "
                f"'{discovered_builder.module_name}' must return migration id "
                f"'{discovered_builder.expected_migration_id}', got '{migration.id}'"
            )
        migrations.append(migration)
    return migrations


def _get_builder(module: ModuleType, builder_name: str) -> MigrationBuilder:
    builder = getattr(module, builder_name, None)
    if builder is None:
        raise MigrationLoaderError(f"Migration module '{module.__name__}' must define '{builder_name}'")
    if not callable(builder):
        raise MigrationLoaderError(f"Migration builder '{module.__name__}.{builder_name}' is not callable")
    return builder


def _get_builder_kwargs(builder: MigrationBuilder, context: MigrationBuildContext, module_name: str) -> dict[str, Any]:
    signature = inspect.signature(builder)
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise MigrationLoaderError(
                f"Migration builder '{module_name}.{builder.__name__}' must not use *args or **kwargs"
            )
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise MigrationLoaderError(
                f"Migration builder '{module_name}.{builder.__name__}' must not use positional-only parameters"
            )
        kwargs[parameter.name] = context.get_dependency(parameter.name)
    return kwargs
