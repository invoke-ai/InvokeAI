import locale
from enum import Enum
from importlib.metadata import distributions
from pathlib import Path as FilePath
from threading import Lock
from typing import Any

import torch
import yaml
from fastapi import Body, HTTPException, Path
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field, model_validator

from invokeai.app.api.auth_dependencies import AdminUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.config.config_default import (
    EXTERNAL_PROVIDER_CONFIG_FIELDS,
    IMAGE_SUBFOLDER_STRATEGY,
    DefaultInvokeAIAppConfig,
    InvokeAIAppConfig,
    get_config,
    load_and_migrate_config,
    load_external_api_keys,
)
from invokeai.app.services.external_generation.external_generation_common import ExternalProviderStatus
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.logging import logging
from invokeai.version import __version__


class LogLevel(int, Enum):
    NotSet = logging.NOTSET
    Debug = logging.DEBUG
    Info = logging.INFO
    Warning = logging.WARNING
    Error = logging.ERROR
    Critical = logging.CRITICAL


app_router = APIRouter(prefix="/v1/app", tags=["app"])


class AppVersion(BaseModel):
    """App Version Response"""

    version: str = Field(description="App version")


@app_router.get("/version", operation_id="app_version", status_code=200, response_model=AppVersion)
async def get_version() -> AppVersion:
    return AppVersion(version=__version__)


@app_router.get("/app_deps", operation_id="get_app_deps", status_code=200, response_model=dict[str, str])
async def get_app_deps() -> dict[str, str]:
    deps: dict[str, str] = {dist.metadata["Name"]: dist.version for dist in distributions()}
    try:
        cuda = getattr(getattr(torch, "version", None), "cuda", None) or "N/A"  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        cuda = "N/A"

    deps["CUDA"] = cuda

    sorted_deps = dict(sorted(deps.items(), key=lambda item: item[0].lower()))

    return sorted_deps


@app_router.get("/patchmatch_status", operation_id="get_patchmatch_status", status_code=200, response_model=bool)
async def get_patchmatch_status() -> bool:
    return PatchMatch.patchmatch_available()


class InvokeAIAppConfigWithSetFields(BaseModel):
    """InvokeAI App Config with model fields set"""

    set_fields: set[str] = Field(description="The set fields")
    config: InvokeAIAppConfig = Field(description="The InvokeAI App Config")


class ExternalProviderStatusModel(BaseModel):
    provider_id: str = Field(description="The external provider identifier")
    configured: bool = Field(description="Whether credentials are configured for the provider")
    message: str | None = Field(default=None, description="Optional provider status detail")


class ExternalProviderConfigUpdate(BaseModel):
    api_key: str | None = Field(default=None, description="API key for the external provider")
    base_url: str | None = Field(default=None, description="Optional base URL override for the provider")


class ExternalProviderConfigModel(BaseModel):
    provider_id: str = Field(description="The external provider identifier")
    api_key_configured: bool = Field(description="Whether an API key is configured")
    base_url: str | None = Field(default=None, description="Optional base URL override")


EXTERNAL_PROVIDER_FIELDS: dict[str, tuple[str, str]] = {
    "alibabacloud": ("external_alibabacloud_api_key", "external_alibabacloud_base_url"),
    "gemini": ("external_gemini_api_key", "external_gemini_base_url"),
    "openai": ("external_openai_api_key", "external_openai_base_url"),
    "seedream": ("external_seedream_api_key", "external_seedream_base_url"),
}
_EXTERNAL_PROVIDER_CONFIG_LOCK = Lock()


def _remove_nullable_default_from_schema(schema: dict[str, Any]) -> None:
    schema.pop("default", None)
    any_of = schema.pop("anyOf", None)
    if isinstance(any_of, list):
        non_null_schemas = [
            subschema for subschema in any_of if isinstance(subschema, dict) and subschema.get("type") != "null"
        ]
        if len(non_null_schemas) == 1:
            schema.update(non_null_schemas[0])


class UpdateAppGenerationSettingsRequest(BaseModel):
    """Writable generation-related app settings."""

    image_subfolder_strategy: IMAGE_SUBFOLDER_STRATEGY | None = Field(
        default=None,
        description="Strategy for organizing images into subfolders.",
        json_schema_extra=_remove_nullable_default_from_schema,
    )
    max_queue_history: int | None = Field(
        default=None,
        ge=0,
        description="Keep the last N completed, failed, and canceled queue items on startup. Set to 0 to prune all terminal items.",
    )

    @model_validator(mode="after")
    def validate_explicit_nulls(self) -> "UpdateAppGenerationSettingsRequest":
        if "image_subfolder_strategy" in self.model_fields_set and self.image_subfolder_strategy is None:
            raise ValueError("image_subfolder_strategy may not be null")
        return self


@app_router.get(
    "/runtime_config", operation_id="get_runtime_config", status_code=200, response_model=InvokeAIAppConfigWithSetFields
)
async def get_runtime_config() -> InvokeAIAppConfigWithSetFields:
    config = get_config()
    return InvokeAIAppConfigWithSetFields(set_fields=config.model_fields_set, config=config)


@app_router.patch(
    "/runtime_config",
    operation_id="update_runtime_config",
    status_code=200,
    response_model=InvokeAIAppConfigWithSetFields,
)
async def update_runtime_config(
    _: AdminUserOrDefault,
    changes: UpdateAppGenerationSettingsRequest = Body(description="Writable runtime configuration changes"),
) -> InvokeAIAppConfigWithSetFields:
    with _EXTERNAL_PROVIDER_CONFIG_LOCK:
        config = get_config()
        update_dict = changes.model_dump(exclude_unset=True)
        config.update_config(update_dict)

        if config.config_file_path.exists():
            persisted_config = load_and_migrate_config(config.config_file_path)
        else:
            persisted_config = DefaultInvokeAIAppConfig()

        persisted_config.update_config(update_dict)
        persisted_config.write_file(config.config_file_path)
        return InvokeAIAppConfigWithSetFields(set_fields=config.model_fields_set, config=config)


@app_router.get(
    "/external_providers/status",
    operation_id="get_external_provider_statuses",
    status_code=200,
    response_model=list[ExternalProviderStatusModel],
)
async def get_external_provider_statuses() -> list[ExternalProviderStatusModel]:
    statuses = ApiDependencies.invoker.services.external_generation.get_provider_statuses()
    return [status_to_model(status) for status in statuses.values()]


@app_router.get(
    "/external_providers/config",
    operation_id="get_external_provider_configs",
    status_code=200,
    response_model=list[ExternalProviderConfigModel],
)
async def get_external_provider_configs() -> list[ExternalProviderConfigModel]:
    config = get_config()
    return [_build_external_provider_config(provider_id, config) for provider_id in EXTERNAL_PROVIDER_FIELDS]


@app_router.post(
    "/external_providers/config/{provider_id}",
    operation_id="set_external_provider_config",
    status_code=200,
    response_model=ExternalProviderConfigModel,
)
async def set_external_provider_config(
    provider_id: str = Path(description="The external provider identifier"),
    update: ExternalProviderConfigUpdate = Body(description="External provider configuration settings"),
) -> ExternalProviderConfigModel:
    api_key_field, base_url_field = _get_external_provider_fields(provider_id)
    updates: dict[str, str | None] = {}

    if update.api_key is not None:
        api_key = update.api_key.strip()
        updates[api_key_field] = api_key or None
    if update.base_url is not None:
        base_url = update.base_url.strip()
        updates[base_url_field] = base_url or None

    if not updates:
        raise HTTPException(status_code=400, detail="No external provider config fields provided")

    api_key_removed = update.api_key is not None and updates.get(api_key_field) is None
    _apply_external_provider_update(updates)
    if api_key_removed:
        _remove_external_models_for_provider(provider_id)
    return _build_external_provider_config(provider_id, get_config())


@app_router.delete(
    "/external_providers/config/{provider_id}",
    operation_id="reset_external_provider_config",
    status_code=200,
    response_model=ExternalProviderConfigModel,
)
async def reset_external_provider_config(
    provider_id: str = Path(description="The external provider identifier"),
) -> ExternalProviderConfigModel:
    api_key_field, base_url_field = _get_external_provider_fields(provider_id)
    _apply_external_provider_update({api_key_field: None, base_url_field: None})
    _remove_external_models_for_provider(provider_id)
    return _build_external_provider_config(provider_id, get_config())


def status_to_model(status: ExternalProviderStatus) -> ExternalProviderStatusModel:
    return ExternalProviderStatusModel(
        provider_id=status.provider_id,
        configured=status.configured,
        message=status.message,
    )


def _get_external_provider_fields(provider_id: str) -> tuple[str, str]:
    if provider_id not in EXTERNAL_PROVIDER_FIELDS:
        raise HTTPException(status_code=404, detail=f"Unknown external provider '{provider_id}'")
    return EXTERNAL_PROVIDER_FIELDS[provider_id]


def _write_external_api_keys_file(api_keys_file_path: FilePath, api_keys: dict[str, str]) -> None:
    if not api_keys:
        if api_keys_file_path.exists():
            api_keys_file_path.unlink()
        return

    api_keys_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(api_keys_file_path, "w", encoding=locale.getpreferredencoding()) as api_keys_file:
        yaml.safe_dump(api_keys, api_keys_file, sort_keys=False)


def _apply_external_provider_update(updates: dict[str, str | None]) -> None:
    with _EXTERNAL_PROVIDER_CONFIG_LOCK:
        runtime_config = get_config()
        config_path = runtime_config.config_file_path
        api_keys_file_path = runtime_config.api_keys_file_path
        if config_path.exists():
            file_config = load_and_migrate_config(config_path)
        else:
            file_config = DefaultInvokeAIAppConfig()

        runtime_config.update_config(updates)
        provider_config_fields = set(EXTERNAL_PROVIDER_CONFIG_FIELDS)
        provider_updates = {field: value for field, value in updates.items() if field in provider_config_fields}
        non_provider_updates = {field: value for field, value in updates.items() if field not in provider_config_fields}

        if non_provider_updates:
            file_config.update_config(non_provider_updates)

        persisted_api_keys = load_external_api_keys(api_keys_file_path)
        for field_name in EXTERNAL_PROVIDER_CONFIG_FIELDS:
            file_value = getattr(file_config, field_name, None)
            if field_name not in persisted_api_keys and isinstance(file_value, str) and file_value.strip():
                persisted_api_keys[field_name] = file_value

        for field_name, value in provider_updates.items():
            if value is None:
                persisted_api_keys.pop(field_name, None)
            else:
                persisted_api_keys[field_name] = value

        _write_external_api_keys_file(api_keys_file_path, persisted_api_keys)

        for field_name in EXTERNAL_PROVIDER_CONFIG_FIELDS:
            setattr(file_config, field_name, None)

        file_config_to_write = type(file_config).model_validate(
            file_config.model_dump(exclude_unset=True, exclude_none=True)
        )
        file_config_to_write.write_file(config_path, as_example=False)


def _build_external_provider_config(provider_id: str, config: InvokeAIAppConfig) -> ExternalProviderConfigModel:
    api_key_field, base_url_field = _get_external_provider_fields(provider_id)
    return ExternalProviderConfigModel(
        provider_id=provider_id,
        api_key_configured=bool(getattr(config, api_key_field)),
        base_url=getattr(config, base_url_field),
    )


def _remove_external_models_for_provider(provider_id: str) -> None:
    model_manager = ApiDependencies.invoker.services.model_manager
    external_models = model_manager.store.search_by_attr(
        base_model=BaseModelType.External,
        model_type=ModelType.ExternalImageGenerator,
    )

    for model in external_models:
        if getattr(model, "provider_id", None) != provider_id:
            continue
        try:
            model_manager.install.delete(model.key)
        except UnknownModelException:
            logging.warning(f"External model key '{model.key}' was already removed while resetting '{provider_id}'")
        except Exception as error:
            logging.warning(f"Failed removing external model key '{model.key}' for '{provider_id}': {error}")


@app_router.get(
    "/logging",
    operation_id="get_log_level",
    responses={200: {"description": "The operation was successful"}},
    response_model=LogLevel,
)
async def get_log_level() -> LogLevel:
    """Returns the log level"""
    return LogLevel(ApiDependencies.invoker.services.logger.level)


@app_router.post(
    "/logging",
    operation_id="set_log_level",
    responses={200: {"description": "The operation was successful"}},
    response_model=LogLevel,
)
async def set_log_level(
    level: LogLevel = Body(description="New log verbosity level"),
) -> LogLevel:
    """Sets the log verbosity level"""
    ApiDependencies.invoker.services.logger.setLevel(level)
    return LogLevel(ApiDependencies.invoker.services.logger.level)


@app_router.delete(
    "/invocation_cache",
    operation_id="clear_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
async def clear_invocation_cache() -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.clear()


@app_router.put(
    "/invocation_cache/enable",
    operation_id="enable_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
async def enable_invocation_cache() -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.enable()


@app_router.put(
    "/invocation_cache/disable",
    operation_id="disable_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
async def disable_invocation_cache() -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.disable()


@app_router.get(
    "/invocation_cache/status",
    operation_id="get_invocation_cache_status",
    responses={200: {"model": InvocationCacheStatus}},
)
async def get_invocation_cache_status() -> InvocationCacheStatus:
    """Clears the invocation cache"""
    return ApiDependencies.invoker.services.invocation_cache.get_status()
