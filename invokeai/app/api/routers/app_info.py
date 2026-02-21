from enum import Enum
from importlib.metadata import distributions

import torch
from fastapi import Body, HTTPException, Path
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.config.config_default import (
    DefaultInvokeAIAppConfig,
    InvokeAIAppConfig,
    get_config,
    load_and_migrate_config,
)
from invokeai.app.services.external_generation.external_generation_common import ExternalProviderStatus
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch
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
    "gemini": ("external_gemini_api_key", "external_gemini_base_url"),
    "openai": ("external_openai_api_key", "external_openai_base_url"),
}


@app_router.get(
    "/runtime_config", operation_id="get_runtime_config", status_code=200, response_model=InvokeAIAppConfigWithSetFields
)
async def get_runtime_config() -> InvokeAIAppConfigWithSetFields:
    config = get_config()
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

    _apply_external_provider_update(updates)
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


def _apply_external_provider_update(updates: dict[str, str | None]) -> None:
    runtime_config = get_config()
    config_path = runtime_config.config_file_path
    if config_path.exists():
        file_config = load_and_migrate_config(config_path)
    else:
        file_config = DefaultInvokeAIAppConfig()

    for config in (runtime_config, file_config):
        config.update_config(updates)
        for field_name, value in updates.items():
            if value is None:
                config.model_fields_set.discard(field_name)

    file_config.write_file(config_path, as_example=False)


def _build_external_provider_config(provider_id: str, config: InvokeAIAppConfig) -> ExternalProviderConfigModel:
    api_key_field, base_url_field = _get_external_provider_fields(provider_id)
    return ExternalProviderConfigModel(
        provider_id=provider_id,
        api_key_configured=bool(getattr(config, api_key_field)),
        base_url=getattr(config, base_url_field),
    )


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
