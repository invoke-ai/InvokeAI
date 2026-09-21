import locale
import os
import re
from enum import Enum
from importlib.metadata import distributions
from pathlib import Path as FilePath
from threading import Lock
from typing import Any, Literal, Union

import torch
import yaml
from fastapi import Body, HTTPException, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
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
from invokeai.app.services.external_generation.providers.fal_catalog import (
    FalCatalogClient,
    FalEndpointKind,
    FalEndpointSchema,
    classify_endpoint,
)
from invokeai.app.services.invocation_cache.invocation_cache_common import InvocationCacheStatus
from invokeai.app.services.model_install.model_install_common import ModelInstallJob
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges, UnknownModelException
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalModelCapabilities,
    ExternalModelPanelSchema,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.devices import TorchDevice
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
def get_version() -> AppVersion:
    return AppVersion(version=__version__)


@app_router.get("/app_deps", operation_id="get_app_deps", status_code=200, response_model=dict[str, str])
def get_app_deps(current_user: CurrentUserOrDefault) -> dict[str, str]:
    deps: dict[str, str] = {dist.metadata["Name"]: dist.version for dist in distributions()}
    try:
        cuda = getattr(getattr(torch, "version", None), "cuda", None) or "N/A"  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        cuda = "N/A"

    deps["CUDA"] = cuda

    sorted_deps = dict(sorted(deps.items(), key=lambda item: item[0].lower()))

    return sorted_deps


@app_router.get("/patchmatch_status", operation_id="get_patchmatch_status", status_code=200, response_model=bool)
def get_patchmatch_status(current_user: CurrentUserOrDefault) -> bool:
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


class FalCatalogModelResponse(BaseModel):
    endpoint_id: str
    display_name: str
    description: str
    category: str
    kind: FalEndpointKind
    model_url: str | None
    thumbnail_url: str | None
    tags: list[str]
    installed: bool


class FalCatalogResponse(BaseModel):
    models: list[FalCatalogModelResponse]
    next_cursor: str | None
    has_more: bool


class FalEndpointSchemaResponse(BaseModel):
    endpoint_id: str
    kind: FalEndpointKind
    output_kind: FalEndpointKind
    category: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    common_fields: dict[str, str]
    public_properties: list[str]


class FalModelInstallRequest(BaseModel):
    endpoint_id: str = Field(min_length=1, max_length=512, description="fal.ai endpoint identifier")


EXTERNAL_PROVIDER_FIELDS: dict[str, tuple[str, str]] = {
    "fal": ("external_fal_api_key", "external_fal_base_url"),
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


_GENERATION_DEVICE_PATTERN = re.compile(r"^(cpu|mps|xpu(:\d+)?|cuda(:\d+)?)$")


class GenerationDeviceOption(BaseModel):
    """A device that may be selected for generation."""

    device: str = Field(description="The device identifier, e.g. 'cuda:0', 'mps', or 'cpu'")
    name: str = Field(description="Human-readable device name")


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
    generation_devices: Union[Literal["auto"], list[str]] | None = Field(
        default=None,
        description="Devices to use for parallel generation. `auto` uses every available GPU; provide an explicit list (e.g. `[cuda:0, cuda:1]`) to use specific devices. Takes effect after restarting InvokeAI.",
        json_schema_extra=_remove_nullable_default_from_schema,
    )

    @field_validator("generation_devices")
    @classmethod
    def validate_generation_devices(
        cls, v: Union[Literal["auto"], list[str], None]
    ) -> Union[Literal["auto"], list[str], None]:
        if v is None or v == "auto":
            return v
        # Mirror the InvokeAIAppConfig validator: an empty list would be rejected there anyway,
        # but catching it here turns an eventual 500 into a request-validation 422.
        if len(v) == 0:
            raise ValueError("generation_devices cannot be an empty list. Use 'auto' or a list of devices.")
        for device in v:
            if not _GENERATION_DEVICE_PATTERN.match(device):
                raise ValueError(
                    f"Invalid generation device '{device}'. Valid values are 'auto', 'cpu', 'mps', 'cuda', 'cuda:N', "
                    "'xpu', or 'xpu:N'."
                )
        return v

    @model_validator(mode="after")
    def validate_explicit_nulls(self) -> "UpdateAppGenerationSettingsRequest":
        if "image_subfolder_strategy" in self.model_fields_set and self.image_subfolder_strategy is None:
            raise ValueError("image_subfolder_strategy may not be null")
        if "generation_devices" in self.model_fields_set and self.generation_devices is None:
            raise ValueError("generation_devices may not be null")
        return self


REDACTED_SECRET = "**********"


def _redact_config_secrets(config: InvokeAIAppConfig) -> InvokeAIAppConfig:
    """Return a copy of the config with credential fields masked.

    The runtime config carries provider API keys and model-download bearer tokens. The route is admin-only, but no
    client - not even an admin's browser - has any use for the raw values; the UI only cares whether a credential is
    configured.

    NOTE: coverage is by convention, not automatic. Only `*_api_key` fields listed in
    EXTERNAL_PROVIDER_CONFIG_FIELDS plus `remote_api_tokens` and credentials embedded in
    `download_proxy` are masked. If you add a credential to InvokeAIAppConfig under any other
    name, you must extend this function or it will be served verbatim.
    """
    updates: dict[str, Any] = {}

    for field_name in EXTERNAL_PROVIDER_CONFIG_FIELDS:
        if not field_name.endswith("_api_key"):
            continue
        if getattr(config, field_name, None):
            updates[field_name] = REDACTED_SECRET

    if config.remote_api_tokens:
        updates["remote_api_tokens"] = [
            pair.model_copy(update={"token": REDACTED_SECRET}) for pair in config.remote_api_tokens
        ]

    if config.download_proxy and "@" in config.download_proxy:
        updates["download_proxy"] = REDACTED_SECRET

    return config.model_copy(update=updates) if updates else config


@app_router.get(
    "/generation_device_options",
    operation_id="get_generation_device_options",
    status_code=200,
    response_model=list[GenerationDeviceOption],
)
def get_generation_device_options(current_user: CurrentUserOrDefault) -> list[GenerationDeviceOption]:
    """List the devices available for generation, for use with the `generation_devices` setting."""
    options: list[GenerationDeviceOption] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            try:
                name = torch.cuda.get_device_name(index)
            except Exception:
                name = device
            options.append(GenerationDeviceOption(device=device, name=name))
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        for index in range(torch.xpu.device_count()):
            device = f"xpu:{index}"
            try:
                name = torch.xpu.get_device_name(index)
            except Exception:
                name = device
            options.append(GenerationDeviceOption(device=device, name=name))
    elif torch.backends.mps.is_available():
        options.append(GenerationDeviceOption(device="mps", name="Apple MPS"))
    else:
        options.append(GenerationDeviceOption(device="cpu", name="CPU"))
    return options


@app_router.get(
    "/runtime_config", operation_id="get_runtime_config", status_code=200, response_model=InvokeAIAppConfigWithSetFields
)
def get_runtime_config(current_admin: AdminUserOrDefault) -> InvokeAIAppConfigWithSetFields:
    config = get_config()
    return InvokeAIAppConfigWithSetFields(set_fields=config.model_fields_set, config=_redact_config_secrets(config))


@app_router.patch(
    "/runtime_config",
    operation_id="update_runtime_config",
    status_code=200,
    response_model=InvokeAIAppConfigWithSetFields,
)
def update_runtime_config(
    _: AdminUserOrDefault,
    changes: UpdateAppGenerationSettingsRequest = Body(description="Writable runtime configuration changes"),
) -> InvokeAIAppConfigWithSetFields:
    # The request model validates the *shape* of generation_devices; also verify the devices exist
    # on this machine before persisting, so we can't write a config that fails on the next startup
    # (e.g. 'cuda:99' on a 2-GPU box). Same resolution the startup path uses.
    if changes.generation_devices is not None:
        try:
            TorchDevice.get_generation_devices(changes.generation_devices)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
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
        return InvokeAIAppConfigWithSetFields(set_fields=config.model_fields_set, config=_redact_config_secrets(config))


@app_router.get(
    "/external_providers/status",
    operation_id="get_external_provider_statuses",
    status_code=200,
    response_model=list[ExternalProviderStatusModel],
)
def get_external_provider_statuses(current_user: CurrentUserOrDefault) -> list[ExternalProviderStatusModel]:
    statuses = ApiDependencies.invoker.services.external_generation.get_provider_statuses()
    return [status_to_model(status) for status in statuses.values()]


@app_router.get(
    "/external_providers/config",
    operation_id="get_external_provider_configs",
    status_code=200,
    response_model=list[ExternalProviderConfigModel],
)
def get_external_provider_configs(current_admin: AdminUserOrDefault) -> list[ExternalProviderConfigModel]:
    config = get_config()
    return [_build_external_provider_config(provider_id, config) for provider_id in EXTERNAL_PROVIDER_FIELDS]


@app_router.post(
    "/external_providers/config/{provider_id}",
    operation_id="set_external_provider_config",
    status_code=200,
    response_model=ExternalProviderConfigModel,
)
def set_external_provider_config(
    _: AdminUserOrDefault,
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
def reset_external_provider_config(
    _: AdminUserOrDefault,
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
    api_key_configured = bool(getattr(config, api_key_field))
    if provider_id == "fal" and not api_key_configured:
        api_key_configured = bool(os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY"))
    return ExternalProviderConfigModel(
        provider_id=provider_id,
        api_key_configured=api_key_configured,
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
def get_log_level(current_admin: AdminUserOrDefault) -> LogLevel:
    """Returns the log level"""
    return LogLevel(ApiDependencies.invoker.services.logger.level)


@app_router.post(
    "/logging",
    operation_id="set_log_level",
    responses={200: {"description": "The operation was successful"}},
    response_model=LogLevel,
)
def set_log_level(
    current_admin: AdminUserOrDefault,
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
def clear_invocation_cache(current_admin: AdminUserOrDefault) -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.clear()


@app_router.put(
    "/invocation_cache/enable",
    operation_id="enable_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
def enable_invocation_cache(current_admin: AdminUserOrDefault) -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.enable()


@app_router.put(
    "/invocation_cache/disable",
    operation_id="disable_invocation_cache",
    responses={200: {"description": "The operation was successful"}},
)
def disable_invocation_cache(current_admin: AdminUserOrDefault) -> None:
    """Clears the invocation cache"""
    ApiDependencies.invoker.services.invocation_cache.disable()


@app_router.get(
    "/invocation_cache/status",
    operation_id="get_invocation_cache_status",
    responses={200: {"model": InvocationCacheStatus}},
)
def get_invocation_cache_status(current_admin: AdminUserOrDefault) -> InvocationCacheStatus:
    """Clears the invocation cache"""
    return ApiDependencies.invoker.services.invocation_cache.get_status()


@app_router.get(
    "/external_providers/fal/models",
    operation_id="list_fal_models",
    status_code=200,
    response_model=FalCatalogResponse,
)
def list_fal_models(
    _: AdminUserOrDefault,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=4096),
    search: str | None = Query(default=None, max_length=200),
) -> FalCatalogResponse:
    client = _get_fal_catalog_client()
    try:
        page = client.list_models(limit=limit, cursor=cursor, search=search)
    except Exception as exc:
        _raise_fal_catalog_http_error(exc)
    installed = _get_installed_fal_models()
    return FalCatalogResponse(
        models=[
            FalCatalogModelResponse(
                endpoint_id=model.endpoint_id,
                display_name=model.display_name,
                description=model.description,
                category=model.category,
                kind=classify_endpoint(model.category, {}, endpoint_id=model.endpoint_id),
                model_url=model.model_url,
                thumbnail_url=model.thumbnail_url,
                tags=list(model.tags),
                installed=model.endpoint_id in installed,
            )
            for model in page.models
        ],
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )


@app_router.get(
    "/external_providers/fal/models/{endpoint_id:path}/schema",
    operation_id="get_fal_model_schema",
    status_code=200,
    response_model=FalEndpointSchemaResponse,
)
def get_fal_model_schema(
    _: AdminUserOrDefault,
    endpoint_id: str = Path(description="fal.ai endpoint identifier"),
) -> FalEndpointSchemaResponse:
    client = _get_fal_catalog_client()
    try:
        schema = client.get_schema(endpoint_id)
    except Exception as exc:
        _raise_fal_catalog_http_error(exc)
    return _fal_schema_to_response(schema)


@app_router.post(
    "/external_providers/fal/models/install",
    operation_id="install_fal_model",
    status_code=201,
    response_model=ModelInstallJob,
)
def install_fal_model(
    _: AdminUserOrDefault,
    request: FalModelInstallRequest,
) -> ModelInstallJob:
    client = _get_fal_catalog_client()
    try:
        schema = client.get_schema(request.endpoint_id)
    except Exception as exc:
        _raise_fal_catalog_http_error(exc)

    if schema.kind not in _FAL_NATIVE_IMAGE_KINDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"fal.ai endpoint '{request.endpoint_id}' is {schema.kind.value}; "
                "use the fal.ai generic media invocation for this endpoint"
            ),
        )

    config = ModelRecordChanges(
        name=f"fal.ai {request.endpoint_id}",
        description=f"Dynamic fal.ai endpoint ({schema.category}).",
        provider_id="fal",
        provider_model_id=request.endpoint_id,
        source_url=f"https://fal.ai/models/{request.endpoint_id}",
        capabilities=_fal_capabilities_from_schema(schema),
        panel_schema=_fal_panel_schema_from_schema(schema),
    )
    try:
        return ApiDependencies.invoker.services.model_manager.install.heuristic_import(
            source=f"external://fal/{request.endpoint_id}",
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Unable to install fal.ai model: {exc}") from exc


_FAL_NATIVE_IMAGE_KINDS = {
    FalEndpointKind.TEXT_TO_IMAGE,
    FalEndpointKind.IMAGE_TO_IMAGE,
    FalEndpointKind.INPAINT,
    FalEndpointKind.UPSCALE,
}


def _fal_panel_schema_from_schema(schema: FalEndpointSchema) -> ExternalModelPanelSchema:
    prompts = [{"name": "reference_images"}] if "reference_images" in schema.common_fields else []
    image_controls: list[dict[str, str]] = []
    if any(name in schema.common_fields for name in ("width", "height", "aspect_ratio", "image_size")):
        image_controls.append({"name": "dimensions"})
    if "seed" in schema.common_fields:
        image_controls.append({"name": "seed"})
    return ExternalModelPanelSchema(prompts=prompts, image=image_controls)


def _fal_capabilities_from_schema(schema: FalEndpointSchema) -> ExternalModelCapabilities:
    if schema.kind is FalEndpointKind.TEXT_TO_IMAGE:
        modes = ["txt2img"]
    elif schema.kind is FalEndpointKind.INPAINT:
        modes = ["inpaint"]
    else:
        modes = ["img2img"]

    properties = schema.input_schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    ratios = properties.get(schema.common_fields.get("aspect_ratio", ""), {})
    allowed_ratios = ratios.get("enum") if isinstance(ratios, dict) else None
    if not isinstance(allowed_ratios, list) or not all(isinstance(value, str) for value in allowed_ratios):
        allowed_ratios = None

    num_images_name = schema.common_fields.get("num_images", "")
    num_images_schema = properties.get(num_images_name, {})
    maximum = num_images_schema.get("maximum") if isinstance(num_images_schema, dict) else None
    max_images = maximum if isinstance(maximum, int) and maximum > 0 else None
    reference_name = schema.common_fields.get("reference_images", "")
    reference_schema = properties.get(reference_name, {})
    max_references = reference_schema.get("maxItems") if isinstance(reference_schema, dict) else None
    max_references = max_references if isinstance(max_references, int) and max_references > 0 else None

    return ExternalModelCapabilities(
        modes=modes,  # type: ignore[arg-type]
        supports_reference_images="reference_images" in schema.common_fields,
        max_reference_images=max_references,
        supports_negative_prompt="negative_prompt" in schema.common_fields,
        supports_seed="seed" in schema.common_fields,
        max_images_per_request=max_images,
        allowed_aspect_ratios=allowed_ratios,
        mask_format="binary" if "mask_image" in schema.common_fields else "none",
        input_image_required_for=[modes[0]] if modes[0] != "txt2img" else None,  # type: ignore[list-item]
    )


def _get_fal_catalog_client() -> FalCatalogClient:
    config = get_config()
    api_key = config.external_fal_api_key or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=409, detail="fal.ai API key is not configured")
    return FalCatalogClient(api_key)


def _get_installed_fal_models() -> set[str]:
    models = ApiDependencies.invoker.services.model_manager.store.search_by_attr(
        base_model=BaseModelType.External,
        model_type=ModelType.ExternalImageGenerator,
    )
    return {
        model.provider_model_id
        for model in models
        if isinstance(model, ExternalApiModelConfig) and model.provider_id == "fal" and model.provider_model_id
    }


def _fal_schema_to_response(schema: FalEndpointSchema) -> FalEndpointSchemaResponse:
    return FalEndpointSchemaResponse(
        endpoint_id=schema.endpoint_id,
        kind=schema.kind,
        output_kind=schema.output_kind,
        category=schema.category,
        input_schema=schema.input_schema,
        output_schema=schema.output_schema,
        common_fields=schema.common_fields,
        public_properties=list(schema.public_properties),
    )


def _raise_fal_catalog_http_error(exc: Exception) -> None:
    from invokeai.app.services.external_generation.errors import (
        ExternalProviderRateLimitError,
        ExternalProviderRequestError,
    )

    if isinstance(exc, ExternalProviderRateLimitError):
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if isinstance(exc, ExternalProviderRequestError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    raise HTTPException(status_code=502, detail="fal.ai catalog request failed") from exc
