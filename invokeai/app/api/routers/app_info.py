import csv
import io
import locale
import subprocess
from enum import Enum
from importlib.metadata import distributions
from pathlib import Path as FilePath
from threading import Lock
from typing import Any

import psutil
import torch
import yaml
from fastapi import Body, HTTPException, Path
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field, model_validator

from invokeai.app.invocations.model import ModelIdentifierField
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
from invokeai.app.util.t5_model_identifier import preprocess_t5_encoder_model_identifier
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch
from invokeai.backend.model_manager.load.model_cache.model_cache import get_model_cache_key
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.model_manager.taxonomy import SubModelType
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
    try:
        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        cuda_device_count = 0
    deps["CUDA Devices"] = str(cuda_device_count)
    for device_index in range(cuda_device_count):
        try:
            deps[f"CUDA Device {device_index}"] = torch.cuda.get_device_name(device_index)
        except Exception:
            deps[f"CUDA Device {device_index}"] = "Unknown CUDA device"

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
    model_cache_keep_alive_min: float | None = Field(
        default=None,
        ge=0,
        description="How long to keep unlocked models in cache after last use, in minutes. 0 keeps models indefinitely.",
    )
    use_second_gpu_for_text_encoder: bool | None = Field(
        default=None,
        description="Run text encoder models on the CUDA device that is not the main execution device when at least two CUDA GPUs are available.",
    )

    @model_validator(mode="after")
    def validate_explicit_nulls(self) -> "UpdateAppGenerationSettingsRequest":
        if "image_subfolder_strategy" in self.model_fields_set and self.image_subfolder_strategy is None:
            raise ValueError("image_subfolder_strategy may not be null")
        return self


class SyncTextEncoderCacheRequest(BaseModel):
    """Request to actively sync selected text encoder cache entries with the second-GPU toggle state."""

    enabled: bool = Field(description="Whether second-GPU text encoder mode is enabled.")
    text_encoder_models: list[ModelIdentifierField] = Field(
        default_factory=list,
        description="Selected text encoder models to unload or prewarm.",
    )


class SyncTextEncoderCacheResponse(BaseModel):
    """Text encoder cache sync result."""

    dropped: int = Field(description="Number of cache entries immediately dropped.")
    loaded: int = Field(description="Number of selected encoder entries loaded onto their target device.")
    status: "TextEncoderCacheStatusResponse" = Field(description="Text encoder cache status after sync.")


class TextEncoderCacheModelStatus(BaseModel):
    """Status for one selected text encoder cache entry."""

    key: str = Field(description="Model key.")
    name: str = Field(description="Model name.")
    cache_key: str = Field(description="Resolved cache key.")
    loaded: bool = Field(description="Whether the cache entry exists and has weights on its execution device.")
    device: str | None = Field(default=None, description="Execution device for the cache entry.")
    vram_gb: float = Field(description="Estimated model VRAM resident size in GB.")
    total_gb: float = Field(description="Estimated model size in GB.")


class CudaDeviceStatus(BaseModel):
    """CUDA device memory status."""

    index: int = Field(description="CUDA device index.")
    name: str = Field(description="CUDA device name.")
    used_gb: float = Field(description="Total device memory used in GB, including non-InvokeAI processes.")
    invoke_cache_gb: float = Field(description="InvokeAI model cache memory used on this device in GB.")
    total_gb: float = Field(description="Total device memory in GB.")


class TextEncoderCacheStatusResponse(BaseModel):
    """Selected text encoder cache and CUDA memory status."""

    models: list[TextEncoderCacheModelStatus] = Field(description="Selected text encoder cache statuses.")
    cuda_devices: list[CudaDeviceStatus] = Field(description="CUDA memory status.")


class SystemGpuStatus(BaseModel):
    """Basic GPU status."""

    index: int = Field(description="GPU device index.")
    name: str = Field(description="GPU device name.")
    utilization_percent: float | None = Field(default=None, description="GPU utilization percent.")
    loaded_gb: float = Field(description="GPU memory used in GB.")
    total_gb: float = Field(description="Total GPU memory in GB.")


class SystemStatusResponse(BaseModel):
    """Basic system status."""

    cpu_percent: float = Field(description="CPU utilization percent.")
    cpu_frequency_ghz: float | None = Field(default=None, description="Current CPU frequency in GHz.")
    memory_used_gb: float = Field(description="System memory used in GB.")
    memory_total_gb: float = Field(description="Total system memory in GB.")
    memory_percent: float = Field(description="System memory utilization percent.")
    gpus: list[SystemGpuStatus] = Field(description="GPU statuses.")


_PREWARM_STANDALONE_TEXT_ENCODER_TYPES = {
    ModelType.CLIPEmbed,
    ModelType.Qwen3Encoder,
    ModelType.QwenVLEncoder,
    ModelType.T5Encoder,
}


def _normalize_text_encoder_identifier(model: ModelIdentifierField) -> ModelIdentifierField:
    if model.type == ModelType.T5Encoder:
        return preprocess_t5_encoder_model_identifier(model)
    if model.submodel_type is None and (
        model.type in _PREWARM_STANDALONE_TEXT_ENCODER_TYPES or model.type == ModelType.Main
    ):
        return model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
    return model


def _get_cuda_device_statuses() -> list[CudaDeviceStatus]:
    if not torch.cuda.is_available():
        return []
    ram_cache = ApiDependencies.invoker.services.model_manager.load.ram_cache
    invoke_cache_usage = ram_cache.get_cuda_cache_usage_bytes()
    statuses: list[CudaDeviceStatus] = []
    for device_index in range(torch.cuda.device_count()):
        device = torch.device("cuda", device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_bytes = total_bytes - free_bytes
        statuses.append(
            CudaDeviceStatus(
                index=device_index,
                name=torch.cuda.get_device_name(device),
                used_gb=round(used_bytes / (1024**3), 2),
                invoke_cache_gb=round(invoke_cache_usage.get(device_index, 0) / (1024**3), 2),
                total_gb=round(total_bytes / (1024**3), 2),
            )
        )
    return statuses


def _get_nvidia_smi_statuses() -> dict[int, tuple[float | None, float, float]]:
    """Return GPU utilization percent, memory used MB, and memory total MB keyed by device index."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    if result.returncode != 0:
        return {}

    statuses: dict[int, tuple[float | None, float, float]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            utilization_percent = float(parts[1])
            memory_used_mb = float(parts[2])
            memory_total_mb = float(parts[3])
        except ValueError:
            continue
        statuses[index] = (utilization_percent, memory_used_mb, memory_total_mb)
    return statuses


def _get_system_gpu_statuses() -> list[SystemGpuStatus]:
    if not torch.cuda.is_available():
        return []

    nvidia_smi_statuses = _get_nvidia_smi_statuses()
    statuses: list[SystemGpuStatus] = []
    for device_index in range(torch.cuda.device_count()):
        device = torch.device("cuda", device_index)
        nvidia_smi_status = nvidia_smi_statuses.get(device_index)
        if nvidia_smi_status is not None:
            utilization_percent, memory_used_mb, memory_total_mb = nvidia_smi_status
            loaded_gb = memory_used_mb / 1024
            total_gb = memory_total_mb / 1024
        else:
            utilization_percent = None
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            loaded_gb = (total_bytes - free_bytes) / (1024**3)
            total_gb = total_bytes / (1024**3)

        statuses.append(
            SystemGpuStatus(
                index=device_index,
                name=torch.cuda.get_device_name(device),
                utilization_percent=(
                    None if utilization_percent is None else round(utilization_percent, 1)
                ),
                loaded_gb=round(loaded_gb, 1),
                total_gb=round(total_gb, 1),
            )
        )
    return statuses


def _get_windows_task_manager_cpu_status() -> tuple[float, float | None] | None:
    """Get CPU status using the same Windows performance counters Task Manager uses."""
    try:
        result = subprocess.run(
            [
                "typeperf",
                r"\Processor Information(_Total)\% Processor Utility",
                r"\Processor Information(_Total)\% Processor Performance",
                "-sc",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    reader = csv.reader(io.StringIO(result.stdout))
    for row in reader:
        if len(row) < 3 or row[0].startswith("(PDH-CSV"):
            continue
        try:
            cpu_percent = float(row[1])
            processor_performance = float(row[2])
        except ValueError:
            continue

        cpu_frequency = psutil.cpu_freq()
        base_frequency_mhz = None if cpu_frequency is None else cpu_frequency.max or cpu_frequency.current
        cpu_frequency_ghz = (
            None if base_frequency_mhz is None else round((base_frequency_mhz * processor_performance / 100) / 1000, 2)
        )
        return round(cpu_percent, 1), cpu_frequency_ghz

    return None


def _get_system_status() -> SystemStatusResponse:
    memory = psutil.virtual_memory()
    cpu_status = _get_windows_task_manager_cpu_status()
    if cpu_status is None:
        cpu_frequency = psutil.cpu_freq()
        cpu_percent = round(psutil.cpu_percent(interval=0.2), 1)
        cpu_frequency_ghz = None if cpu_frequency is None else round(cpu_frequency.current / 1000, 2)
    else:
        cpu_percent, cpu_frequency_ghz = cpu_status

    return SystemStatusResponse(
        cpu_percent=cpu_percent,
        cpu_frequency_ghz=cpu_frequency_ghz,
        memory_used_gb=round((memory.total - memory.available) / (1024**3), 1),
        memory_total_gb=round(memory.total / (1024**3), 1),
        memory_percent=round(memory.percent, 1),
        gpus=_get_system_gpu_statuses(),
    )


def _get_text_encoder_cache_status(models: list[ModelIdentifierField]) -> TextEncoderCacheStatusResponse:
    ram_cache = ApiDependencies.invoker.services.model_manager.load.ram_cache
    statuses: list[TextEncoderCacheModelStatus] = []
    normalized_models = [_normalize_text_encoder_identifier(model) for model in models]
    for model in normalized_models:
        cache_key = get_model_cache_key(model.key, model.submodel_type)
        snapshot = ram_cache.get_cache_entry_snapshot(cache_key)
        statuses.append(
            TextEncoderCacheModelStatus(
                key=model.key,
                name=model.name,
                cache_key=cache_key,
                loaded=snapshot is not None and snapshot.current_vram_bytes > 0,
                device=snapshot.compute_device if snapshot is not None else None,
                vram_gb=round((snapshot.current_vram_bytes if snapshot is not None else 0) / (1024**3), 2),
                total_gb=round((snapshot.total_bytes if snapshot is not None else 0) / (1024**3), 2),
            )
        )
    return TextEncoderCacheStatusResponse(models=statuses, cuda_devices=_get_cuda_device_statuses())


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
        if "model_cache_keep_alive_min" in update_dict:
            ApiDependencies.invoker.services.model_manager.load.ram_cache.set_keep_alive_minutes(
                config.model_cache_keep_alive_min
            )
        if update_dict.get("use_second_gpu_for_text_encoder") is False:
            ApiDependencies.invoker.services.model_manager.load.ram_cache.drop_cuda_entries_except(
                keep_execution_device=TorchDevice.choose_torch_device()
            )

        if config.config_file_path.exists():
            persisted_config = load_and_migrate_config(config.config_file_path)
        else:
            persisted_config = DefaultInvokeAIAppConfig()

        persisted_config.update_config(update_dict)
        persisted_config.write_file(config.config_file_path)
        return InvokeAIAppConfigWithSetFields(set_fields=config.model_fields_set, config=config)


@app_router.post(
    "/sync_text_encoder_cache",
    operation_id="sync_text_encoder_cache",
    status_code=200,
    response_model=SyncTextEncoderCacheResponse,
)
async def sync_text_encoder_cache(
    _: AdminUserOrDefault,
    request: SyncTextEncoderCacheRequest = Body(description="Selected text encoder cache sync request"),
) -> SyncTextEncoderCacheResponse:
    ram_cache = ApiDependencies.invoker.services.model_manager.load.ram_cache
    dropped = 0
    loaded = 0

    normalized_models = [_normalize_text_encoder_identifier(model) for model in request.text_encoder_models]
    for model in normalized_models:
        dropped += ram_cache.drop_cache_key(get_model_cache_key(model.key, model.submodel_type))

    if not request.enabled:
        dropped += ram_cache.drop_cuda_entries_except(keep_execution_device=TorchDevice.choose_torch_device())
        return SyncTextEncoderCacheResponse(
            dropped=dropped, loaded=loaded, status=_get_text_encoder_cache_status(request.text_encoder_models)
        )

    for model in normalized_models:
        try:
            config = ApiDependencies.invoker.services.model_manager.store.get_model(model.key)
            loaded_model = ApiDependencies.invoker.services.model_manager.load.load_model(config, model.submodel_type)
            with loaded_model.model_on_device():
                pass
            loaded += 1
        except UnknownModelException:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model.key}")

    return SyncTextEncoderCacheResponse(
        dropped=dropped, loaded=loaded, status=_get_text_encoder_cache_status(request.text_encoder_models)
    )


@app_router.post(
    "/text_encoder_cache_status",
    operation_id="get_text_encoder_cache_status",
    status_code=200,
    response_model=TextEncoderCacheStatusResponse,
)
async def get_text_encoder_cache_status(
    _: AdminUserOrDefault,
    request: SyncTextEncoderCacheRequest = Body(description="Selected text encoder cache status request"),
) -> TextEncoderCacheStatusResponse:
    return _get_text_encoder_cache_status(request.text_encoder_models)


@app_router.get(
    "/system_status",
    operation_id="get_system_status",
    status_code=200,
    response_model=SystemStatusResponse,
)
async def get_system_status(_: AdminUserOrDefault) -> SystemStatusResponse:
    return _get_system_status()


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
async def reset_external_provider_config(
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
