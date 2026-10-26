"""Authenticated per-user remote-worker credential management.

Only a saved/not-saved indicator and email are returned; no passwords or JWTs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.invocations.remote_worker.credential_vault import (
    delete_credentials,
    get_saved_credentials,
    get_saved_settings,
    normalize_url,
    save_credentials,
    save_settings,
)
from invokeai.app.invocations.remote_worker.diffusers_transfer import (
    cancel_directory_install_job,
    get_directory_install_job,
    start_directory_install,
)
from invokeai.app.invocations.remote_worker.model_transfer import model_layout_signature
from invokeai.app.invocations.remote_worker.remote_client import RemoteConfig, RemoteInvokeClient, RemoteInvokeError

remote_workers_router = APIRouter(prefix="/v1/remote_workers", tags=["remote_workers"])


class RemoteWorkerCredentialRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=4096)
    remember_me: bool = True


class RemoteWorkerCredentialStatus(BaseModel):
    saved: bool
    email: str | None = None


def _status(user_id: str, url: str) -> RemoteWorkerCredentialStatus:
    record = get_saved_credentials(user_id, url)
    return RemoteWorkerCredentialStatus(
        saved=record is not None,
        email=str(record["email"]) if record and isinstance(record.get("email"), str) else None,
    )


class RemoteWorkerAvailability(BaseModel):
    status: Literal["online", "offline", "login_required"]


class RemoteWorkersSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = False
    dispatch_mode: Literal["distributed", "remote_only"] = Field(default="distributed", alias="dispatchMode")
    worker_urls: str = Field(default="", alias="workerUrls", max_length=32768)
    worker_names: dict[str, str] = Field(default_factory=dict, alias="workerNames")
    disabled_worker_urls: list[str] = Field(default_factory=list, alias="disabledWorkerUrls")
    auto_transfer_missing_models: bool = Field(default=True, alias="autoTransferMissingModels")
    keep_remote_copies: bool = Field(default=False, alias="keepRemoteCopies")
    model_transfer_host: str = Field(default="", alias="modelTransferHost", max_length=2048)


@remote_workers_router.get("/settings", response_model=RemoteWorkersSettings)
def get_remote_worker_settings(current_user: CurrentUserOrDefault) -> RemoteWorkersSettings:
    saved = get_saved_settings(current_user.user_id)
    return RemoteWorkersSettings.model_validate(saved) if saved is not None else RemoteWorkersSettings()


@remote_workers_router.put("/settings", response_model=RemoteWorkersSettings)
def put_remote_worker_settings(
    current_user: CurrentUserOrDefault,
    body: RemoteWorkersSettings,
) -> RemoteWorkersSettings:
    save_settings(current_user.user_id, body.model_dump(by_alias=True))
    return body


@remote_workers_router.get("/status", response_model=RemoteWorkerAvailability)
def get_remote_worker_status(
    current_user: CurrentUserOrDefault,
    url: str = Query(min_length=1, max_length=2048),
) -> RemoteWorkerAvailability:
    """Probe this user's ability to reach a configured worker using existing InvokeAI APIs.

    Keep the probe short; credentials stay in the primary's per-user vault.
    Do not mistake a reachable worker with rejected credentials for an offline host.
    """
    try:
        normalized = normalize_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    client = RemoteInvokeClient(
        RemoteConfig.from_environment(base_url=normalized, verify_ssl=False, user_id=current_user.user_id),
        request_timeout_seconds=2.5,
    )
    try:
        client.get_current_item()
        return RemoteWorkerAvailability(status="online")
    except RemoteInvokeError as exc:
        message = str(exc).lower()
        if any(
            marker in message
            for marker in (
                "requires login",
                "login failed",
                "initial admin setup",
                "http 401",
                "http 403",
                "credentials file",
            )
        ):
            return RemoteWorkerAvailability(status="login_required")
        return RemoteWorkerAvailability(status="offline")
    except Exception:
        # An unreachable worker should never make the primary's status API fail.
        return RemoteWorkerAvailability(status="offline")


@remote_workers_router.get("/credentials", response_model=RemoteWorkerCredentialStatus)
def get_remote_worker_credentials_status(
    current_user: CurrentUserOrDefault,
    url: str = Query(min_length=1, max_length=2048),
) -> RemoteWorkerCredentialStatus:
    try:
        return _status(current_user.user_id, normalize_url(url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@remote_workers_router.put("/credentials", response_model=RemoteWorkerCredentialStatus)
def put_remote_worker_credentials(
    current_user: CurrentUserOrDefault,
    body: RemoteWorkerCredentialRequest,
) -> RemoteWorkerCredentialStatus:
    try:
        save_credentials(current_user.user_id, body.url, body.email, body.password, body.remember_me)
        return _status(current_user.user_id, body.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@remote_workers_router.delete("/credentials", response_model=RemoteWorkerCredentialStatus)
def remove_remote_worker_credentials(
    current_user: CurrentUserOrDefault,
    url: str = Query(min_length=1, max_length=2048),
) -> RemoteWorkerCredentialStatus:
    try:
        delete_credentials(current_user.user_id, url)
        return RemoteWorkerCredentialStatus(saved=False)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class RemoteModelLayout(BaseModel):
    kind: Literal["file", "directory"]
    signature: str


@remote_workers_router.get("/models/{key}/layout", response_model=RemoteModelLayout)
def get_remote_model_layout(current_user: CurrentUserOrDefault, key: str) -> RemoteModelLayout:
    """Return a non-secret signature of one registered model's file layout."""
    services = ApiDependencies.invoker.services
    try:
        config = services.model_manager.store.get_model(key)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Model not found") from exc

    model_path = Path(str(getattr(config, "path", "") or ""))
    if not model_path.is_absolute():
        model_path = Path(services.configuration.models_path) / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model files not found")

    kind, signature = model_layout_signature(model_path)
    return RemoteModelLayout(kind=kind, signature=signature)


@remote_workers_router.post("/diffusers/install")
def install_remote_directory(current_admin: AdminUserOrDefault, body: dict[str, Any]) -> dict[str, Any]:
    """Admin-only receiver for short-lived, manifest-verified model transfers."""
    try:
        return start_directory_install(body, ApiDependencies.invoker.services)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@remote_workers_router.get("/diffusers/install/{job_id}")
def get_remote_directory_install(current_admin: AdminUserOrDefault, job_id: int) -> dict[str, Any]:
    """Poll a directory download and normal model installation."""
    job = get_directory_install_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Directory transfer job not found")
    return job


@remote_workers_router.delete("/diffusers/install/{job_id}")
def cancel_remote_directory_install(current_admin: AdminUserOrDefault, job_id: int) -> dict[str, Any]:
    """Cancel only this temporary download, preserving completed models."""
    job = cancel_directory_install_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Directory transfer job not found")
    return job
