"""Tests for API-level authorization on model-manager and app-info read endpoints.

These cover the security fix for GH #9365: in multi-user mode a number of model-management and app-info read
routes carried no auth dependency at all, so an unauthenticated network attacker could reach them. The highest
impact was `GET /api/v2/models/scan_folder`, which enumerates an attacker-chosen filesystem path.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.config.config_default import InvokeAIAppConfig, URLRegexTokenPair
from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_SD1_Config
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelSourceType,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
)
from tests.app.routers.conftest import _auth

# (method, path) for routes that must reject unauthenticated callers in multiuser mode.
PROTECTED_ROUTES = [
    ("get", "/api/v2/models/scan_folder?scan_path=/etc"),
    ("get", "/api/v2/models/"),
    ("get", "/api/v2/models/missing"),
    ("get", "/api/v2/models/get_by_attrs?name=x&type=main&base=sd-1"),
    ("get", "/api/v2/models/get_by_hash?hash=x"),
    ("get", "/api/v2/models/i/some-key"),
    ("get", "/api/v2/models/hugging_face?hugging_face_repo=x/y"),
    ("get", "/api/v2/models/starter_models"),
    ("get", "/api/v2/models/stats"),
    ("get", "/api/v2/models/hf_login"),
    ("get", "/api/v1/app/runtime_config"),
    ("get", "/api/v1/app/app_deps"),
    ("get", "/api/v1/app/patchmatch_status"),
    ("get", "/api/v1/app/external_providers/status"),
    ("get", "/api/v1/app/external_providers/config"),
    ("get", "/api/v1/app/logging"),
    ("post", "/api/v1/app/logging"),
    ("get", "/api/v1/app/invocation_cache/status"),
    ("delete", "/api/v1/app/invocation_cache"),
    ("put", "/api/v1/app/invocation_cache/enable"),
    ("put", "/api/v1/app/invocation_cache/disable"),
    ("post", "/api/v1/images/"),
]

# Routes that must additionally reject authenticated non-admin users.
ADMIN_ONLY_ROUTES = [
    ("get", "/api/v2/models/scan_folder?scan_path=/etc"),
    ("get", "/api/v2/models/hugging_face?hugging_face_repo=x/y"),
    ("get", "/api/v2/models/starter_models"),
    ("get", "/api/v2/models/stats"),
    ("get", "/api/v2/models/hf_login"),
    ("get", "/api/v1/app/external_providers/config"),
    ("get", "/api/v1/app/runtime_config"),
    ("get", "/api/v1/app/logging"),
    ("post", "/api/v1/app/logging"),
    ("get", "/api/v1/app/invocation_cache/status"),
    ("delete", "/api/v1/app/invocation_cache"),
    ("put", "/api/v1/app/invocation_cache/enable"),
    ("put", "/api/v1/app/invocation_cache/disable"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED_ROUTES)
def test_route_rejects_unauthenticated(
    method: str, path: str, enable_multiuser: Any, setup_jwt_secret: None, client: TestClient
) -> None:
    """In multiuser mode, no Authorization header must yield 401 - never a 200 or a data-bearing error."""
    response = getattr(client, method)(path)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.parametrize(("method", "path"), ADMIN_ONLY_ROUTES)
def test_route_rejects_non_admin(
    method: str, path: str, enable_multiuser: Any, client: TestClient, user1_token: str
) -> None:
    """Model-management routes are administrator-only, per the documented multi-user model."""
    response = getattr(client, method)(path, headers=_auth(user1_token))
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_scan_folder_allows_admin(enable_multiuser: Any, client: TestClient, admin_token: str) -> None:
    """An admin still reaches the handler - 400 here is the scan failing, not the auth layer rejecting."""
    response = client.get(
        "/api/v2/models/scan_folder?scan_path=/nonexistent_xyz",
        headers=_auth(admin_token),
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_scan_folder_missing_param_is_a_clean_400(enable_multiuser: Any, client: TestClient, admin_token: str) -> None:
    """Omitting scan_path must not crash: the None default is guarded before pathlib.Path() sees it."""
    response = client.get("/api/v2/models/scan_folder", headers=_auth(admin_token))
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_scan_folder_is_not_an_existence_oracle(
    enable_multiuser: Any, client: TestClient, admin_token: str, tmp_path: Path
) -> None:
    """A nonexistent path and an existing non-directory must be indistinguishable from the response.

    Bounded to tmp_path fixtures - scanning real host paths ('/', '/etc') makes runtime depend on the host
    filesystem and can walk enormous trees on permissive CI runners.
    """
    not_a_dir = tmp_path / "weights.ckpt"
    not_a_dir.write_text("not a directory")

    for p in (tmp_path / "does_not_exist", not_a_dir):
        response = client.get(f"/api/v2/models/scan_folder?scan_path={p}", headers=_auth(admin_token))
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json()["detail"] == f"The search path '{p}' could not be scanned"


def test_scan_folder_mid_scan_failure_is_a_generic_500(
    enable_multiuser: Any, client: TestClient, admin_token: str, tmp_path: Path, monkeypatch: Any
) -> None:
    """A failure after path validation is a server fault: 500, with the details only in the server log.

    By that point is_dir() has already confirmed existence, so a distinct status leaks nothing new - and it
    preserves the caller-error vs server-fault distinction for the admin and for retrying clients.
    """

    class ExplodingSearch:
        def search(self, path: Path) -> None:
            raise PermissionError("permission denied inside the tree")

    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ModelSearch", ExplodingSearch)

    response = client.get(f"/api/v2/models/scan_folder?scan_path={tmp_path}", headers=_auth(admin_token))
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    # The response must not echo the exception - details go to the server log only.
    assert "PermissionError" not in response.text
    assert "permission denied" not in response.text


def test_create_image_upload_entry_requires_auth_before_the_501_stub(
    enable_multiuser: Any, client: TestClient, user1_token: str
) -> None:
    """The unimplemented upload-entry endpoint must authenticate first, so that implementing it later
    cannot silently ship an unauthenticated state-mutating route (the 401 side is in PROTECTED_ROUTES)."""
    response = client.post(
        "/api/v1/images/",
        headers=_auth(user1_token),
        json={"width": 512, "height": 512},
    )
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED


# (method, path) pairs that are deliberately public. Everything else must carry an auth dependency.
#
# - auth bootstrap: reachable pre-login by construction
# - binary <img src>/thumbnail routes: browsers cannot attach a Bearer header to <img> requests; closing
#   these needs a signed-URL or cookie scheme (tracked separately, see PR #9367)
# - version: intentionally public
# - docs/redoc: API documentation
PUBLIC_ROUTES = {
    ("GET", "/api/v1/app/version"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/setup"),
    ("GET", "/api/v1/auth/status"),
    ("GET", "/api/v1/images/i/{image_name}/full"),
    ("HEAD", "/api/v1/images/i/{image_name}/full"),
    ("GET", "/api/v1/images/i/{image_name}/thumbnail"),
    ("GET", "/api/v1/workflows/i/{workflow_id}/thumbnail"),
    ("GET", "/api/v2/models/i/{key}/image"),
    ("GET", "/docs"),
    ("GET", "/redoc"),
}


def _routes_without_auth() -> set[tuple[str, str]]:
    """Every (method, path) in the app whose dependency tree contains no auth dependency."""
    from fastapi.routing import APIRoute

    from invokeai.app.api import auth_dependencies
    from invokeai.app.api_app import app

    auth_functions = {auth_dependencies.get_current_user, auth_dependencies.get_current_user_or_default}

    def has_auth(dependant: Any) -> bool:
        if dependant.call in auth_functions:
            return True
        return any(has_auth(sub) for sub in dependant.dependencies)

    return {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute) and not has_auth(route.dependant)
        for method in route.methods
    }


def test_every_route_is_authenticated_or_explicitly_public() -> None:
    """Default-deny: a new route without an auth dependency must be a conscious, allowlisted decision.

    GH #9365 happened because auth was opt-in per route and nothing failed when a route lacked it. A
    hand-maintained list of protected routes cannot catch the next unauthenticated route; walking the app's
    actual dependency trees can.
    """
    unauthenticated = _routes_without_auth()

    unlisted = unauthenticated - PUBLIC_ROUTES
    assert not unlisted, (
        f"Routes without an auth dependency that are not on the public allowlist: {sorted(unlisted)}. "
        "Add an auth dependency (CurrentUserOrDefault / AdminUserOrDefault), or - only for routes that must "
        "be public - add them to PUBLIC_ROUTES with a justification."
    )

    # Keep the allowlist honest: an entry that gained auth (or disappeared) must be removed.
    stale = PUBLIC_ROUTES - unauthenticated
    assert not stale, f"PUBLIC_ROUTES entries that are no longer unauthenticated routes: {sorted(stale)}"


def test_regular_user_can_still_list_models(
    enable_multiuser: Any, client: TestClient, user1_token: str, mock_invoker: Invoker
) -> None:
    """Listing models is needed for ordinary generation, so it must stay available to non-admins."""
    # model_manager is a MagicMock from the enable_multiuser fixture; give the store a real, empty result set.
    mock_invoker.services.model_manager.store.search_by_attr.return_value = []

    response = client.get("/api/v2/models/", headers=_auth(user1_token))
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"models": []}


def _sd1_checkpoint(key: str, path: str) -> Main_Checkpoint_SD1_Config:
    return Main_Checkpoint_SD1_Config(
        key=key,
        path=path,
        name=key,
        format=ModelFormat.Checkpoint,
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        config_path="/tmp/foo.yaml",
        variant=ModelVariantType.Normal,
        hash="111222333444",
        file_size=8192,
        source=path,
        source_type=ModelSourceType.Path,
        prediction_type=SchedulerPredictionType.Epsilon,
    )


def test_non_admin_can_filter_out_missing_models(
    enable_multiuser: Any,
    client: TestClient,
    user1_token: str,
    mock_invoker: Invoker,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A non-admin must be able to read `/missing`, or missing models stay in their generation dropdowns.

    The frontend's model hooks (`modelsByType.ts`) subtract `/missing` from `/`. If `/missing` 403s the
    subtraction silently becomes a no-op and every missing model is offered for selection, failing only
    when execution tries to load its absent files.
    """
    (tmp_path / "present.ckpt").write_text("weights")
    present = _sd1_checkpoint("present-key", "present.ckpt")
    absent = _sd1_checkpoint("absent-key", "absent.ckpt")

    # models_path is a read-only derived property, so patch it on the class for the duration of the test.
    monkeypatch.setattr(InvokeAIAppConfig, "models_path", property(lambda _self: tmp_path))
    mock_invoker.services.model_images = MagicMock()
    mock_invoker.services.model_manager.store.all_models.return_value = [present, absent]
    mock_invoker.services.model_manager.store.search_by_attr.return_value = [present, absent]

    missing = client.get("/api/v2/models/missing", headers=_auth(user1_token))
    assert missing.status_code == status.HTTP_200_OK
    missing_keys = {m["key"] for m in missing.json()["models"]}
    assert missing_keys == {"absent-key"}

    all_models = client.get("/api/v2/models/", headers=_auth(user1_token))
    assert all_models.status_code == status.HTTP_200_OK

    # This mirrors what the frontend hooks do with the two responses.
    selectable = {m["key"] for m in all_models.json()["models"]} - missing_keys
    assert selectable == {"present-key"}


def test_runtime_config_redacts_secrets() -> None:
    """API keys and download bearer tokens must never be serialized to a client."""
    from invokeai.app.api.routers.app_info import REDACTED_SECRET, _redact_config_secrets

    config = InvokeAIAppConfig(
        external_openai_api_key="sk-super-secret",
        external_gemini_base_url="https://example.invalid",
        remote_api_tokens=[URLRegexTokenPair(url_regex="example.com", token="bearer-secret")],
    )
    redacted = _redact_config_secrets(config)

    assert redacted.external_openai_api_key == REDACTED_SECRET
    assert redacted.remote_api_tokens is not None
    assert redacted.remote_api_tokens[0].token == REDACTED_SECRET
    # Non-secret fields are untouched, and the original config is not mutated.
    assert redacted.remote_api_tokens[0].url_regex == "example.com"
    assert redacted.external_gemini_base_url == "https://example.invalid"
    assert config.external_openai_api_key == "sk-super-secret"
    assert config.remote_api_tokens is not None
    assert config.remote_api_tokens[0].token == "bearer-secret"


def test_runtime_config_response_has_no_secrets(
    enable_multiuser: Any, client: TestClient, admin_token: str, monkeypatch: Any
) -> None:
    """End-to-end: even an admin's browser never receives the raw credentials."""
    config = InvokeAIAppConfig(
        external_openai_api_key="sk-super-secret",
        remote_api_tokens=[URLRegexTokenPair(url_regex="example.com", token="bearer-secret")],
    )
    monkeypatch.setattr("invokeai.app.api.routers.app_info.get_config", lambda: config)

    response = client.get("/api/v1/app/runtime_config", headers=_auth(admin_token))
    assert response.status_code == status.HTTP_200_OK
    assert "sk-super-secret" not in response.text
    assert "bearer-secret" not in response.text


def test_runtime_config_patch_response_has_no_secrets(
    enable_multiuser: Any, client: TestClient, admin_token: str, monkeypatch: Any, tmp_path: Path
) -> None:
    """PATCH echoes the updated config back, so it must redact exactly as GET does.

    Otherwise an admin changing an unrelated setting pulls the raw credentials into the browser's
    RTK Query cache, defeating the server-side masking on GET.
    """
    config = InvokeAIAppConfig(
        external_openai_api_key="sk-super-secret",
        remote_api_tokens=[URLRegexTokenPair(url_regex="example.com", token="bearer-secret")],
    )
    config._config_file = tmp_path / "invokeai.yaml"
    monkeypatch.setattr("invokeai.app.api.routers.app_info.get_config", lambda: config)

    response = client.patch("/api/v1/app/runtime_config", json={"max_queue_history": 5}, headers=_auth(admin_token))
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["config"]["max_queue_history"] == 5
    assert "sk-super-secret" not in response.text
    assert "bearer-secret" not in response.text
    # The in-memory config itself must keep the real values - masking is response-only.
    assert config.external_openai_api_key == "sk-super-secret"
