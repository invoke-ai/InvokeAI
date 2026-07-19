"""Tests for API-level authorization on model-manager and app-info read endpoints.

These cover the security fix for GH #9365: in multi-user mode a number of model-management and app-info read
routes carried no auth dependency at all, so an unauthenticated network attacker could reach them. The highest
impact was `GET /api/v2/models/scan_folder`, which enumerates an attacker-chosen filesystem path.
"""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.config.config_default import InvokeAIAppConfig, URLRegexTokenPair
from invokeai.app.services.invoker import Invoker
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
    ("get", "/api/v1/app/invocation_cache/status"),
    ("delete", "/api/v1/app/invocation_cache"),
    ("put", "/api/v1/app/invocation_cache/enable"),
    ("put", "/api/v1/app/invocation_cache/disable"),
]

# Routes that must additionally reject authenticated non-admin users.
ADMIN_ONLY_ROUTES = [
    ("get", "/api/v2/models/scan_folder?scan_path=/etc"),
    ("get", "/api/v2/models/missing"),
    ("get", "/api/v2/models/hugging_face?hugging_face_repo=x/y"),
    ("get", "/api/v2/models/starter_models"),
    ("get", "/api/v2/models/stats"),
    ("get", "/api/v2/models/hf_login"),
    ("get", "/api/v1/app/external_providers/config"),
    ("get", "/api/v1/app/logging"),
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


@pytest.mark.parametrize("scan_path", ["/nonexistent_xyz", "/etc/ssl", "/"])
def test_scan_folder_is_not_an_existence_oracle(
    scan_path: str, enable_multiuser: Any, client: TestClient, admin_token: str
) -> None:
    """Nonexistent, unreadable and readable-but-unscannable paths must be indistinguishable from the response.

    Previously these returned 400 / 500 / 200 respectively, letting a caller probe arbitrary paths.
    """
    response = client.get(f"/api/v2/models/scan_folder?scan_path={scan_path}", headers=_auth(admin_token))
    assert response.status_code in (status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST)
    if response.status_code == status.HTTP_400_BAD_REQUEST:
        assert response.json()["detail"] == f"The search path '{scan_path}' could not be scanned"


def test_regular_user_can_still_list_models(
    enable_multiuser: Any, client: TestClient, user1_token: str, mock_invoker: Invoker
) -> None:
    """Listing models is needed for ordinary generation, so it must stay available to non-admins."""
    # model_manager is a MagicMock from the enable_multiuser fixture; give the store a real, empty result set.
    mock_invoker.services.model_manager.store.search_by_attr.return_value = []

    response = client.get("/api/v2/models/", headers=_auth(user1_token))
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"models": []}


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
    enable_multiuser: Any, client: TestClient, user1_token: str, monkeypatch: Any
) -> None:
    """End-to-end: an authenticated non-admin gets the config, but with credentials masked."""
    config = InvokeAIAppConfig(
        external_openai_api_key="sk-super-secret",
        remote_api_tokens=[URLRegexTokenPair(url_regex="example.com", token="bearer-secret")],
    )
    monkeypatch.setattr("invokeai.app.api.routers.app_info.get_config", lambda: config)

    response = client.get("/api/v1/app/runtime_config", headers=_auth(user1_token))
    assert response.status_code == status.HTTP_200_OK
    assert "sk-super-secret" not in response.text
    assert "bearer-secret" not in response.text
