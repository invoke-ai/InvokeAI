"""Router-level tests for /api/v1/model_relationships.

Covers:
- Auth gating (CurrentUserOrDefault on read/batch, AdminUserOrDefault on add/remove).
- Bug regression: self-relationship checks must return 400 (not 500 — the previous
  broad `except Exception` swallowed the HTTPException and converted it).
- Service exception mapping: ValueError → 409 on add, 404 on remove.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.invoker import Invoker


REQ_BODY = {
    "model_key_1": "aa3b247f-90c9-4416-bfcd-aeaa57a5339e",
    "model_key_2": "ac32b914-10ab-496e-a24a-3068724b9c35",
}


# ----------------------------- Auth gating -----------------------------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/v1/model_relationships/i/some-key", None),
        ("POST", "/api/v1/model_relationships/", REQ_BODY),
        ("DELETE", "/api/v1/model_relationships/", REQ_BODY),
        ("POST", "/api/v1/model_relationships/batch", {"model_keys": ["a", "b"]}),
    ],
)
def test_routes_require_auth(
    enable_multiuser: Any, client: TestClient, method: str, path: str, body: dict | None
):
    r = client.request(method, path, json=body)
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_related_models_allowed_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    mock_invoker.services.model_relationships.get_related_model_keys = MagicMock(return_value=["k1"])
    r = client.get(
        "/api/v1/model_relationships/i/some-key",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    assert r.json() == ["k1"]


def test_batch_allowed_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    mock_invoker.services.model_relationships.get_related_model_keys = MagicMock(side_effect=lambda k: [f"r-{k}"])
    r = client.post(
        "/api/v1/model_relationships/batch",
        json={"model_keys": ["a", "b"]},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    assert set(r.json()) == {"r-a", "r-b"}


def test_add_forbidden_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    r = client.post(
        "/api/v1/model_relationships/",
        json=REQ_BODY,
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.model_relationships.add_model_relationship.assert_not_called()


def test_remove_forbidden_for_regular_user(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    r = client.request(
        "DELETE",
        "/api/v1/model_relationships/",
        json=REQ_BODY,
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.model_relationships.remove_model_relationship.assert_not_called()


def test_add_allowed_for_admin(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    r = client.post(
        "/api/v1/model_relationships/",
        json=REQ_BODY,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_204_NO_CONTENT
    mock_invoker.services.model_relationships.add_model_relationship.assert_called_once()


def test_remove_allowed_for_admin(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    r = client.request(
        "DELETE",
        "/api/v1/model_relationships/",
        json=REQ_BODY,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_204_NO_CONTENT
    mock_invoker.services.model_relationships.remove_model_relationship.assert_called_once()


# ----------------------------- Bug A regression: self-relationship → 400 -----------------------------


def test_add_self_relationship_returns_400_not_500(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    """Before the fix, the inner HTTPException(400) was caught by `except Exception`
    and re-raised as 500."""
    r = client.post(
        "/api/v1/model_relationships/",
        json={"model_key_1": "same-key", "model_key_2": "same-key"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.model_relationships.add_model_relationship.assert_not_called()


def test_remove_self_relationship_returns_400_not_500(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    r = client.request(
        "DELETE",
        "/api/v1/model_relationships/",
        json={"model_key_1": "same-key", "model_key_2": "same-key"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_400_BAD_REQUEST
    mock_invoker.services.model_relationships.remove_model_relationship.assert_not_called()


# ----------------------------- Service exception mapping -----------------------------


def test_add_value_error_returns_409(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    mock_invoker.services.model_relationships.add_model_relationship = MagicMock(
        side_effect=ValueError("relationship already exists")
    )
    r = client.post(
        "/api/v1/model_relationships/",
        json=REQ_BODY,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_409_CONFLICT


def test_remove_value_error_returns_404(
    client: TestClient, admin_token: str, mock_invoker: Invoker
):
    mock_invoker.services.model_relationships.remove_model_relationship = MagicMock(
        side_effect=ValueError("relationship not found")
    )
    r = client.request(
        "DELETE",
        "/api/v1/model_relationships/",
        json=REQ_BODY,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_404_NOT_FOUND
