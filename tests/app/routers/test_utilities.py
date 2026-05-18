"""Router-level tests for /api/v1/utilities.

Covers:
- Auth gating (CurrentUserOrDefault on all three utility routes).
- image-to-prompt: image read-access check must fire BEFORE the model is loaded,
  so non-owners can't probe stored images.
- image-to-prompt: a missing image surfaces as 404, not 500.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker


def _save_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=100,
        height=100,
        has_workflow=False,
        user_id=user_id,
    )


def _create_extra_user(mock_invoker: Invoker, email: str) -> str:
    from invokeai.app.services.users.users_common import UserCreateRequest

    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=email, password="TestPass123", is_admin=False)
    )
    return user.user_id


# ----------------------------- Auth gating -----------------------------


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/v1/utilities/dynamicprompts", {"prompt": "hi"}),
        ("/api/v1/utilities/expand-prompt", {"prompt": "hi", "model_key": "m"}),
        ("/api/v1/utilities/image-to-prompt", {"image_name": "img-1", "model_key": "m"}),
    ],
)
def test_routes_require_auth(enable_multiuser: Any, client: TestClient, mock_invoker: Invoker, path: str, body: dict):
    r = client.post(path, json=body)
    assert r.status_code == status.HTTP_401_UNAUTHORIZED
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_dynamicprompts_works_for_user(client: TestClient, user1_token: str):
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a {b|c}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert "prompts" in body


# ----------------------------- image_to_prompt: ownership / read-access -----------------------------


def test_image_to_prompt_forbidden_for_non_owner(
    client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
):
    """A second user must not be able to read a private image via image-to-prompt."""
    # Need to discover user1's id, then save an image under that id.
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "private-img.png", user1.user_id)

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "private-img.png", "model_key": "some-key"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN
    # The model must not have been loaded — ownership must fire BEFORE inference.
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_image_to_prompt_owner_reaches_model_load(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """The owner passes the read-access check and the model load is attempted.
    We force an UnknownModelException to keep the test light and assert 404."""
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "owned-img.png", user1.user_id)

    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no such model"))

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "owned-img.png", "model_key": "missing-model"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.model_manager.store.get_model.assert_called_once()


def test_image_to_prompt_admin_can_access_any_image(
    client: TestClient, admin_token: str, user1_token: str, mock_invoker: Invoker
):
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "user1-img.png", user1.user_id)

    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no model"))

    r = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "user1-img.png", "model_key": "x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    # Admin passes the read-access check; model loading then fails with 404.
    assert r.status_code == status.HTTP_404_NOT_FOUND
