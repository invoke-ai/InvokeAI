"""Tests for the utilities router."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def setup_jwt_secret():
    from invokeai.app.services.auth.token_service import set_jwt_secret

    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")


@pytest.fixture
def client(invokeai_root_dir: Path):
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


def setup_test_user(
    mock_invoker: Invoker,
    email: str,
    display_name: str,
    password: str = "TestPass123",
    is_admin: bool = False,
) -> str:
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name=display_name,
        password=password,
        is_admin=is_admin,
    )
    user = user_service.create(user_data)
    return user.user_id


def get_user_token(client: TestClient, email: str, password: str = "TestPass123") -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password, "remember_me": False},
    )
    assert response.status_code == 200
    return response.json()["token"]


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


@pytest.fixture
def patch_utilities_dependencies(monkeypatch: Any, mock_invoker: Invoker, invokeai_root_dir: Path):
    mock_invoker.services.configuration._root = invokeai_root_dir
    mock_invoker.services.model_manager = MagicMock()
    mock_invoker.services.model_manager.store.get_model = MagicMock()
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.utilities.ApiDependencies", mock_deps)
    yield mock_invoker


@pytest.fixture
def enable_multiuser(
    patch_utilities_dependencies: Invoker,
):
    patch_utilities_dependencies.services.configuration.multiuser = True
    yield patch_utilities_dependencies


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser: Invoker, client: TestClient):
    setup_test_user(enable_multiuser, "admin@test.com", "Test Admin", is_admin=True)
    return get_user_token(client, "admin@test.com")


@pytest.fixture
def user1_token(enable_multiuser: Invoker, client: TestClient, admin_token: str):
    setup_test_user(enable_multiuser, "user1@test.com", "Test User 1")
    return get_user_token(client, "user1@test.com")


@pytest.fixture
def user2_token(enable_multiuser: Invoker, client: TestClient, admin_token: str):
    setup_test_user(enable_multiuser, "user2@test.com", "Test User 2")
    return get_user_token(client, "user2@test.com")


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/v1/utilities/dynamicprompts", {"prompt": "hi"}),
        ("/api/v1/utilities/expand-prompt", {"prompt": "hi", "model_key": "m"}),
        ("/api/v1/utilities/image-to-prompt", {"image_name": "img-1", "model_key": "m"}),
    ],
)
def test_routes_require_auth(enable_multiuser: Any, client: TestClient, mock_invoker: Invoker, path: str, body: dict):
    response = client.post(path, json=body)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_dynamicprompts_works_for_user(client: TestClient, user1_token: str):
    response = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a {b|c}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert "prompts" in response.json()


def test_image_to_prompt_forbidden_for_non_owner(
    client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "private-img.png", user1.user_id)

    response = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "private-img.png", "model_key": "some-key"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.model_manager.store.get_model.assert_not_called()


def test_image_to_prompt_owner_reaches_model_load(client: TestClient, user1_token: str, mock_invoker: Invoker):
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "owned-img.png", user1.user_id)
    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no such model"))

    response = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "owned-img.png", "model_key": "missing-model"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.model_manager.store.get_model.assert_called_once()


def test_image_to_prompt_admin_can_access_any_image(
    client: TestClient, admin_token: str, user1_token: str, mock_invoker: Invoker
):
    from invokeai.app.services.model_records.model_records_base import UnknownModelException

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _save_image(mock_invoker, "user1-img.png", user1.user_id)
    mock_invoker.services.model_manager.store.get_model = MagicMock(side_effect=UnknownModelException("no model"))

    response = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "user1-img.png", "model_key": "x"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_list_user_fonts_requires_auth(enable_multiuser: Invoker, client: TestClient, invokeai_root_dir: Path) -> None:
    fonts_dir = invokeai_root_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_user_font_file_requires_auth(
    enable_multiuser: Invoker, client: TestClient, invokeai_root_dir: Path
) -> None:
    fonts_dir = invokeai_root_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts/MyFont.ttf")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_user_fonts_allows_authenticated_access(
    admin_token: str, client: TestClient, invokeai_root_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fonts_dir = invokeai_root_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")
    monkeypatch.setattr(
        "invokeai.app.api.routers.utilities._get_font_metadata",
        lambda _font_file: ("My Font", "My Font", 400, "normal"),
    )

    response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["fonts"]) == 1
    assert data["fonts"][0]["id"] == "user:MyFont.ttf"


def test_list_user_fonts_skips_malformed_fonts_and_logs_warning(
    admin_token: str,
    client: TestClient,
    invokeai_root_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fonts_dir = invokeai_root_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "BrokenFont.ttf").write_bytes(b"not-a-real-font")

    with caplog.at_level("WARNING"):
        response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["fonts"] == []
    assert "Skipping font file" in caplog.text


def test_get_user_font_file_rejects_symlink(
    admin_token: str, client: TestClient, invokeai_root_dir: Path, tmp_path: Path
) -> None:
    fonts_dir = invokeai_root_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.ttf"
    outside_file.write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked.ttf"

    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    response = client.get("/api/v1/utilities/fonts/linked.ttf", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_list_user_fonts_skips_symlinked_files(
    admin_token: str, client: TestClient, invokeai_root_dir: Path, tmp_path: Path
) -> None:
    fonts_dir = invokeai_root_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.ttf"
    outside_file.write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked.ttf"

    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["fonts"] == []
