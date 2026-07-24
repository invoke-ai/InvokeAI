"""Router-level tests for /api/v1/utilities."""

from pathlib import Path
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


@pytest.fixture
def font_root(mock_invoker: Invoker, invokeai_root_dir: Path) -> Path:
    mock_invoker.services.configuration._root = invokeai_root_dir
    return invokeai_root_dir


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


def test_dynamicprompts_unknown_wildcard_returns_error_without_hanging(client: TestClient, user1_token: str):
    """An unknown wildcard used as a variant value would otherwise loop forever in the combinatorial generator.

    The endpoint must instead return promptly with a clear error and the original prompt echoed back.
    """
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__random__8chan|fenster|stuff}"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is not None
    assert "random" in body["error"]
    assert body["prompts"] == ["{__random__8chan|fenster|stuff}"]


def test_dynamicprompts_bare_unknown_wildcard_still_generates(client: TestClient, user1_token: str):
    """A wildcard used as plain literal text (not a variant value) does not hang and must not error."""
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a photo, __my_style__"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is None
    assert body["prompts"]  # non-empty
    assert all(p == "a photo, __my_style__" for p in body["prompts"])


def test_dynamicprompts_random_generator_ignores_unknown_wildcard(client: TestClient, user1_token: str):
    """The random generator handles unknown wildcards gracefully, so the guard must not fire for it."""
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__random__8chan|fenster|stuff}", "combinatorial": False, "seed": 0},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    assert body["error"] is None
    assert body["prompts"]  # non-empty; the random generator expanded the variant


# ----------------------------- image_to_prompt: ownership / read-access -----------------------------


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


def test_list_user_fonts_requires_auth(enable_multiuser: Any, font_root: Path, client: TestClient) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_user_font_file_requires_auth(enable_multiuser: Any, font_root: Path, client: TestClient) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts/MyFont.ttf")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_user_fonts_allows_authenticated_access(
    admin_token: str, client: TestClient, font_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fonts_dir = font_root / "fonts"
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
    font_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fonts_dir = font_root / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "BrokenFont.ttf").write_bytes(b"not-a-real-font")

    with caplog.at_level("WARNING"):
        response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["fonts"] == []
    assert "Skipping font file" in caplog.text


def test_get_user_font_file_rejects_symlink(
    admin_token: str, client: TestClient, font_root: Path, tmp_path: Path
) -> None:
    fonts_dir = font_root / "fonts"
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
    admin_token: str, client: TestClient, font_root: Path, tmp_path: Path
) -> None:
    fonts_dir = font_root / "fonts"
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
