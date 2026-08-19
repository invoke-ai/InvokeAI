"""The last-admin invariant, at the level the user sees it.

`tests/app/services/users/test_last_admin_invariant.py` covers the guard itself, including
its behaviour under concurrency. These pin the HTTP contract: the service raises a
`LastAdministratorError`, and because that subclasses `ValueError` the existing handlers in
`auth.py` turn it into a 400 rather than letting it escape as a 500.

`PATCH /auth/users/{id}` is the case that had no guard at all before this change — only
`delete_user` checked, so demoting or deactivating the sole administrator succeeded and left
the instance with none.
"""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.invoker import Invoker


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _admin_id(mock_invoker: Invoker) -> str:
    admin = mock_invoker.services.users.get_by_email("admin@test.com")
    assert admin is not None
    return admin.user_id


def test_demoting_the_last_admin_returns_400(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    user_id = _admin_id(mock_invoker)

    r = client.patch(f"/api/v1/auth/users/{user_id}", headers=_auth(admin_token), json={"is_admin": False})

    assert r.status_code == status.HTTP_400_BAD_REQUEST, r.text
    assert mock_invoker.services.users.count_admins() == 1


def test_deactivating_the_last_admin_returns_400(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    user_id = _admin_id(mock_invoker)

    r = client.patch(f"/api/v1/auth/users/{user_id}", headers=_auth(admin_token), json={"is_active": False})

    assert r.status_code == status.HTTP_400_BAD_REQUEST, r.text
    assert mock_invoker.services.users.count_admins() == 1


def test_deleting_the_last_admin_returns_400(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    user_id = _admin_id(mock_invoker)

    r = client.delete(f"/api/v1/auth/users/{user_id}", headers=_auth(admin_token))

    assert r.status_code == status.HTTP_400_BAD_REQUEST, r.text
    assert mock_invoker.services.users.count_admins() == 1


def test_renaming_the_last_admin_still_succeeds(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    """The guard must not turn into a blanket lock on the last admin's record."""
    user_id = _admin_id(mock_invoker)

    r = client.patch(f"/api/v1/auth/users/{user_id}", headers=_auth(admin_token), json={"display_name": "Renamed"})

    assert r.status_code == status.HTTP_200_OK, r.text
    assert r.json()["display_name"] == "Renamed"


def test_demoting_an_admin_when_another_exists_succeeds(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    from invokeai.app.services.users.users_common import UserCreateRequest

    second = mock_invoker.services.users.create(
        UserCreateRequest(email="admin2@test.com", display_name="Second", password="TestPass123", is_admin=True)
    )

    r = client.patch(f"/api/v1/auth/users/{second.user_id}", headers=_auth(admin_token), json={"is_admin": False})

    assert r.status_code == status.HTTP_200_OK, r.text
    assert r.json()["is_admin"] is False
    assert mock_invoker.services.users.count_admins() == 1


def test_updating_an_unknown_user_returns_404(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    """`get_user` and `delete_user` 404 for an unknown id; this endpoint documents the same
    contract but used to fall through to the service's "User ... not found" as a 400."""
    r = client.patch("/api/v1/auth/users/does-not-exist", headers=_auth(admin_token), json={"display_name": "x"})

    assert r.status_code == status.HTTP_404_NOT_FOUND, r.text


def test_promoting_the_system_user_cannot_launder_away_the_last_admin(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    """The system row is active but has an empty password hash, so it can never log in.
    Promoting it would raise `count_admins()` to 2 — enough for the last-admin guard to
    allow the real administrator to be demoted, leaving nobody able to administer the
    instance and `/auth/setup` still closed."""
    users = mock_invoker.services.users
    admin_id = _admin_id(mock_invoker)

    promote = client.patch("/api/v1/auth/users/system", headers=_auth(admin_token), json={"is_admin": True})

    assert promote.status_code == status.HTTP_400_BAD_REQUEST, promote.text
    assert users.count_admins() == 1

    demote = client.patch(f"/api/v1/auth/users/{admin_id}", headers=_auth(admin_token), json={"is_admin": False})

    assert demote.status_code == status.HTTP_400_BAD_REQUEST, demote.text
    assert users.count_admins() == 1


def test_setting_a_password_on_the_system_user_returns_400(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    """Otherwise the owner of every pre-multiuser board, image, and workflow becomes a
    login account."""
    r = client.patch("/api/v1/auth/users/system", headers=_auth(admin_token), json={"password": "SystemPass123"})

    assert r.status_code == status.HTTP_400_BAD_REQUEST, r.text

    login = client.post(
        "/api/v1/auth/login",
        json={"email": "system@system.invokeai", "password": "SystemPass123", "remember_me": False},
    )
    assert login.status_code != status.HTTP_200_OK


def test_deleting_the_system_user_returns_400(
    enable_multiuser: Any, client: TestClient, admin_token: str, mock_invoker: Invoker
) -> None:
    r = client.delete("/api/v1/auth/users/system", headers=_auth(admin_token))

    assert r.status_code == status.HTTP_400_BAD_REQUEST, r.text
    assert mock_invoker.services.users.get("system") is not None
