"""Router-level tests for /api/v1/wildcards.

Covers ownership scoping (a user may only see and mutate their own wildcards, and someone else's
reads as absent rather than forbidden) and the end-to-end effect on prompt expansion.
"""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create(client: TestClient, token: str, name: str, values: list[str]):
    return client.post("/api/v1/wildcards/", json={"name": name, "values": values}, headers=_auth(token))


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/wildcards/"),
        ("post", "/api/v1/wildcards/"),
        ("patch", "/api/v1/wildcards/some-id"),
        ("delete", "/api/v1/wildcards/some-id"),
    ],
)
def test_routes_require_auth(enable_multiuser: Any, client: TestClient, method: str, path: str):
    r = client.request(method.upper(), path, json={"name": "colors", "values": ["red"]})
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_create_then_list_returns_the_wildcard(client: TestClient, user1_token: str):
    assert _create(client, user1_token, "colors", ["red", "green"]).status_code == status.HTTP_201_CREATED

    listed = client.get("/api/v1/wildcards/", headers=_auth(user1_token)).json()
    assert [(w["name"], w["values"]) for w in listed] == [("colors", ["red", "green"])]


def test_a_user_only_sees_their_own_wildcards(client: TestClient, user1_token: str, user2_token: str):
    _create(client, user1_token, "colors", ["red"])

    assert client.get("/api/v1/wildcards/", headers=_auth(user2_token)).json() == []


def test_another_users_wildcard_reads_as_absent_rather_than_forbidden(
    client: TestClient, user1_token: str, user2_token: str
):
    # 403 would confirm the id exists, so a non-owner gets the same 404 as a bad id.
    created = _create(client, user1_token, "colors", ["red"]).json()

    assert (
        client.patch(
            f"/api/v1/wildcards/{created['id']}", json={"values": ["blue"]}, headers=_auth(user2_token)
        ).status_code
        == status.HTTP_404_NOT_FOUND
    )
    assert (
        client.delete(f"/api/v1/wildcards/{created['id']}", headers=_auth(user2_token)).status_code
        == status.HTTP_404_NOT_FOUND
    )
    # ...and the record is untouched.
    assert client.get("/api/v1/wildcards/", headers=_auth(user1_token)).json()[0]["values"] == ["red"]


def test_duplicate_name_for_the_same_user_conflicts(client: TestClient, user1_token: str):
    _create(client, user1_token, "colors", ["red"])

    assert _create(client, user1_token, "colors", ["blue"]).status_code == status.HTTP_409_CONFLICT


def test_invalid_name_is_rejected(client: TestClient, user1_token: str):
    assert _create(client, user1_token, "has space", ["red"]).status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_update_and_delete(client: TestClient, user1_token: str):
    created = _create(client, user1_token, "colors", ["red"]).json()

    updated = client.patch(
        f"/api/v1/wildcards/{created['id']}", json={"values": ["red", "blue"]}, headers=_auth(user1_token)
    )
    assert updated.status_code == status.HTTP_200_OK
    assert updated.json()["values"] == ["red", "blue"]

    assert (
        client.delete(f"/api/v1/wildcards/{created['id']}", headers=_auth(user1_token)).status_code
        == status.HTTP_204_NO_CONTENT
    )
    assert client.get("/api/v1/wildcards/", headers=_auth(user1_token)).json() == []


def test_dynamicprompts_expands_the_callers_wildcards(client: TestClient, user1_token: str, user2_token: str):
    _create(client, user1_token, "colors", ["red", "green", "blue"])

    owner = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a __colors__ ball"},
        headers=_auth(user1_token),
    ).json()
    assert owner["error"] is None
    assert sorted(owner["prompts"]) == ["a blue ball", "a green ball", "a red ball"]

    # The same prompt is unresolvable for a user who does not own that wildcard.
    other = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a __colors__ ball"},
        headers=_auth(user2_token),
    ).json()
    assert other["error"] == "No values found for wildcard(s): colors"


@pytest.mark.timeout(30)
def test_dynamicprompts_resolves_a_wildcard_nested_in_a_variant(client: TestClient, user1_token: str):
    # Unresolvable, this shape hangs the combinatorial generator. Resolvable, it just expands.
    _create(client, user1_token, "colors", ["red", "green"])

    body = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "{__colors__|x}"},
        headers=_auth(user1_token),
    ).json()

    assert body["error"] is None
    assert sorted(body["prompts"]) == ["green", "red", "x"]
