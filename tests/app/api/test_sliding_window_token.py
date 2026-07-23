"""Tests for SlidingWindowTokenMiddleware and token refresh behavior."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from invokeai.app.services.auth.token_service import TokenData, create_access_token, set_jwt_secret


@pytest.fixture(autouse=True)
def _setup_jwt_secret(monkeypatch: pytest.MonkeyPatch):
    """Ensure JWT secret is set for all tests."""
    from invokeai.app.api.dependencies import ApiDependencies

    set_jwt_secret("test-secret-key-for-sliding-window-tests")
    user = SimpleNamespace(
        user_id="test-user",
        email="test@test.com",
        is_admin=False,
        is_active=True,
    )
    users = SimpleNamespace(get=lambda user_id: user if user_id == user.user_id else None)
    monkeypatch.setattr(
        ApiDependencies,
        "invoker",
        SimpleNamespace(services=SimpleNamespace(users=users)),
        raising=False,
    )
    return user


def _create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with the SlidingWindowTokenMiddleware."""
    from invokeai.app.api_app import SlidingWindowTokenMiddleware

    test_app = FastAPI()
    test_app.add_middleware(SlidingWindowTokenMiddleware)

    @test_app.get("/test")
    async def get_endpoint():
        return {"ok": True}

    @test_app.post("/test")
    async def post_endpoint():
        return {"ok": True}

    @test_app.put("/test")
    async def put_endpoint():
        return {"ok": True}

    @test_app.delete("/test")
    async def delete_endpoint():
        return {"ok": True}

    @test_app.post("/api/v1/auth/logout")
    async def logout_endpoint(response: Response):
        response.delete_cookie("invokeai_media_token", path="/api/v1")
        return {"ok": True}

    @test_app.post("/api/v1/auth/media-cookie")
    async def media_cookie_endpoint(request: Request, response: Response):
        from invokeai.app.api.routers.auth import _set_media_cookie

        _set_media_cookie(request, response, request.headers["authorization"][7:], 300)
        return {"ok": True}

    @test_app.post("/api/v1/auth/login")
    async def login_endpoint(response: Response):
        response.set_cookie("invokeai_media_token", "login-user-token", path="/api/v1")
        return {"token": "login-user-token"}

    return test_app


def _make_token(remember_me: bool = False, expires_delta: timedelta | None = None) -> str:
    """Create a test token."""
    token_data = TokenData(
        user_id="test-user",
        email="test@test.com",
        is_admin=False,
        remember_me=remember_me,
    )
    return create_access_token(token_data, expires_delta)


class TestSlidingWindowTokenMiddleware:
    """Tests for SlidingWindowTokenMiddleware."""

    def test_mutating_request_returns_refreshed_token(self):
        """Authenticated POST/PUT/PATCH/DELETE requests return X-Refreshed-Token."""
        app = _create_test_app()
        client = TestClient(app)
        token = _make_token()

        for method in ["post", "put", "delete"]:
            response = getattr(client, method)("/test", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 200
            assert "X-Refreshed-Token" in response.headers, f"{method.upper()} should return refreshed token"

    def test_get_request_does_not_return_refreshed_token(self):
        """Authenticated GET requests do NOT return X-Refreshed-Token."""
        app = _create_test_app()
        client = TestClient(app)
        token = _make_token()

        response = client.get("/test", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert "X-Refreshed-Token" not in response.headers
        assert "set-cookie" not in response.headers

    def test_mutating_request_does_not_write_media_cookie_directly(self):
        app = _create_test_app()
        client = TestClient(app)
        token = _make_token(expires_delta=timedelta(minutes=5))

        response = client.post("/test", headers={"Authorization": f"Bearer {token}"})

        assert "X-Refreshed-Token" in response.headers
        assert "set-cookie" not in response.headers

    @pytest.mark.parametrize("path", ["/api/v1/auth/logout", "/api/v1/auth/media-cookie"])
    def test_cookie_management_endpoints_do_not_refresh_token(self, path: str):
        app = _create_test_app()
        client = TestClient(app)
        token = _make_token(expires_delta=timedelta(minutes=5))

        response = client.post(path, headers={"Authorization": f"Bearer {token}"})

        assert "X-Refreshed-Token" not in response.headers
        if path.endswith("logout"):
            assert response.cookies.get("invokeai_media_token") is None
            assert 'invokeai_media_token=""' in response.headers["set-cookie"]
        else:
            assert response.cookies.get("invokeai_media_token") == token

    def test_login_response_is_not_overwritten_by_existing_bearer_token(self):
        app = _create_test_app()
        client = TestClient(app)
        existing_token = _make_token(expires_delta=timedelta(minutes=5))

        response = client.post("/api/v1/auth/login", headers={"Authorization": f"Bearer {existing_token}"})

        assert "X-Refreshed-Token" not in response.headers
        assert response.cookies.get("invokeai_media_token") == "login-user-token"
        assert len(response.headers.get_list("set-cookie")) == 1

    def test_unauthenticated_request_does_not_return_refreshed_token(self):
        """Requests without a token do NOT return X-Refreshed-Token."""
        app = _create_test_app()
        client = TestClient(app)

        response = client.post("/test")
        assert response.status_code == 200
        assert "X-Refreshed-Token" not in response.headers

    def test_remember_me_token_refreshes_to_remember_me_duration(self):
        """A remember_me=True token refreshes with the remember-me duration, not the normal duration."""
        from jose import jwt

        from invokeai.app.api.routers.auth import TOKEN_EXPIRATION_REMEMBER_ME
        from invokeai.app.services.auth.token_service import ALGORITHM, get_jwt_secret

        app = _create_test_app()
        client = TestClient(app)

        # Create a remember-me token with only 1 hour remaining (less than 24h)
        token = _make_token(remember_me=True, expires_delta=timedelta(hours=1))

        response = client.post("/test", headers={"Authorization": f"Bearer {token}"})
        assert "X-Refreshed-Token" in response.headers

        # Decode the refreshed token and check its expiry
        refreshed_token = response.headers["X-Refreshed-Token"]
        payload = jwt.decode(refreshed_token, get_jwt_secret(), algorithms=[ALGORITHM])

        # The refreshed token should have ~7 days of remaining life, not ~1 day
        from datetime import datetime, timezone

        remaining_seconds = payload["exp"] - datetime.now(timezone.utc).timestamp()
        remaining_days = remaining_seconds / 86400

        # Should be close to TOKEN_EXPIRATION_REMEMBER_ME (7 days), not TOKEN_EXPIRATION_NORMAL (1 day)
        assert remaining_days > TOKEN_EXPIRATION_REMEMBER_ME - 0.1, (
            f"Remember-me token was downgraded: {remaining_days:.1f} days remaining, "
            f"expected ~{TOKEN_EXPIRATION_REMEMBER_ME}"
        )

    def test_normal_token_refreshes_to_normal_duration(self):
        """A remember_me=False token refreshes with the normal duration."""
        from jose import jwt

        from invokeai.app.api.routers.auth import TOKEN_EXPIRATION_NORMAL
        from invokeai.app.services.auth.token_service import ALGORITHM, get_jwt_secret

        app = _create_test_app()
        client = TestClient(app)

        token = _make_token(remember_me=False)

        response = client.post("/test", headers={"Authorization": f"Bearer {token}"})
        refreshed_token = response.headers["X-Refreshed-Token"]
        payload = jwt.decode(refreshed_token, get_jwt_secret(), algorithms=[ALGORITHM])

        from datetime import datetime, timezone

        remaining_seconds = payload["exp"] - datetime.now(timezone.utc).timestamp()
        remaining_days = remaining_seconds / 86400

        # Should be close to TOKEN_EXPIRATION_NORMAL (1 day), not TOKEN_EXPIRATION_REMEMBER_ME (7 days)
        assert remaining_days < TOKEN_EXPIRATION_NORMAL + 0.1, (
            f"Normal token got remember-me duration: {remaining_days:.1f} days"
        )
        assert remaining_days > TOKEN_EXPIRATION_NORMAL - 0.1, (
            f"Normal token duration too short: {remaining_days:.1f} days"
        )

    def test_remember_me_claim_preserved_in_refreshed_token(self):
        """The remember_me claim is preserved when a token is refreshed."""
        from invokeai.app.services.auth.token_service import verify_token

        app = _create_test_app()
        client = TestClient(app)

        # Test with remember_me=True
        token = _make_token(remember_me=True)
        response = client.post("/test", headers={"Authorization": f"Bearer {token}"})
        refreshed_data = verify_token(response.headers["X-Refreshed-Token"])
        assert refreshed_data is not None
        assert refreshed_data.remember_me is True

        # Test with remember_me=False
        token = _make_token(remember_me=False)
        response = client.post("/test", headers={"Authorization": f"Bearer {token}"})
        refreshed_data = verify_token(response.headers["X-Refreshed-Token"])
        assert refreshed_data is not None
        assert refreshed_data.remember_me is False

    def test_refresh_uses_current_user_role_and_email(self, _setup_jwt_secret):
        from invokeai.app.services.auth.token_service import verify_token

        user = _setup_jwt_secret
        stale_token = create_access_token(
            TokenData(
                user_id=user.user_id,
                email="old@test.com",
                is_admin=True,
                remember_me=True,
            )
        )
        user.email = "new@test.com"
        user.is_admin = False

        response = TestClient(_create_test_app()).post("/test", headers={"Authorization": f"Bearer {stale_token}"})

        refreshed = verify_token(response.headers["X-Refreshed-Token"])
        assert refreshed is not None
        assert refreshed.email == "new@test.com"
        assert refreshed.is_admin is False
        assert refreshed.remember_me is True

    def test_inactive_or_deleted_user_is_not_refreshed(self, _setup_jwt_secret):
        user = _setup_jwt_secret
        token = _make_token()
        user.is_active = False

        response = TestClient(_create_test_app()).post("/test", headers={"Authorization": f"Bearer {token}"})

        assert "X-Refreshed-Token" not in response.headers
        assert "set-cookie" not in response.headers


class TestSlidingWindowBehindSubPathProxy:
    """Behind a sub-path proxy the middleware sees prefixed public paths; the refreshed
    media cookie must be scoped to the prefixed path and the auth-route exclusions must
    still match."""

    BASE_PATH = "/invoke"

    def _proxied_client(self, preserve: bool) -> TestClient:
        from invokeai.app.api_app import SubPathASGIMiddleware

        wrapped = SubPathASGIMiddleware(_create_test_app(), self.BASE_PATH)
        root = f"http://testserver{self.BASE_PATH}" if preserve else "http://testserver"
        return TestClient(wrapped, base_url=root)

    @pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
    def test_media_cookie_refresh_is_scoped_to_public_prefix(self, preserve: bool):
        client = self._proxied_client(preserve)
        token = _make_token(expires_delta=timedelta(minutes=5))

        mutation_response = client.post("/test", headers={"Authorization": f"Bearer {token}"})
        refreshed_token = mutation_response.headers["X-Refreshed-Token"]
        response = client.post("/api/v1/auth/media-cookie", headers={"Authorization": f"Bearer {refreshed_token}"})

        assert response.cookies.get("invokeai_media_token") == refreshed_token
        assert f"Path={self.BASE_PATH}/api/v1" in response.headers["set-cookie"]

    @pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
    @pytest.mark.parametrize("path", ["/api/v1/auth/login", "/api/v1/auth/logout", "/api/v1/auth/media-cookie"])
    def test_auth_route_exclusions_match_under_prefix(self, preserve: bool, path: str):
        """The exclusion list is unprefixed; the middleware must strip root_path before
        matching or a proxied logout would mint a replacement token/cookie."""
        client = self._proxied_client(preserve)
        token = _make_token(expires_delta=timedelta(minutes=5))

        response = client.post(path, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        assert "X-Refreshed-Token" not in response.headers
