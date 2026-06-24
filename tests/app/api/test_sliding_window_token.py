"""Tests for SlidingWindowTokenMiddleware and token refresh behavior."""

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invokeai.app.services.auth.token_service import TokenData, create_access_token, set_jwt_secret


@pytest.fixture(autouse=True)
def _setup_jwt_secret():
    """Ensure JWT secret is set for all tests."""
    set_jwt_secret("test-secret-key-for-sliding-window-tests")


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
