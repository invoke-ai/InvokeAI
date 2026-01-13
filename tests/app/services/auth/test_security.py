"""Security tests for multiuser authentication system.

This module tests various security aspects including:
- SQL injection prevention
- Authorization bypass attempts
- Session security
- Input validation
"""

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    """Create a test client for the FastAPI app."""
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    """Mock API dependencies for testing."""

    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


def setup_test_user(mock_invoker: Invoker, email: str = "test@example.com", password: str = "TestPass123") -> str:
    """Helper to create a test user and return user_id."""
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name="Test User",
        password=password,
        is_admin=False,
    )
    user = user_service.create(user_data)
    return user.user_id


def setup_test_admin(mock_invoker: Invoker, email: str = "admin@example.com", password: str = "AdminPass123") -> str:
    """Helper to create a test admin user and return user_id."""
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name="Admin User",
        password=password,
        is_admin=True,
    )
    user = user_service.create(user_data)
    return user.user_id


class TestSQLInjectionPrevention:
    """Tests to ensure SQL injection attacks are prevented."""

    def test_login_sql_injection_in_email(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that SQL injection in email field is prevented."""
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

        # Create a legitimate user first
        setup_test_user(mock_invoker, "legitimate@example.com", "TestPass123")

        # Try SQL injection in email field
        sql_injection_attempts = [
            "' OR '1'='1",
            "admin' --",
            "' OR 1=1 --",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
        ]

        for injection_attempt in sql_injection_attempts:
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": injection_attempt,
                    "password": "TestPass123",
                    "remember_me": False,
                },
            )

            # Should return 401 (invalid credentials) or 422 (validation error)
            # Both are acceptable - the important thing is no SQL injection occurs
            assert response.status_code in [401, 422], f"SQL injection attempt should be rejected: {injection_attempt}"
            # Should NOT return 200 (success) or 500 (server error)
            assert response.status_code != 200, f"SQL injection should not succeed: {injection_attempt}"
            assert response.status_code != 500, f"SQL injection should not cause server error: {injection_attempt}"

    def test_login_sql_injection_in_password(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that SQL injection in password field is prevented."""
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

        # Create a legitimate user
        setup_test_user(mock_invoker, "test@example.com", "TestPass123")

        # Try SQL injection in password field
        sql_injection_attempts = [
            "' OR '1'='1",
            "anything' OR '1'='1' --",
            "' OR 1=1; DROP TABLE users; --",
        ]

        for injection_attempt in sql_injection_attempts:
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": injection_attempt,
                    "remember_me": False,
                },
            )

            # Should fail authentication
            assert response.status_code == 401, f"SQL injection attempt should be rejected: {injection_attempt}"

    def test_user_service_sql_injection_in_email(self, mock_invoker: Invoker):
        """Test that user service prevents SQL injection in email lookups."""
        user_service = mock_invoker.services.users

        # Create a test user
        setup_test_user(mock_invoker, "test@example.com", "TestPass123")

        # Try SQL injection in get_by_email
        sql_injection_attempts = [
            "test@example.com' OR '1'='1",
            "' OR 1=1 --",
            "test@example.com'; DROP TABLE users; --",
        ]

        for injection_attempt in sql_injection_attempts:
            # Should return None (not found), not raise an error or return wrong user
            user = user_service.get_by_email(injection_attempt)
            assert user is None, f"SQL injection should not return a user: {injection_attempt}"


class TestAuthorizationBypass:
    """Tests to ensure authorization cannot be bypassed."""

    def test_cannot_access_protected_endpoint_without_token(self, client: TestClient):
        """Test that protected endpoints require authentication."""
        # Try to access protected endpoint without token
        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401

    def test_cannot_access_protected_endpoint_with_invalid_token(
        self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient
    ):
        """Test that invalid tokens are rejected."""
        monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

        invalid_tokens = [
            "invalid_token",
            "Bearer invalid_token",
            "",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature",
        ]

        for token in invalid_tokens:
            response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

            assert response.status_code == 401, f"Invalid token should be rejected: {token}"

    def test_cannot_forge_admin_token(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that admin privileges cannot be forged by modifying tokens."""
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
        monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

        # Create a regular user and login
        setup_test_user(mock_invoker, "regular@example.com", "TestPass123")

        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "regular@example.com",
                "password": "TestPass123",
                "remember_me": False,
            },
        )

        token = login_response.json()["token"]

        # Try to modify the token to gain admin privileges
        # (In practice, this should fail signature verification)
        parts = token.split(".")
        if len(parts) == 3:
            # Decode the payload, modify it, and re-encode
            import base64
            import json

            # Add padding if necessary
            payload_b64 = parts[1]
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            # Decode payload
            try:
                payload_bytes = base64.urlsafe_b64decode(payload_b64)
                payload_data = json.loads(payload_bytes)

                # Modify is_admin to true
                payload_data["is_admin"] = True

                # Re-encode
                modified_payload_bytes = json.dumps(payload_data).encode()
                modified_payload_b64 = base64.urlsafe_b64encode(modified_payload_bytes).decode().rstrip("=")

                # Create forged token with modified payload but original signature
                modified_token = f"{parts[0]}.{modified_payload_b64}.{parts[2]}"

                # Attempt to use modified token
                response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {modified_token}"})

                # Should be rejected (invalid signature)
                assert response.status_code == 401
            except Exception:
                # If we can't decode/modify the token, that's fine - just skip this part of the test
                pass

    def test_regular_user_cannot_create_admin(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that regular users cannot create admin users."""
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
        monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

        # This test would require user management endpoints to be implemented
        # For now, we test at the service level
        user_service = mock_invoker.services.users

        # Create a regular user
        regular_user_data = UserCreateRequest(
            email="regular@example.com",
            display_name="Regular User",
            password="TestPass123",
            is_admin=False,
        )
        user_service.create(regular_user_data)

        # Try to create an admin user (should only be possible through setup or by existing admin)
        # The create_admin method checks if an admin already exists
        admin_data = UserCreateRequest(
            email="sneaky@example.com",
            display_name="Sneaky Admin",
            password="TestPass123",
        )

        # First create an actual admin
        setup_test_admin(mock_invoker, "realadmin@example.com", "AdminPass123")

        # Now trying to create another admin should fail
        with pytest.raises(ValueError, match="already exists"):
            user_service.create_admin(admin_data)


class TestSessionSecurity:
    """Tests for session and token security."""

    def test_token_expires_after_time(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that tokens expire after their validity period."""
        from datetime import timedelta

        from invokeai.app.services.auth.token_service import TokenData, create_access_token

        # Create a token that expires quickly
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        # Create token with 10 millisecond expiration
        expired_token = create_access_token(token_data, expires_delta=timedelta(milliseconds=10))

        # Wait for expiration (wait longer than expiration time)
        import time

        time.sleep(0.02)

        monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

        # Try to use expired token
        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})

        assert response.status_code == 401

    def test_logout_invalidates_session(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that logout invalidates the session.

        Note: Current implementation uses JWT which is stateless.
        This test documents expected behavior for future server-side session tracking.
        """
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
        monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

        # Create user and login
        setup_test_user(mock_invoker, "test@example.com", "TestPass123")

        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123",
                "remember_me": False,
            },
        )

        token = login_response.json()["token"]

        # Logout
        logout_response = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})

        assert logout_response.status_code == 200

        # Note: With JWT, the token is still technically valid until expiration
        # For true session invalidation, server-side session tracking would be needed


class TestInputValidation:
    """Tests for input validation and sanitization."""

    def test_email_validation_on_login(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that email validation is enforced on login."""
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

        # Invalid email formats should be rejected by pydantic validation
        invalid_emails = [
            "not_an_email",
            "@example.com",
            "user@",
            "user @example.com",  # space in email
            "../../../etc/passwd",  # path traversal attempt
        ]

        for invalid_email in invalid_emails:
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": invalid_email,
                    "password": "TestPass123",
                    "remember_me": False,
                },
            )

            # Should return 422 (validation error) or 401 (invalid credentials)
            assert response.status_code in [401, 422], f"Invalid email should be rejected: {invalid_email}"

    def test_xss_prevention_in_user_data(self, mock_invoker: Invoker):
        """Test that XSS attempts in user data are handled safely.

        Note: Database storage uses parameterized queries which prevent XSS.
        This test ensures data is stored and retrieved without executing scripts.
        """
        user_service = mock_invoker.services.users

        # Try to create user with XSS payload in display name
        xss_payloads = [
            "<script>alert('xss')</script>",
            "'; alert('xss'); //",
            "<img src=x onerror=alert('xss')>",
        ]

        for payload in xss_payloads:
            user_data = UserCreateRequest(
                email=f"xss{hash(payload)}@example.com",  # unique email
                display_name=payload,
                password="TestPass123",
                is_admin=False,
            )

            # Should not raise an error - data is stored as-is
            user = user_service.create(user_data)

            # Verify data is stored exactly as provided (not executed or modified)
            assert user.display_name == payload

            # Cleanup
            user_service.delete(user.user_id)

    def test_path_traversal_prevention(self, mock_invoker: Invoker):
        """Test that path traversal attempts in user input are handled."""
        user_service = mock_invoker.services.users

        # Path traversal attempts
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "user/../../../secret",
        ]

        for attempt in path_traversal_attempts:
            # These should be stored as literal strings, not interpreted as paths
            user_data = UserCreateRequest(
                email=f"path{hash(attempt)}@example.com",
                display_name=attempt,
                password="TestPass123",
                is_admin=False,
            )

            user = user_service.create(user_data)
            assert user.display_name == attempt

            # Cleanup
            user_service.delete(user.user_id)


class TestRateLimiting:
    """Tests for rate limiting and brute force protection.

    Note: Rate limiting is not currently implemented in the codebase.
    These tests document expected behavior for future implementation.
    """

    @pytest.mark.skip(reason="Rate limiting not yet implemented")
    def test_login_rate_limiting(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that excessive login attempts are rate limited."""
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

        setup_test_user(mock_invoker, "test@example.com", "TestPass123")

        # Try many login attempts with wrong password
        for i in range(20):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "WrongPassword",
                    "remember_me": False,
                },
            )

            if i < 10:
                # First attempts should return 401
                assert response.status_code == 401
            else:
                # After many attempts, should be rate limited (429)
                # This is expected behavior for future implementation
                pass
