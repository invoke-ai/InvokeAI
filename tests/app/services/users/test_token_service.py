"""Tests for token service."""

from datetime import timedelta

from invokeai.app.services.auth.token_service import TokenData, create_access_token, verify_token


def test_create_access_token():
    """Test creating an access token."""
    data = TokenData(user_id="test-user", email="test@example.com", is_admin=False)
    token = create_access_token(data)

    assert token is not None
    assert len(token) > 0


def test_verify_valid_token():
    """Test verifying a valid token."""
    data = TokenData(user_id="test-user", email="test@example.com", is_admin=True)
    token = create_access_token(data)

    verified_data = verify_token(token)

    assert verified_data is not None
    assert verified_data.user_id == data.user_id
    assert verified_data.email == data.email
    assert verified_data.is_admin == data.is_admin


def test_verify_invalid_token():
    """Test verifying an invalid token."""
    verified_data = verify_token("invalid-token")
    assert verified_data is None


def test_token_with_custom_expiration():
    """Test creating token with custom expiration."""
    data = TokenData(user_id="test-user", email="test@example.com", is_admin=False)
    token = create_access_token(data, expires_delta=timedelta(hours=1))

    verified_data = verify_token(token)
    assert verified_data is not None
    assert verified_data.user_id == data.user_id
