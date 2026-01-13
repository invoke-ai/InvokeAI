"""Unit tests for JWT token service."""

import time
from datetime import timedelta

from invokeai.app.services.auth.token_service import TokenData, create_access_token, verify_token


class TestTokenCreation:
    """Tests for JWT token creation."""

    def test_create_access_token_basic(self):
        """Test creating a basic access token."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_expiration(self):
        """Test creating token with custom expiration."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data, expires_delta=timedelta(hours=1))

        assert token is not None
        # Verify token is valid
        verified_data = verify_token(token)
        assert verified_data is not None
        assert verified_data.user_id == "user123"

    def test_create_access_token_admin_user(self):
        """Test creating token for admin user."""
        token_data = TokenData(
            user_id="admin123",
            email="admin@example.com",
            is_admin=True,
        )

        token = create_access_token(token_data)
        verified_data = verify_token(token)

        assert verified_data is not None
        assert verified_data.is_admin is True

    def test_create_access_token_preserves_all_data(self):
        """Test that all token data is preserved."""
        token_data = TokenData(
            user_id="user_with_complex_id_12345",
            email="complex.email+tag@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)
        verified_data = verify_token(token)

        assert verified_data is not None
        assert verified_data.user_id == token_data.user_id
        assert verified_data.email == token_data.email
        assert verified_data.is_admin == token_data.is_admin

    def test_create_access_token_different_each_time(self):
        """Test that creating token with same data produces different tokens (due to timestamps)."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        # Create tokens with different expiration times to ensure uniqueness
        token1 = create_access_token(token_data, expires_delta=timedelta(hours=1))
        token2 = create_access_token(token_data, expires_delta=timedelta(hours=2))

        # Tokens should be different due to different exp timestamps
        assert token1 != token2


class TestTokenVerification:
    """Tests for JWT token verification."""

    def test_verify_valid_token(self):
        """Test verifying a valid token."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)
        verified_data = verify_token(token)

        assert verified_data is not None
        assert verified_data.user_id == "user123"
        assert verified_data.email == "test@example.com"
        assert verified_data.is_admin is False

    def test_verify_invalid_token(self):
        """Test verifying an invalid token."""
        verified_data = verify_token("invalid_token_string")

        assert verified_data is None

    def test_verify_malformed_token(self):
        """Test verifying malformed tokens."""
        malformed_tokens = [
            "",
            "not.a.token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
            "header.payload",  # Missing signature
        ]

        for token in malformed_tokens:
            verified_data = verify_token(token)
            assert verified_data is None, f"Should reject malformed token: {token}"

    def test_verify_expired_token(self):
        """Test verifying an expired token."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        # Create token that expires in 100 milliseconds (0.1 seconds)
        token = create_access_token(token_data, expires_delta=timedelta(milliseconds=100))

        # Wait for token to expire (wait longer than expiration - 200ms to be safe)
        time.sleep(0.2)

        # Token should be invalid now
        verified_data = verify_token(token)
        assert verified_data is None

    def test_verify_token_with_modified_payload(self):
        """Test that tokens with modified payloads are rejected."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)

        # Try to modify the token by changing a character
        # JWT tokens are base64 encoded, so changing any character should invalidate the signature
        if len(token) > 10:
            modified_token = token[:-1] + ("X" if token[-1] != "X" else "Y")
            verified_data = verify_token(modified_token)
            assert verified_data is None

    def test_verify_token_preserves_admin_status(self):
        """Test that admin status is correctly preserved through token lifecycle."""
        # Test with regular user
        token_data = TokenData(
            user_id="user123",
            email="user@example.com",
            is_admin=False,
        )
        token = create_access_token(token_data)
        verified = verify_token(token)
        assert verified is not None
        assert verified.is_admin is False

        # Test with admin user
        admin_token_data = TokenData(
            user_id="admin123",
            email="admin@example.com",
            is_admin=True,
        )
        admin_token = create_access_token(admin_token_data)
        admin_verified = verify_token(admin_token)
        assert admin_verified is not None
        assert admin_verified.is_admin is True


class TestTokenExpiration:
    """Tests for token expiration handling."""

    def test_token_not_expired_immediately(self):
        """Test that freshly created token is not expired."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data, expires_delta=timedelta(hours=1))
        verified_data = verify_token(token)

        assert verified_data is not None

    def test_token_with_long_expiration(self):
        """Test token with long expiration time."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        # Create token that expires in 7 days
        token = create_access_token(token_data, expires_delta=timedelta(days=7))
        verified_data = verify_token(token)

        assert verified_data is not None
        assert verified_data.user_id == "user123"

    def test_token_with_short_expiration_not_expired(self):
        """Test token with short but not yet expired time."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        # Create token that expires in 1 second
        token = create_access_token(token_data, expires_delta=timedelta(seconds=1))

        # Immediately verify - should still be valid
        verified_data = verify_token(token)
        assert verified_data is not None


class TestTokenDataModel:
    """Tests for TokenData model."""

    def test_token_data_creation(self):
        """Test creating TokenData instance."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        assert token_data.user_id == "user123"
        assert token_data.email == "test@example.com"
        assert token_data.is_admin is False

    def test_token_data_with_admin(self):
        """Test TokenData for admin user."""
        token_data = TokenData(
            user_id="admin123",
            email="admin@example.com",
            is_admin=True,
        )

        assert token_data.is_admin is True

    def test_token_data_model_dump(self):
        """Test that TokenData can be serialized."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        data_dict = token_data.model_dump()

        assert isinstance(data_dict, dict)
        assert data_dict["user_id"] == "user123"
        assert data_dict["email"] == "test@example.com"
        assert data_dict["is_admin"] is False


class TestTokenSecurity:
    """Tests for token security properties."""

    def test_token_signature_verification(self):
        """Test that token signature is verified."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)

        # Token should verify correctly
        assert verify_token(token) is not None

        # Modified token should fail verification
        if len(token) > 50:
            # Change a character in the signature part (last part of JWT)
            parts = token.split(".")
            if len(parts) == 3:
                modified_signature = parts[2][:-1] + ("X" if parts[2][-1] != "X" else "Y")
                modified_token = f"{parts[0]}.{parts[1]}.{modified_signature}"
                assert verify_token(modified_token) is None

    def test_cannot_forge_admin_token(self):
        """Test that admin status cannot be forged by modifying token."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)

        # Any modification to the token should invalidate it
        # This prevents attackers from changing is_admin=false to is_admin=true
        parts = token.split(".")
        if len(parts) == 3:
            # Try to modify the payload
            modified_payload = parts[1][:-1] + ("X" if parts[1][-1] != "X" else "Y")
            modified_token = f"{parts[0]}.{modified_payload}.{parts[2]}"

            verified_data = verify_token(modified_token)
            # Modified token should be rejected
            assert verified_data is None

    def test_token_uses_strong_algorithm(self):
        """Test that token uses secure algorithm (HS256)."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)

        # JWT tokens have format: header.payload.signature
        # Header contains algorithm information
        import base64
        import json

        parts = token.split(".")
        if len(parts) >= 1:
            # Decode header (add padding if necessary)
            header_b64 = parts[0]
            # Add padding if necessary
            padding = 4 - len(header_b64) % 4
            if padding != 4:
                header_b64 += "=" * padding

            header = json.loads(base64.urlsafe_b64decode(header_b64))
            # Should use HS256 algorithm
            assert header.get("alg") == "HS256"
