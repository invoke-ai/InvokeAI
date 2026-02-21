"""JWT token generation and validation."""

from datetime import datetime, timedelta, timezone
from typing import cast

from jose import JWTError, jwt
from pydantic import BaseModel

ALGORITHM = "HS256"
DEFAULT_EXPIRATION_HOURS = 24

# Module-level variable to store the JWT secret. This is set during application initialization
# by calling set_jwt_secret(). The secret is loaded from the database where it is stored
# securely after being generated during database migration.
_jwt_secret: str | None = None


class TokenData(BaseModel):
    """Data stored in JWT token."""

    user_id: str
    email: str
    is_admin: bool


def set_jwt_secret(secret: str) -> None:
    """Set the JWT secret key for token signing and verification.

    This should be called once during application initialization with the secret
    loaded from the database.

    Args:
        secret: The JWT secret key
    """
    global _jwt_secret
    _jwt_secret = secret


def get_jwt_secret() -> str:
    """Get the JWT secret key.

    Returns:
        The JWT secret key

    Raises:
        RuntimeError: If the secret has not been initialized
    """
    if _jwt_secret is None:
        raise RuntimeError("JWT secret has not been initialized. Call set_jwt_secret() during application startup.")
    return _jwt_secret


def create_access_token(data: TokenData, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        data: The token data to encode
        expires_delta: Optional expiration time delta. Defaults to 24 hours.

    Returns:
        The encoded JWT token
    """
    to_encode = data.model_dump()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=DEFAULT_EXPIRATION_HOURS))
    to_encode.update({"exp": expire})
    return cast(str, jwt.encode(to_encode, get_jwt_secret(), algorithm=ALGORITHM))


def verify_token(token: str) -> TokenData | None:
    """Verify and decode a JWT token.

    Args:
        token: The JWT token to verify

    Returns:
        TokenData if valid, None if invalid or expired
    """
    try:
        # python-jose 3.5.0 has a bug where exp verification doesn't work properly
        # We need to manually check expiration, but MUST verify signature first
        # to prevent accepting tokens with valid payloads but invalid signatures

        # First, verify the signature - this will raise JWTError if signature is invalid
        # Note: python-jose won't reject expired tokens here due to the bug
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=[ALGORITHM],
        )

        # Now manually check expiration (because python-jose 3.5.0 doesn't do this properly)
        if "exp" in payload:
            exp_timestamp = payload["exp"]
            current_timestamp = datetime.now(timezone.utc).timestamp()
            if current_timestamp >= exp_timestamp:
                # Token is expired
                return None

        return TokenData(**payload)
    except JWTError:
        # Token is invalid (bad signature, malformed, etc.)
        return None
    except Exception:
        # Catch any other exceptions (e.g., Pydantic validation errors)
        return None
