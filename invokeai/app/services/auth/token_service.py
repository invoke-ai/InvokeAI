"""JWT token generation and validation."""

from datetime import datetime, timedelta, timezone
from typing import cast

import jwt
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
    remember_me: bool = False
    # Revocation epoch copied from the user record when the token was minted. A token
    # whose epoch no longer matches the record is rejected, which is how a password
    # change invalidates sessions a JWT would otherwise keep alive until expiry.
    # Defaults to 0 so tokens issued before this claim existed keep working against
    # records that have never been bumped.
    token_epoch: int = 0


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
        # PyJWT verifies the signature before the claims, and rejects an expired `exp`.
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        return TokenData(**payload)
    except jwt.PyJWTError:
        # Token is invalid (bad signature, expired, malformed, etc.)
        return None
    except Exception:
        # Catch any other exceptions (e.g., Pydantic validation errors)
        return None


def get_token_remaining_seconds(token: str) -> int | None:
    """Return the number of seconds until a *valid* token expires.

    Verifies the token first (signature + expiry + payload shape); returns None if
    it fails verification. A valid token without an ``exp`` claim gets the default
    expiration window, matching what ``create_access_token`` would have assigned.
    """
    if verify_token(token) is None:
        return None
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return None
    exp = claims.get("exp")
    if exp is None:
        return int(timedelta(hours=DEFAULT_EXPIRATION_HOURS).total_seconds())
    # PyJWT accepted the claim only if `int(exp)` succeeds, which also admits a numeric string.
    remaining = int(float(exp) - datetime.now(timezone.utc).timestamp())
    return remaining if remaining > 0 else None
