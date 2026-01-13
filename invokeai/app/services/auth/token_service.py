"""JWT token generation and validation."""

from datetime import datetime, timedelta, timezone
from typing import cast

from jose import JWTError, jwt
from pydantic import BaseModel

# SECURITY WARNING: This is a placeholder secret key for development only.
# In production, this MUST be:
# 1. Generated using a cryptographically secure random generator
# 2. Stored in environment variables or secure configuration
# 3. Never committed to source control
# 4. Rotated periodically
# TODO: Move to config system - see invokeai.app.services.config.config_default
SECRET_KEY = "your-secret-key-should-be-in-config-change-this-in-production"
ALGORITHM = "HS256"
DEFAULT_EXPIRATION_HOURS = 24


class TokenData(BaseModel):
    """Data stored in JWT token."""

    user_id: str
    email: str
    is_admin: bool


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
    return cast(str, jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM))


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
            SECRET_KEY,
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
