"""Password hashing and validation utilities."""

from typing import Literal

import bcrypt

# bcrypt's work factor. This is also bcrypt.gensalt()'s default and what every stored hash so far was
# produced with (via passlib, which this module used to wrap); stated explicitly so a library default
# change cannot silently weaken new hashes.
BCRYPT_ROUNDS = 12


def _password_bytes(password: str) -> bytes:
    """Encode a password for bcrypt, truncating to its 72-byte limit.

    Truncation drops any incomplete trailing UTF-8 sequence, so hashing and verification agree byte
    for byte on long passwords rather than depending on how the library treats over-long input.
    """
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72].decode("utf-8", errors="ignore").encode("utf-8")
    return password_bytes


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    bcrypt has a maximum password length of 72 bytes. Longer passwords
    are automatically truncated to comply with this limit.

    Args:
        password: The plain text password to hash

    Returns:
        The hashed password
    """
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.

    bcrypt has a maximum password length of 72 bytes. Longer passwords
    are automatically truncated to match hash_password behavior.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to verify against

    Returns:
        True if the password matches the hash, False otherwise
    """
    try:
        return bcrypt.checkpw(_password_bytes(plain_password), hashed_password.encode("ascii"))
    except Exception:
        # Invalid hash format or other error - return False
        return False


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password meets minimum security requirements.

    Password requirements:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit

    Args:
        password: The password to validate

    Returns:
        A tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)

    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and numbers"

    return True, ""


def get_password_strength(password: str) -> Literal["weak", "moderate", "strong"]:
    """Determine the strength of a password.

    Strength levels:
    - weak: less than 8 characters
    - moderate: 8+ characters but missing at least one of uppercase, lowercase, or digit
    - strong: 8+ characters with uppercase, lowercase, and digit

    Args:
        password: The password to evaluate

    Returns:
        One of "weak", "moderate", or "strong"
    """
    if len(password) < 8:
        return "weak"

    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)

    if not (has_upper and has_lower and has_digit):
        return "moderate"

    return "strong"
