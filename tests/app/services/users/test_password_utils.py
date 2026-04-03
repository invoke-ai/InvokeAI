"""Tests for password utilities."""

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength, verify_password


def test_hash_password():
    """Test password hashing."""
    password = "TestPassword123"
    hashed = hash_password(password)

    assert hashed != password
    assert len(hashed) > 0


def test_verify_password():
    """Test password verification."""
    password = "TestPassword123"
    hashed = hash_password(password)

    assert verify_password(password, hashed)
    assert not verify_password("WrongPassword", hashed)


def test_validate_password_strength_valid():
    """Test password strength validation with valid passwords."""
    valid, msg = validate_password_strength("ValidPass123")
    assert valid
    assert msg == ""


def test_validate_password_strength_too_short():
    """Test password strength validation with short password."""
    valid, msg = validate_password_strength("Pass1")
    assert not valid
    assert "at least 8 characters" in msg


def test_validate_password_strength_no_uppercase():
    """Test password strength validation without uppercase."""
    valid, msg = validate_password_strength("password123")
    assert not valid
    assert "uppercase" in msg.lower()


def test_validate_password_strength_no_lowercase():
    """Test password strength validation without lowercase."""
    valid, msg = validate_password_strength("PASSWORD123")
    assert not valid
    assert "lowercase" in msg.lower()


def test_validate_password_strength_no_digit():
    """Test password strength validation without digit."""
    valid, msg = validate_password_strength("PasswordTest")
    assert not valid
    assert "number" in msg.lower()
