"""Unit tests for password utilities."""

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength, verify_password


class TestPasswordHashing:
    """Tests for password hashing functionality."""

    def test_hash_password_returns_different_hash_each_time(self):
        """Test that hashing the same password twice produces different hashes (due to salt)."""
        password = "TestPassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2
        assert hash1 != password
        assert hash2 != password

    def test_hash_password_with_special_characters(self):
        """Test hashing passwords with special characters."""
        password = "Test!@#$%^&*()_+{}[]|:;<>?,./~`"
        hashed = hash_password(password)

        assert hashed is not None
        assert verify_password(password, hashed)

    def test_hash_password_with_unicode(self):
        """Test hashing passwords with Unicode characters."""
        password = "Test密码123パスワード"
        hashed = hash_password(password)

        assert hashed is not None
        assert verify_password(password, hashed)

    def test_hash_password_empty_string(self):
        """Test hashing empty password (should work but fail validation)."""
        password = ""
        hashed = hash_password(password)

        assert hashed is not None
        assert verify_password(password, hashed)

    def test_hash_password_very_long(self):
        """Test hashing very long passwords (bcrypt has 72 byte limit)."""
        # Create a password longer than 72 bytes
        password = "A" * 100
        hashed = hash_password(password)

        assert hashed is not None
        # Verify with original password
        assert verify_password(password, hashed)
        # Should also match the truncated version
        assert verify_password("A" * 72, hashed)

    def test_hash_password_with_newlines(self):
        """Test hashing passwords containing newlines."""
        password = "Test\nPassword\n123"
        hashed = hash_password(password)

        assert hashed is not None
        assert verify_password(password, hashed)


class TestPasswordVerification:
    """Tests for password verification functionality."""

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert verify_password("WrongPassword123", hashed) is False

    def test_verify_password_case_sensitive(self):
        """Test that password verification is case-sensitive."""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert verify_password("testpassword123", hashed) is False
        assert verify_password("TESTPASSWORD123", hashed) is False

    def test_verify_password_whitespace_sensitive(self):
        """Test that whitespace matters in password verification."""
        password = "TestPassword123"
        hashed = hash_password(password)

        assert verify_password(" TestPassword123", hashed) is False
        assert verify_password("TestPassword123 ", hashed) is False
        assert verify_password("Test Password123", hashed) is False

    def test_verify_password_with_special_characters(self):
        """Test verifying passwords with special characters."""
        password = "Test!@#$%^&*()_+"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True
        assert verify_password("Test!@#$%^&*()_+X", hashed) is False

    def test_verify_password_with_unicode(self):
        """Test verifying passwords with Unicode."""
        password = "Test密码123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True
        assert verify_password("Test密码124", hashed) is False

    def test_verify_password_empty_against_hashed(self):
        """Test verifying empty password."""
        password = ""
        hashed = hash_password(password)

        assert verify_password("", hashed) is True
        assert verify_password("notEmpty", hashed) is False

    def test_verify_password_invalid_hash_format(self):
        """Test verifying password against invalid hash format."""
        password = "TestPassword123"

        # Should return False for invalid hash, not raise exception
        assert verify_password(password, "not_a_valid_hash") is False
        assert verify_password(password, "") is False


class TestPasswordStrengthValidation:
    """Tests for password strength validation."""

    def test_validate_strong_password(self):
        """Test validating a strong password."""
        valid, message = validate_password_strength("StrongPass123")

        assert valid is True
        assert message == ""

    def test_validate_password_too_short(self):
        """Test validating password shorter than 8 characters."""
        valid, message = validate_password_strength("Short1")

        assert valid is False
        assert "at least 8 characters" in message

    def test_validate_password_minimum_length(self):
        """Test validating password with exactly 8 characters."""
        valid, message = validate_password_strength("Pass123A")

        assert valid is True
        assert message == ""

    def test_validate_password_no_uppercase(self):
        """Test validating password without uppercase letters."""
        valid, message = validate_password_strength("lowercase123")

        assert valid is False
        assert "uppercase" in message.lower()

    def test_validate_password_no_lowercase(self):
        """Test validating password without lowercase letters."""
        valid, message = validate_password_strength("UPPERCASE123")

        assert valid is False
        assert "lowercase" in message.lower()

    def test_validate_password_no_digits(self):
        """Test validating password without digits."""
        valid, message = validate_password_strength("NoDigitsHere")

        assert valid is False
        assert "number" in message.lower()

    def test_validate_password_with_special_characters(self):
        """Test that special characters are allowed but not required."""
        # With special characters
        valid, message = validate_password_strength("Pass!@#$123")
        assert valid is True

        # Without special characters (but meets other requirements)
        valid, message = validate_password_strength("Password123")
        assert valid is True

    def test_validate_password_with_spaces(self):
        """Test validating password with spaces."""
        # Password with spaces that meets requirements
        valid, message = validate_password_strength("Pass Word 123")

        assert valid is True
        assert message == ""

    def test_validate_password_unicode(self):
        """Test validating password with Unicode characters."""
        # Unicode with uppercase, lowercase, and digits
        valid, message = validate_password_strength("密码Pass123")

        assert valid is True

    def test_validate_password_empty(self):
        """Test validating empty password."""
        valid, message = validate_password_strength("")

        assert valid is False
        assert "at least 8 characters" in message

    def test_validate_password_all_requirements_barely_met(self):
        """Test password that barely meets all requirements."""
        # 8 chars, 1 upper, 1 lower, 1 digit
        valid, message = validate_password_strength("Passwor1")

        assert valid is True
        assert message == ""

    def test_validate_password_very_long(self):
        """Test validating very long password."""
        # Very long password that meets requirements
        password = "A" * 50 + "a" * 50 + "1" * 50
        valid, message = validate_password_strength(password)

        assert valid is True
        assert message == ""


class TestPasswordSecurityProperties:
    """Tests for security properties of password handling."""

    def test_timing_attack_resistance_same_length(self):
        """Test that password verification takes similar time for correct and incorrect passwords.

        Note: This is a basic check. Real timing attack resistance requires more sophisticated testing.
        """
        import time

        password = "TestPassword123"
        hashed = hash_password(password)

        # Measure time for correct password
        start = time.perf_counter()
        for _ in range(100):
            verify_password(password, hashed)
        correct_time = time.perf_counter() - start

        # Measure time for incorrect password of same length
        start = time.perf_counter()
        for _ in range(100):
            verify_password("WrongPassword12", hashed)
        incorrect_time = time.perf_counter() - start

        # Times should be relatively similar (within 50% difference)
        # This is a loose check as bcrypt is designed to be slow and timing-resistant
        ratio = max(correct_time, incorrect_time) / min(correct_time, incorrect_time)
        assert ratio < 1.5, "Timing difference too large, potential timing attack vulnerability"

    def test_different_hashes_for_same_password(self):
        """Test that the same password produces different hashes (salt randomization)."""
        password = "TestPassword123"
        hashes = {hash_password(password) for _ in range(10)}

        # All hashes should be unique due to random salt
        assert len(hashes) == 10

    def test_hash_output_format(self):
        """Test that hash output follows bcrypt format."""
        password = "TestPassword123"
        hashed = hash_password(password)

        # Bcrypt hashes start with $2b$ (or other valid bcrypt identifiers)
        assert hashed.startswith("$2")
        # Bcrypt hashes are 60 characters long
        assert len(hashed) == 60
