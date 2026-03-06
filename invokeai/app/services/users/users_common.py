"""Common types and data models for user service."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError


def validate_email_with_special_domains(email: str) -> str:
    """Validate email address, allowing special-use domains like .local for testing.

    This validator first tries standard email validation using email-validator library.
    If it fails due to special-use domains (like .local, .test, .localhost), it performs
    a basic syntax check instead. This allows development/testing with non-routable domains
    while still catching actual typos and malformed emails.

    Args:
        email: The email address to validate

    Returns:
        The validated email address (lowercased)

    Raises:
        PydanticCustomError: If the email format is invalid
    """
    try:
        # Try standard email validation using email-validator
        from email_validator import EmailNotValidError, validate_email

        result = validate_email(email, check_deliverability=False)
        return result.normalized
    except EmailNotValidError as e:
        error_msg = str(e)

        # Check if the error is specifically about special-use/reserved domains or localhost
        if (
            "special-use" in error_msg.lower()
            or "reserved" in error_msg.lower()
            or "should have a period" in error_msg.lower()
        ):
            # Perform basic email syntax validation
            email = email.strip().lower()

            if "@" not in email:
                raise PydanticCustomError(
                    "value_error",
                    "Email address must contain an @ symbol",
                )

            local_part, domain = email.rsplit("@", 1)

            if not local_part or not domain:
                raise PydanticCustomError(
                    "value_error",
                    "Email address must have both local and domain parts",
                )

            # Allow localhost and domains with dots
            if domain == "localhost" or "." in domain:
                return email

            raise PydanticCustomError(
                "value_error",
                "Email domain must contain a dot or be 'localhost'",
            )
        else:
            # Re-raise other validation errors
            raise PydanticCustomError(
                "value_error",
                f"Invalid email address: {error_msg}",
            )


class UserDTO(BaseModel):
    """User data transfer object."""

    user_id: str = Field(description="Unique user identifier")
    email: str = Field(description="User email address")
    display_name: str | None = Field(default=None, description="Display name")
    is_admin: bool = Field(default=False, description="Whether user has admin privileges")
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(description="When the user was created")
    updated_at: datetime = Field(description="When the user was last updated")
    last_login_at: datetime | None = Field(default=None, description="When user last logged in")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email address, allowing special-use domains."""
        return validate_email_with_special_domains(v)


class UserCreateRequest(BaseModel):
    """Request to create a new user."""

    email: str = Field(description="User email address")
    display_name: str | None = Field(default=None, description="Display name")
    password: str = Field(description="User password")
    is_admin: bool = Field(default=False, description="Whether user should have admin privileges")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email address, allowing special-use domains."""
        return validate_email_with_special_domains(v)


class UserUpdateRequest(BaseModel):
    """Request to update a user."""

    display_name: str | None = Field(default=None, description="Display name")
    password: str | None = Field(default=None, description="New password")
    is_admin: bool | None = Field(default=None, description="Whether user should have admin privileges")
    is_active: bool | None = Field(default=None, description="Whether user account should be active")
