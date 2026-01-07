"""Common types and data models for user service."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserDTO(BaseModel):
    """User data transfer object."""

    user_id: str = Field(description="Unique user identifier")
    email: EmailStr = Field(description="User email address")
    display_name: str | None = Field(default=None, description="Display name")
    is_admin: bool = Field(default=False, description="Whether user has admin privileges")
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(description="When the user was created")
    updated_at: datetime = Field(description="When the user was last updated")
    last_login_at: datetime | None = Field(default=None, description="When user last logged in")


class UserCreateRequest(BaseModel):
    """Request to create a new user."""

    email: EmailStr = Field(description="User email address")
    display_name: str | None = Field(default=None, description="Display name")
    password: str = Field(description="User password")
    is_admin: bool = Field(default=False, description="Whether user should have admin privileges")


class UserUpdateRequest(BaseModel):
    """Request to update a user."""

    display_name: str | None = Field(default=None, description="Display name")
    password: str | None = Field(default=None, description="New password")
    is_admin: bool | None = Field(default=None, description="Whether user should have admin privileges")
    is_active: bool | None = Field(default=None, description="Whether user account should be active")
