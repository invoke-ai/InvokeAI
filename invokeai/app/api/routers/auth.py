"""Authentication endpoints."""

import secrets
import string
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Path, status
from pydantic import BaseModel, Field, field_validator

from invokeai.app.api.auth_dependencies import AdminUser, CurrentUser
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.auth.token_service import TokenData, create_access_token
from invokeai.app.services.users.users_common import (
    UserCreateRequest,
    UserDTO,
    UserUpdateRequest,
    validate_email_with_special_domains,
)

auth_router = APIRouter(prefix="/v1/auth", tags=["authentication"])

# Token expiration constants (in days)
TOKEN_EXPIRATION_NORMAL = 1  # 1 day for normal login
TOKEN_EXPIRATION_REMEMBER_ME = 7  # 7 days for "remember me" login


class LoginRequest(BaseModel):
    """Request body for user login."""

    email: str = Field(description="User email address")
    password: str = Field(description="User password")
    remember_me: bool = Field(default=False, description="Whether to extend session duration")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email address, allowing special-use domains."""
        return validate_email_with_special_domains(v)


class LoginResponse(BaseModel):
    """Response from successful login."""

    token: str = Field(description="JWT access token")
    user: UserDTO = Field(description="User information")
    expires_in: int = Field(description="Token expiration time in seconds")


class SetupRequest(BaseModel):
    """Request body for initial admin setup."""

    email: str = Field(description="Admin email address")
    display_name: str | None = Field(default=None, description="Admin display name")
    password: str = Field(description="Admin password")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email address, allowing special-use domains."""
        return validate_email_with_special_domains(v)


class SetupResponse(BaseModel):
    """Response from successful admin setup."""

    success: bool = Field(description="Whether setup was successful")
    user: UserDTO = Field(description="Created admin user information")


class LogoutResponse(BaseModel):
    """Response from logout."""

    success: bool = Field(description="Whether logout was successful")


class SetupStatusResponse(BaseModel):
    """Response for setup status check."""

    setup_required: bool = Field(description="Whether initial setup is required")
    multiuser_enabled: bool = Field(description="Whether multiuser mode is enabled")
    strict_password_checking: bool = Field(description="Whether strict password requirements are enforced")
    admin_email: str | None = Field(default=None, description="Email of the first active admin user, if any")


@auth_router.get("/status", response_model=SetupStatusResponse)
async def get_setup_status() -> SetupStatusResponse:
    """Check if initial administrator setup is required.

    Returns:
        SetupStatusResponse indicating whether setup is needed and multiuser mode status
    """
    config = ApiDependencies.invoker.services.configuration

    # If multiuser is disabled, setup is never required
    if not config.multiuser:
        return SetupStatusResponse(
            setup_required=False,
            multiuser_enabled=False,
            strict_password_checking=config.strict_password_checking,
            admin_email=None,
        )

    # In multiuser mode, check if an admin exists
    user_service = ApiDependencies.invoker.services.users
    setup_required = not user_service.has_admin()

    # Only expose admin_email during initial setup to avoid leaking
    # administrator identity on public deployments.
    admin_email = user_service.get_admin_email() if setup_required else None

    return SetupStatusResponse(
        setup_required=setup_required,
        multiuser_enabled=True,
        strict_password_checking=config.strict_password_checking,
        admin_email=admin_email,
    )


@auth_router.post("/login", response_model=LoginResponse)
async def login(
    request: Annotated[LoginRequest, Body(description="Login credentials")],
) -> LoginResponse:
    """Authenticate user and return access token.

    Args:
        request: Login credentials (email and password)

    Returns:
        LoginResponse containing JWT token and user information

    Raises:
        HTTPException: 401 if credentials are invalid or user is inactive
        HTTPException: 403 if multiuser mode is disabled
    """
    config = ApiDependencies.invoker.services.configuration

    # Check if multiuser is enabled
    if not config.multiuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Multiuser mode is disabled. Authentication is not required in single-user mode.",
        )

    user_service = ApiDependencies.invoker.services.users
    user = user_service.authenticate(request.email, request.password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled")

    # Create token with appropriate expiration
    expires_delta = timedelta(days=TOKEN_EXPIRATION_REMEMBER_ME if request.remember_me else TOKEN_EXPIRATION_NORMAL)
    token_data = TokenData(
        user_id=user.user_id,
        email=user.email,
        is_admin=user.is_admin,
        remember_me=request.remember_me,
    )
    token = create_access_token(token_data, expires_delta)

    return LoginResponse(
        token=token,
        user=user,
        expires_in=int(expires_delta.total_seconds()),
    )


@auth_router.post("/logout", response_model=LogoutResponse)
async def logout(
    current_user: CurrentUser,
) -> LogoutResponse:
    """Logout current user.

    Currently a no-op since we use stateless JWT tokens. For token invalidation in
    future implementations, consider:
    - Token blacklist: Store invalidated tokens in Redis/database with expiration
    - Token versioning: Add version field to user record, increment on logout
    - Short-lived tokens: Use refresh token pattern with token rotation
    - Session storage: Track active sessions server-side for revocation

    Args:
        current_user: The authenticated user (validates token)

    Returns:
        LogoutResponse indicating success
    """
    # TODO: Implement token invalidation when server-side session management is added
    # For now, this is a no-op since we use stateless JWT tokens
    return LogoutResponse(success=True)


@auth_router.get("/me", response_model=UserDTO)
async def get_current_user_info(
    current_user: CurrentUser,
) -> UserDTO:
    """Get current authenticated user's information.

    Args:
        current_user: The authenticated user's token data

    Returns:
        UserDTO containing user information

    Raises:
        HTTPException: 404 if user is not found (should not happen normally)
    """
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(current_user.user_id)

    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return user


@auth_router.post("/setup", response_model=SetupResponse)
async def setup_admin(
    request: Annotated[SetupRequest, Body(description="Admin account details")],
) -> SetupResponse:
    """Set up initial administrator account.

    This endpoint can only be called once, when no admin user exists. It creates
    the first admin user for the system.

    Args:
        request: Admin account details (email, display_name, password)

    Returns:
        SetupResponse containing the created admin user

    Raises:
        HTTPException: 400 if admin already exists or password is weak
        HTTPException: 403 if multiuser mode is disabled
    """
    config = ApiDependencies.invoker.services.configuration

    # Check if multiuser is enabled
    if not config.multiuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Multiuser mode is disabled. Admin setup is not required in single-user mode.",
        )

    user_service = ApiDependencies.invoker.services.users

    # Check if any admin exists
    if user_service.has_admin():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Administrator account already configured",
        )

    # Create admin user - this will validate password strength
    try:
        user_data = UserCreateRequest(
            email=request.email,
            display_name=request.display_name,
            password=request.password,
            is_admin=True,
        )
        user = user_service.create_admin(user_data, strict_password_checking=config.strict_password_checking)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    return SetupResponse(success=True, user=user)


# ---------------------------------------------------------------------------
# User management models
# ---------------------------------------------------------------------------

_PASSWORD_ALPHABET = string.ascii_letters + string.digits + string.punctuation


class AdminUserCreateRequest(BaseModel):
    """Request body for admin to create a new user."""

    email: str = Field(description="User email address")
    display_name: str | None = Field(default=None, description="Display name")
    password: str = Field(description="User password")
    is_admin: bool = Field(default=False, description="Whether user should have admin privileges")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email address, allowing special-use domains."""
        return validate_email_with_special_domains(v)


class AdminUserUpdateRequest(BaseModel):
    """Request body for admin to update any user."""

    display_name: str | None = Field(default=None, description="Display name")
    password: str | None = Field(default=None, description="New password")
    is_admin: bool | None = Field(default=None, description="Whether user should have admin privileges")
    is_active: bool | None = Field(default=None, description="Whether user account should be active")


class UserProfileUpdateRequest(BaseModel):
    """Request body for a user to update their own profile."""

    display_name: str | None = Field(default=None, description="New display name")
    current_password: str | None = Field(default=None, description="Current password (required when changing password)")
    new_password: str | None = Field(default=None, description="New password")


class GeneratePasswordResponse(BaseModel):
    """Response containing a generated password."""

    password: str = Field(description="Generated strong password")


# ---------------------------------------------------------------------------
# User management endpoints
# ---------------------------------------------------------------------------


@auth_router.get("/generate-password", response_model=GeneratePasswordResponse)
async def generate_password(
    current_user: CurrentUser,
) -> GeneratePasswordResponse:
    """Generate a strong random password.

    Returns a cryptographically secure random password of 16 characters
    containing uppercase, lowercase, digits, and punctuation.
    """
    # Ensure the generated password always meets strength requirements:
    # at least one uppercase, one lowercase, one digit, one special char.
    while True:
        password = "".join(secrets.choice(_PASSWORD_ALPHABET) for _ in range(16))
        if (
            any(c.isupper() for c in password)
            and any(c.islower() for c in password)
            and any(c.isdigit() for c in password)
        ):
            return GeneratePasswordResponse(password=password)


@auth_router.get("/users", response_model=list[UserDTO])
async def list_users(
    current_user: AdminUser,
) -> list[UserDTO]:
    """List all users. Requires admin privileges.

    The internal 'system' user (created for backward compatibility) is excluded
    from the results since it cannot be managed through this interface.

    Returns:
        List of all real users (system user excluded)
    """
    user_service = ApiDependencies.invoker.services.users
    return [u for u in user_service.list_users() if u.user_id != "system"]


@auth_router.post("/users", response_model=UserDTO, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: Annotated[AdminUserCreateRequest, Body(description="New user details")],
    current_user: AdminUser,
) -> UserDTO:
    """Create a new user. Requires admin privileges.

    Args:
        request: New user details

    Returns:
        The created user

    Raises:
        HTTPException: 400 if email already exists or password is weak
    """
    user_service = ApiDependencies.invoker.services.users
    config = ApiDependencies.invoker.services.configuration
    try:
        user_data = UserCreateRequest(
            email=request.email,
            display_name=request.display_name,
            password=request.password,
            is_admin=request.is_admin,
        )
        return user_service.create(user_data, strict_password_checking=config.strict_password_checking)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@auth_router.get("/users/{user_id}", response_model=UserDTO)
async def get_user(
    user_id: Annotated[str, Path(description="User ID")],
    current_user: AdminUser,
) -> UserDTO:
    """Get a user by ID. Requires admin privileges.

    Args:
        user_id: The user ID

    Returns:
        The user

    Raises:
        HTTPException: 404 if user not found
    """
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@auth_router.patch("/users/{user_id}", response_model=UserDTO)
async def update_user(
    user_id: Annotated[str, Path(description="User ID")],
    request: Annotated[AdminUserUpdateRequest, Body(description="User fields to update")],
    current_user: AdminUser,
) -> UserDTO:
    """Update a user. Requires admin privileges.

    Args:
        user_id: The user ID
        request: Fields to update

    Returns:
        The updated user

    Raises:
        HTTPException: 400 if password is weak
        HTTPException: 404 if user not found
    """
    user_service = ApiDependencies.invoker.services.users
    config = ApiDependencies.invoker.services.configuration
    try:
        changes = UserUpdateRequest(
            display_name=request.display_name,
            password=request.password,
            is_admin=request.is_admin,
            is_active=request.is_active,
        )
        return user_service.update(user_id, changes, strict_password_checking=config.strict_password_checking)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@auth_router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: Annotated[str, Path(description="User ID")],
    current_user: AdminUser,
) -> None:
    """Delete a user. Requires admin privileges.

    Admins can delete any user including other admins, but cannot delete the last
    remaining admin.

    Args:
        user_id: The user ID

    Raises:
        HTTPException: 400 if attempting to delete the last admin
        HTTPException: 404 if user not found
    """
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Prevent deleting the last active admin
    if user.is_admin and user.is_active and user_service.count_admins() <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the last administrator",
        )

    try:
        user_service.delete(user_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@auth_router.patch("/me", response_model=UserDTO)
async def update_current_user(
    request: Annotated[UserProfileUpdateRequest, Body(description="Profile fields to update")],
    current_user: CurrentUser,
) -> UserDTO:
    """Update the current user's own profile.

    To change the password, both ``current_password`` and ``new_password`` must
    be provided. The current password is verified before the change is applied.

    Args:
        request: Profile fields to update
        current_user: The authenticated user

    Returns:
        The updated user

    Raises:
        HTTPException: 400 if current password is incorrect or new password is weak
        HTTPException: 404 if user not found
    """
    user_service = ApiDependencies.invoker.services.users
    config = ApiDependencies.invoker.services.configuration

    # Verify current password when attempting a password change
    if request.new_password is not None:
        if not request.current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is required to set a new password",
            )

        # Re-authenticate to verify the current password
        user = user_service.get(current_user.user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        authenticated = user_service.authenticate(user.email, request.current_password)
        if authenticated is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

    try:
        changes = UserUpdateRequest(
            display_name=request.display_name,
            password=request.new_password,
        )
        return user_service.update(
            current_user.user_id, changes, strict_password_checking=config.strict_password_checking
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
