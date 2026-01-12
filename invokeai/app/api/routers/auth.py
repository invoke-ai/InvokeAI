"""Authentication endpoints."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from invokeai.app.api.auth_dependencies import CurrentUser
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.auth.token_service import TokenData, create_access_token
from invokeai.app.services.users.users_common import UserCreateRequest, UserDTO, validate_email_with_special_domains

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


@auth_router.get("/status", response_model=SetupStatusResponse)
async def get_setup_status() -> SetupStatusResponse:
    """Check if initial administrator setup is required.

    Returns:
        SetupStatusResponse indicating whether setup is needed
    """
    user_service = ApiDependencies.invoker.services.users
    setup_required = not user_service.has_admin()

    return SetupStatusResponse(setup_required=setup_required)


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
    """
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
    """
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
        user = user_service.create_admin(user_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    return SetupResponse(success=True, user=user)
