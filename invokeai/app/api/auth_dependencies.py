"""FastAPI dependencies for authentication."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.auth.token_service import TokenData, verify_token
from invokeai.backend.util.logging import logging

logger = logging.getLogger(__name__)

# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> TokenData:
    """Get current authenticated user from Bearer token.

    Note: This function accesses ApiDependencies.invoker.services.users directly,
    which is the established pattern in this codebase. The ApiDependencies.invoker
    is initialized in the FastAPI lifespan context before any requests are handled.

    Args:
        credentials: The HTTP authorization credentials containing the Bearer token

    Returns:
        TokenData containing user information from the token

    Raises:
        HTTPException: If token is missing, invalid, or expired (401 Unauthorized)
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    token_data = verify_token(token)

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user still exists and is active
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(token_data.user_id)

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive or does not exist",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_data


async def get_current_user_or_default(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> TokenData:
    """Get current authenticated user from Bearer token, or return a default system user if not authenticated.

    This dependency is useful for endpoints that should work in both single-user and multiuser modes.

    When multiuser mode is disabled (default), this always returns a system user with admin privileges,
    allowing unrestricted access to all operations.

    When multiuser mode is enabled, authentication is required and this function validates the token,
    returning authenticated user data or raising 401 Unauthorized if no valid credentials are provided.

    Args:
        credentials: The HTTP authorization credentials containing the Bearer token

    Returns:
        TokenData containing user information from the token, or system user in single-user mode

    Raises:
        HTTPException: 401 Unauthorized if in multiuser mode and credentials are missing, invalid, or user is inactive
    """
    # Get configuration to check if multiuser is enabled
    config = ApiDependencies.invoker.services.configuration

    # In single-user mode (multiuser=False), always return system user with admin privileges
    if not config.multiuser:
        return TokenData(user_id="system", email="system@system.invokeai", is_admin=True)

    # Multiuser mode is enabled - validate credentials
    if credentials is None:
        # In multiuser mode, authentication is required
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    token = credentials.credentials
    token_data = verify_token(token)

    if token_data is None:
        # Invalid token in multiuser mode - reject
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # Verify user still exists and is active
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(token_data.user_id)

    if user is None or not user.is_active:
        # User doesn't exist or is inactive in multiuser mode - reject
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return token_data


async def require_admin(
    current_user: Annotated[TokenData, Depends(get_current_user)],
) -> TokenData:
    """Require admin role for the current user.

    Args:
        current_user: The current authenticated user's token data

    Returns:
        The token data if user is an admin

    Raises:
        HTTPException: If user does not have admin privileges (403 Forbidden)
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user


async def require_admin_or_default(
    current_user: Annotated[TokenData, Depends(get_current_user_or_default)],
) -> TokenData:
    """Require admin role for the current user, or return default system admin in single-user mode.

    This dependency is useful for admin-only endpoints that should work in both single-user and multiuser modes.

    When multiuser mode is disabled (default), this always returns a system user with admin privileges.
    When multiuser mode is enabled, this validates that the authenticated user has admin privileges.

    Args:
        current_user: The current authenticated user's token data (or default system user)

    Returns:
        The token data if user is an admin (or system user in single-user mode)

    Raises:
        HTTPException: If user does not have admin privileges (403 Forbidden) in multiuser mode
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user


# Type aliases for convenient use in route dependencies
CurrentUser = Annotated[TokenData, Depends(get_current_user)]
CurrentUserOrDefault = Annotated[TokenData, Depends(get_current_user_or_default)]
AdminUser = Annotated[TokenData, Depends(require_admin)]
AdminUserOrDefault = Annotated[TokenData, Depends(require_admin_or_default)]
