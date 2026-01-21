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

    This dependency is useful for endpoints that should work in both authenticated and non-authenticated contexts.
    In single-user mode or when authentication is not provided, it returns a TokenData for the 'system' user.

    Args:
        credentials: The HTTP authorization credentials containing the Bearer token

    Returns:
        TokenData containing user information from the token, or system user if no credentials
    """
    if credentials is None:
        # Return system user for unauthenticated requests (single-user mode or backwards compatibility)
        logger.debug("No authentication credentials provided, using system user")
        return TokenData(user_id="system", email="system@system.invokeai", is_admin=False)

    token = credentials.credentials
    token_data = verify_token(token)

    if token_data is None:
        # Invalid token - still fall back to system user for backwards compatibility
        logger.warning("Invalid or expired token provided, falling back to system user")
        return TokenData(user_id="system", email="system@system.invokeai", is_admin=False)

    # Verify user still exists and is active
    user_service = ApiDependencies.invoker.services.users
    user = user_service.get(token_data.user_id)

    if user is None or not user.is_active:
        # User doesn't exist or is inactive - fall back to system user
        logger.warning(f"User {token_data.user_id} does not exist or is inactive, falling back to system user")
        return TokenData(user_id="system", email="system@system.invokeai", is_admin=False)

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


# Type aliases for convenient use in route dependencies
CurrentUser = Annotated[TokenData, Depends(get_current_user)]
CurrentUserOrDefault = Annotated[TokenData, Depends(get_current_user_or_default)]
AdminUser = Annotated[TokenData, Depends(require_admin)]
