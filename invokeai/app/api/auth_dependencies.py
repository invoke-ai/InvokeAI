"""FastAPI dependencies for authentication."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.auth.token_service import TokenData, verify_token

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
AdminUser = Annotated[TokenData, Depends(require_admin)]
