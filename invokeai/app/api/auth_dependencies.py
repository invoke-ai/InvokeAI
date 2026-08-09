"""FastAPI dependencies for authentication."""

from typing import TYPE_CHECKING, Annotated

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.auth.token_service import TokenData, verify_token
from invokeai.app.services.users.users_common import SYSTEM_USER_ID
from invokeai.backend.util.logging import logging

if TYPE_CHECKING:
    from invokeai.app.services.users.users_common import UserDTO

logger = logging.getLogger(__name__)

# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)
MEDIA_TOKEN_COOKIE = "invokeai_media_token"
# Deliberately indistinguishable from an ordinary expiry to a client: a token that fails
# the epoch check is simply no longer valid, and saying *why* would tell a holder of a
# stolen token that the account's password was just rotated.
TOKEN_REVOKED_DETAIL = "Invalid or expired authentication token"


def resolve_authorized_user(token_data: TokenData) -> "UserDTO | None":
    """Return the account a verified token still grants access to, or None.

    This is the single place that decides whether a syntactically valid token is still
    honored, and every authenticated entry point must go through it: the REST
    dependencies below, the Socket.IO handshake, and the video-upload ASGI gate. Keeping
    the rules in one function is deliberate — they were previously repeated at each call
    site, and a check added to some copies but not others is indistinguishable from no
    check at all on the paths that were missed.

    A token is honored when all four hold:

    - it does not claim the internal ``system`` account,
    - the account still exists,
    - it is active,
    - and the token carries the account's current revocation epoch. Any mismatch counts
      as revoked: the token was not issued from the record as it now stands. Tokens
      predating the claim decode to 0 and so remain valid against a record that has never
      been bumped, which is why upgrading logs nobody out.

    Raises whatever the user service raises; callers that must fail closed should catch.
    """
    # `system` owns everything carried over from before multiuser support, and is not a
    # login account: `UserService.authenticate` refuses it and the migration clears any
    # password left on the row. Neither of those reaches a token that was *already issued*
    # — on an instance that set a password through the old `PATCH /auth/users/system` hole
    # and logged in before it was closed, the JWT survives the upgrade, and nothing else
    # here would reject it: the row is deliberately kept active and its epoch still
    # matches, so the sliding-window middleware would renew it indefinitely. Refusing the
    # id is what actually ends those sessions, and it holds for databases that applied an
    # earlier revision of the migration too.
    #
    # Single-user mode, where everything legitimately runs as `system`, never reaches here:
    # its dependencies synthesize the TokenData and return before resolving anything (see
    # `get_current_user_or_default`, `get_current_media_user_or_default`, and
    # `_identify_video_upload_user`). So this refuses only real, minted tokens.
    if token_data.user_id == SYSTEM_USER_ID:
        return None
    user = ApiDependencies.invoker.services.users.get(token_data.user_id)
    if user is None or not user.is_active:
        return None
    if token_data.token_epoch != user.token_epoch:
        return None
    return user


def _validate_token(token: str, invalid_detail: str) -> TokenData:
    token_data = verify_token(token)
    if token_data is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=invalid_detail)

    user = resolve_authorized_user(token_data)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return _db_derived_token_data(token_data, user)


def _db_derived_token_data(token_data: TokenData, user: "UserDTO") -> TokenData:
    """Build TokenData whose authorization fields come from the database record.

    The JWT proves *identity* only. Authorization (``is_admin``) must reflect the
    current database state on every request; otherwise a demoted administrator
    keeps admin rights until their token expires — and sliding-window refresh
    would renew that stale claim indefinitely. A promoted user symmetrically
    gains admin rights on their next request without re-login.

    The epoch is carried through from the record so a refreshed token stays valid
    (callers only reach here once ``_token_epoch_is_current`` has passed).
    """
    return TokenData(
        user_id=user.user_id,
        email=user.email,
        is_admin=user.is_admin,
        remember_me=token_data.remember_me,
        token_epoch=user.token_epoch,
    )


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

    # Verify the token still grants access: user exists, is active, epoch is current.
    user = resolve_authorized_user(token_data)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive or does not exist",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _db_derived_token_data(token_data, user)


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

    # Verify the token still grants access: user exists, is active, epoch is current.
    user = resolve_authorized_user(token_data)

    if user is None:
        # Missing, inactive, or revoked in multiuser mode - reject
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return _db_derived_token_data(token_data, user)


async def get_current_media_user_or_default(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    media_token: Annotated[str | None, Cookie(alias=MEDIA_TOKEN_COOKIE)] = None,
) -> TokenData:
    """Authenticate media requests with a Bearer token or the path-scoped login cookie."""
    if not ApiDependencies.invoker.services.configuration.multiuser:
        return TokenData(user_id="system", email="system@system.invokeai", is_admin=True)
    token = credentials.credentials if credentials is not None else media_token
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return _validate_token(token, "Invalid or expired token")


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
CurrentMediaUserOrDefault = Annotated[TokenData, Depends(get_current_media_user_or_default)]
AdminUser = Annotated[TokenData, Depends(require_admin)]
AdminUserOrDefault = Annotated[TokenData, Depends(require_admin_or_default)]
