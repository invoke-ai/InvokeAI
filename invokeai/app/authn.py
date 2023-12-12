import base64
import binascii

from fastapi import APIRouter
from starlette import status
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.responses import HTMLResponse, PlainTextResponse

from invokeai.app.services.config import InvokeAIAppConfig

BASIC_AUTH_REALM = "InvokeAI"  # might want to make this configurable


class AuthenticationRequired(AuthenticationError):
    pass


class BasicAuthBackend(AuthenticationBackend):
    """Single-user username/password authentication."""

    def __init__(self, app_config: InvokeAIAppConfig):
        self._app_config = app_config
        super().__init__()

    async def authenticate(self, conn):
        if self._app_config.password is None:
            return None

        # TODO: if connection is not secure, reject it as unsuitable for basic auth

        if "Authorization" not in conn.headers:
            raise AuthenticationRequired()

        # Mostly copy/pasted from the starlette docs. It doesn't ship an implementation of this?
        auth = conn.headers["Authorization"]
        try:
            scheme, credentials = auth.split()
            if scheme.lower() != "basic":
                # FIXME: this looks like its designed to fall-through to let another handler deal
                #   with other schemes, but will that bypass our auth requirements? Or will the lack
                #   of an authenticated user object prevent the request from getting through?
                return

            decoded = base64.b64decode(credentials).decode("ascii")
        except (ValueError, UnicodeDecodeError, binascii.Error):
            raise AuthenticationError("Invalid basic auth credentials")

        username, _, password = decoded.partition(":")

        if username != self._app_config.username:
            raise AuthenticationError("Invalid username or password")
        if password != self._app_config.password.get_secret_value():
            raise AuthenticationError("Invalid username or password")
        return AuthCredentials(["authenticated"]), SimpleUser(username)


def auth_required(conn, exc: Exception):
    if isinstance(exc, AuthenticationRequired):
        message = "Authentication Required"
    else:
        message = str(exc)
    return PlainTextResponse(
        message,
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": f"Basic realm='{BASIC_AUTH_REALM}'"},
    )


authn_router = APIRouter()


@authn_router.post("/logout")
def logout():
    """Always sends UNAUTHORIZED, to encourage the browser to discard the current credentials."""
    return PlainTextResponse(
        "Hopefully this reset your browser's authentication credentials!",
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": f"Basic realm='{BASIC_AUTH_REALM}'"},
    )


@authn_router.get("/logout")
def logout_prompt():
    # FIXME: How does CRSF-mitigation work in this app?
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <form action="/logout" method="post">
        <input type="SUBMIT" value="Logout">Logout</input>
        </form>
        """
    )
