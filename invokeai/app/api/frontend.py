import hashlib
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import FileResponse, HTMLResponse, Response
from starlette.staticfiles import NotModifiedResponse
from starlette.types import Scope

from invokeai.app.api.desktop_theme import DESKTOP_THEME_MARKER, read_desktop_theme_style
from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles


def _render_desktop_index(path: str) -> bytes | None:
    style = read_desktop_theme_style()
    if not style:
        return None
    try:
        html = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if DESKTOP_THEME_MARKER not in html:
        return None
    return html.replace(DESKTOP_THEME_MARKER, style, 1).encode("utf-8")


class FrontendStaticFiles(NoCacheStaticFiles):
    """Preserve the static SPA boundary while supplying fresh local colors only with the webv2 entry document."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        # An index-file validator cannot describe a changing desktop palette. Validate the final representation.
        unconditional_scope = {
            **scope,
            "headers": [
                (name, value)
                for name, value in scope["headers"]
                if name.lower() not in (b"if-none-match", b"if-modified-since")
            ],
        }
        response = await super().get_response(path, unconditional_scope)
        if not isinstance(response, FileResponse):
            return response
        if Path(response.path).name == "index.html":
            content = await run_in_threadpool(_render_desktop_index, str(response.path))
            if content is not None:
                headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key not in ("content-length", "etag", "last-modified", "accept-ranges")
                }
                headers["etag"] = f'"{hashlib.sha256(content).hexdigest()}"'
                response = HTMLResponse(content, headers=headers)
                if scope["method"] == "HEAD":
                    response.body = b""
        if response.status_code == 200 and self.is_not_modified(response.headers, Headers(scope=scope)):
            return NotModifiedResponse(response.headers)
        return response


def mount_frontend(app: Starlette, frontend_root: Path, *, legacy: bool = False) -> None:
    """Mount shared documentation assets before the selected UI's catch-all SPA route."""
    ui_root = frontend_root.parent / "webv1" if legacy else frontend_root
    app.mount("/static", NoCacheStaticFiles(directory=frontend_root.parent / "webv1" / "static"), name="static")
    static_files = NoCacheStaticFiles if legacy else FrontendStaticFiles
    app.mount("/", static_files(directory=ui_root / "dist", html=True), name="ui")
