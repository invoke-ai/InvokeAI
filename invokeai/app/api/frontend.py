from pathlib import Path

from starlette.applications import Starlette

from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles


def mount_frontend(app: Starlette, frontend_root: Path, *, legacy: bool = False) -> None:
    """Mount shared documentation assets before the selected UI's catch-all SPA route."""
    ui_root = frontend_root.parent / "webv1" if legacy else frontend_root
    app.mount("/static", NoCacheStaticFiles(directory=frontend_root.parent / "webv1" / "static"), name="static")
    app.mount("/", NoCacheStaticFiles(directory=ui_root / "dist", html=True), name="ui")
