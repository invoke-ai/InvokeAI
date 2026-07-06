from typing import Any

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class NoCacheStaticFiles(StaticFiles):
    """
    This class is used to override the default caching behavior of starlette for static files,
    ensuring we *never* cache static files. It modifies the file response headers to strictly
    never cache the files.

    Static files include the javascript bundles, fonts, locales, and some images. Generated
    images are not included, as they are served by a router.

    This class also implements proper SPA (Single Page Application) routing by serving index.html
    for any routes that don't match static files, enabling client-side routing to work correctly
    in production builds.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.cachecontrol = "max-age=0, no-cache, no-store, , must-revalidate"
        self.pragma = "no-cache"
        self.expires = "0"
        super().__init__(*args, **kwargs)

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers.setdefault("Cache-Control", self.cachecontrol)
        resp.headers.setdefault("Pragma", self.pragma)
        resp.headers.setdefault("Expires", self.expires)
        return resp

    async def get_response(self, path: str, scope: Scope) -> Response:
        """
        Override get_response to implement SPA routing.

        When a file is not found and html mode is enabled, serve index.html instead of raising a 404.
        This allows client-side routing to work correctly in SPAs.
        """
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            # If the file is not found (404) and html mode is enabled, serve index.html
            # This allows client-side routing to handle the path
            if exc.status_code == 404 and self.html:
                return await super().get_response("index.html", scope)
            raise
