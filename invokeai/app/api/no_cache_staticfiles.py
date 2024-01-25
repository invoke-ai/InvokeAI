from typing import Any

from starlette.responses import Response
from starlette.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    """
    This class is used to override the default caching behavior of starlette for static files,
    ensuring we *never* cache static files. It modifies the file response headers to strictly
    never cache the files.

    Static files include the javascript bundles, fonts, locales, and some images. Generated
    images are not included, as they are served by a router.
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
