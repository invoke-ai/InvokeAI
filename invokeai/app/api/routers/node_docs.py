from __future__ import annotations

import importlib.resources as pkg_resources
import mimetypes
import re
from importlib import import_module

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

router = APIRouter()

# validation regexes
VALID_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
IMAGE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@router.get("/nodeDocs/{lang}/{name}.md")
def get_node_doc(lang: str, name: str) -> PlainTextResponse:
    """Return packaged markdown for a node.

    This endpoint reads packaged resources from the installed `invokeai.resources`
    package via importlib.
    """
    # Basic validation
    if not VALID_SEGMENT.match(lang) or not VALID_SEGMENT.match(name):
        raise HTTPException(status_code=400, detail="Invalid path segment")

    try:
        res_pkg = import_module("invokeai.resources")
        pkg_path = pkg_resources.files(res_pkg).joinpath("node_docs", lang, f"{name}.md")
    except Exception as e:
        logger.warning(f"node_docs: unable to import packaged resources: {e}")
        raise HTTPException(status_code=404, detail="Not found")

    # Ensure resource exists in the package
    try:
        if not pkg_path.is_file():
            logger.debug(f"node_docs: resource not found in package: {pkg_path}")
            raise FileNotFoundError
        text = pkg_path.read_text(encoding="utf-8")
        return PlainTextResponse(content=text, media_type="text/markdown")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not found")
    except Exception as e:
        logger.warning(f"node_docs: failed reading resource {pkg_path}: {e}")
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/nodeDocs/{lang}/images/{image_name}")
def get_node_doc_image(lang: str, image_name: str) -> Response:
    """Return packaged image resource for node docs.

    Only reads from `invokeai.resources` packaged data. Adds `X-Content-Type-Options`
    and a conservative Cache-Control header.
    """
    # Validate inputs
    if not IMAGE_SEGMENT.match(image_name) or not VALID_SEGMENT.match(lang):
        raise HTTPException(status_code=400, detail="Invalid path segment")

    try:
        res_pkg = import_module("invokeai.resources")
        pkg_path = pkg_resources.files(res_pkg).joinpath("node_docs", lang, "images", image_name)
    except Exception as e:
        logger.warning(f"node_docs: unable to import packaged resources for image: {e}")
        raise HTTPException(status_code=404, detail="Not found")

    try:
        if not pkg_path.is_file():
            logger.debug(f"node_docs: image resource not found in package: {pkg_path}")
            raise FileNotFoundError
        data = pkg_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(image_name)
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "public, max-age=86400",
        }
        return Response(content=data, media_type=mime_type or "application/octet-stream", headers=headers)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not found")
    except Exception as e:
        logger.warning(f"node_docs: failed reading image resource {pkg_path}: {e}")
        raise HTTPException(status_code=404, detail="Not found")


# Expose the router
node_docs_router = router
