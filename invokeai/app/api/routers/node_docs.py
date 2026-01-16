from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

import importlib.resources as pkg_resources
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

router = APIRouter()

VALID_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
IMAGE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")

RESOURCE_ROOT = Path("invokeai") / "resources" / "node_docs"


@router.get("/nodeDocs/{lang}/{name}.md")
def get_node_doc(lang: str, name: str) -> PlainTextResponse:
    # Basic validation
    if not VALID_SEGMENT.match(lang) or not VALID_SEGMENT.match(name):
        raise HTTPException(status_code=400, detail="Invalid path segment")

    # Attempt to load the resource from package resources
    try:
        # Build the package-relative path: resources/node_docs/{lang}/{name}.md
        # Use the resources package so importlib can find package data reliably
        from importlib import import_module

        pkg_path = None
        try:
            # Preferred: read from the installed package resources
            res_pkg = import_module("invokeai.resources")
            pkg_path = pkg_resources.files(res_pkg).joinpath("node_docs", lang, f"{name}.md")
            logger.debug(f"node_docs: attempting to read resource from package: {pkg_path}")
        except Exception as e:
            # Fall back to reading from the repository tree (useful when running from scripts)
            logger.warning(f"node_docs: failed to import invokeai.resources: {e}; falling back to repo filesystem")
            repo_root = Path(__file__).resolve().parents[4]
            pkg_path = repo_root.joinpath("invokeai", "resources", "node_docs", lang, f"{name}.md")
            logger.debug(f"node_docs: attempting to read resource from repo path: {pkg_path}")

        if not pkg_path.exists():
            logger.warning(f"node_docs: resource not found at {pkg_path}")
            raise FileNotFoundError
        try:
            text = pkg_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"node_docs: failed reading resource {pkg_path}: {e}")
            raise FileNotFoundError
        return PlainTextResponse(content=text, media_type="text/plain")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/nodeDocs/{lang}/images/{image_name}")
def get_node_doc_image(lang: str, image_name: str) -> Response:
    # Use a different validation for images to allow dots and file extensions
    if not IMAGE_SEGMENT.match(image_name):
        raise HTTPException(status_code=400, detail="Invalid path segment")
    # previous validation for lang remains below

    if not VALID_SEGMENT.match(lang):
        raise HTTPException(status_code=400, detail="Invalid path segment")

    try:
        from importlib import import_module

        pkg_path = None
        try:
            # Preferred: read from installed package resources
            res_pkg = import_module("invokeai.resources")
            pkg_path = pkg_resources.files(res_pkg).joinpath("node_docs", lang, "images", image_name)
            logger.debug(f"node_docs: attempting to read image resource from package: {pkg_path}")
        except Exception as e:
            logger.warning(
                f"node_docs: failed to import invokeai.resources for image: {e}; falling back to repo filesystem"
            )
            repo_root = Path(__file__).resolve().parents[4]
            pkg_path = repo_root.joinpath("invokeai", "resources", "node_docs", lang, "images", image_name)
            logger.debug(f"node_docs: attempting to read image resource from repo path: {pkg_path}")

        if not pkg_path.exists():
            logger.warning(f"node_docs: image resource not found at {pkg_path}")
            raise FileNotFoundError
        try:
            data = pkg_path.read_bytes()
        except Exception as e:
            logger.warning(f"node_docs: failed reading image resource {pkg_path}: {e}")
            raise FileNotFoundError
        mime_type, _ = mimetypes.guess_type(image_name)
        return Response(content=data, media_type=mime_type or "application/octet-stream")
    except FileNotFoundError:
        logger.debug(f"node_docs: returning 404 for image {lang}/{image_name}")
        raise HTTPException(status_code=404, detail="Not found")


# Expose the router
node_docs_router = router
