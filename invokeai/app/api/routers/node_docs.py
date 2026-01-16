from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

import importlib.resources as pkg_resources

router = APIRouter()

VALID_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")

RESOURCE_ROOT = Path("invokeai") / "resources" / "node_docs"


@router.get("/nodeDocs/{lang}/{name}.md")
def get_node_doc(lang: str, name: str) -> PlainTextResponse:
    # Basic validation
    if not VALID_SEGMENT.match(lang) or not VALID_SEGMENT.match(name):
        raise HTTPException(status_code=400, detail="Invalid path segment")

    # Attempt to load the resource from package resources
    try:
        # Build the package-relative path: resources/node_docs/{lang}/{name}.md
        pkg_path = pkg_resources.files("invokeai").joinpath("resources", "node_docs", lang, f"{name}.md")
        if not pkg_path.exists():
            raise FileNotFoundError
        text = pkg_path.read_text(encoding="utf-8")
        return PlainTextResponse(content=text, media_type="text/plain")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/nodeDocs/{lang}/images/{image_name}")
def get_node_doc_image(lang: str, image_name: str) -> FileResponse:
    if not VALID_SEGMENT.match(lang) or not VALID_SEGMENT.match(image_name):
        raise HTTPException(status_code=400, detail="Invalid path segment")

    try:
        pkg_path = pkg_resources.files("invokeai").joinpath("resources", "node_docs", lang, "images", image_name)
        if not pkg_path.exists():
            raise FileNotFoundError
        mime_type, _ = mimetypes.guess_type(str(pkg_path))
        return FileResponse(path=str(pkg_path), media_type=mime_type or "application/octet-stream")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Not found")


# Expose the router
node_docs_router = router
