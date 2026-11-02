"""Recovers the ``invokeai_metadata`` / ``invokeai_workflow`` / ``invokeai_graph`` strings a media
file carries, so an uploaded PNG or MP4 is recallable without the database that produced it.

PNGs hold the shared keys and optional ``invokeai_resolved_inputs`` as text chunks; MP4s hold
the three shared keys as QuickTime keyed metadata (see ``invokeai.app.util.mp4_metadata``).
Both sources use the shared validation in ``extract_metadata``. The image entry point also
reads resolved inputs using the same Graph JSON envelope validation as the source graph.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from invokeai.app.services.video_files.video_files_common import (
    EMBEDDED_METADATA_KEYS,
    GRAPH_KEY,
    METADATA_KEY,
    WORKFLOW_KEY,
)
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutIDValidator
from invokeai.app.util.mp4_metadata import read_mp4_tags


@dataclass
class ExtractedMetadata:
    invokeai_metadata: str | None
    invokeai_workflow: str | None
    invokeai_graph: str | None
    invokeai_resolved_inputs: str | None = None


def _extract_graph(raw: object, logger: logging.Logger) -> str | None:
    # Validate only the envelope: connected fields may be unset and community node types may be unavailable.
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Not a dict")
            if not isinstance(parsed.get("nodes"), dict):
                raise ValueError("'nodes' is not a dict")
            if not isinstance(parsed.get("edges"), list):
                raise ValueError("'edges' is not a list")
            return raw
        except Exception as e:
            logger.debug(f"Failed to parse graph for uploaded media, {e}")
    return None


def extract_metadata(
    embedded: Mapping[str, object],
    invokeai_metadata_override: str | None,
    invokeai_workflow_override: str | None,
    invokeai_graph_override: str | None,
    logger: logging.Logger,
) -> ExtractedMetadata:
    """
    Validates the "invokeai_metadata", "invokeai_workflow", and "invokeai_graph" strings found in a media file.

    ``embedded`` maps those keys to whatever the file carried. Each value is expected to be stringified JSON; a value
    that is missing, not a string, or fails validation is returned as None. Values are returned as they came (as
    strings), never re-serialized.

    In some situations, we may prefer to override the values extracted from the file with some other values. For
    example, when uploading via API, the client can optionally provide the metadata directly in the request, as
    opposed to embedding it in the file. In this case, the client-provided metadata will be used instead of the
    embedded metadata.
    """

    # The fallback value for metadata is None.
    stringified_metadata: str | None = None

    metadata_raw = invokeai_metadata_override if invokeai_metadata_override is not None else embedded.get(METADATA_KEY)

    # When we create media, we always store metadata as a stringified JSON dict. So, we expect it to be a string here.
    if isinstance(metadata_raw, str):
        try:
            # Must be a JSON string
            metadata_parsed = json.loads(metadata_raw)
            # Must be a dict
            if isinstance(metadata_parsed, dict):
                # Looks good, overwrite the fallback value
                stringified_metadata = metadata_raw
        except Exception as e:
            logger.debug(f"Failed to parse metadata for uploaded media, {e}")
            pass

    # We expect the workflow, if embedded, to be a JSON-stringified WorkflowWithoutID. We will store it as a string.
    workflow_raw = invokeai_workflow_override if invokeai_workflow_override is not None else embedded.get(WORKFLOW_KEY)

    # The fallback value for workflow is None.
    stringified_workflow: str | None = None

    if isinstance(workflow_raw, str):
        try:
            # Validate the workflow JSON before storing it
            WorkflowWithoutIDValidator.validate_json(workflow_raw)
            # Looks good, overwrite the fallback value
            stringified_workflow = workflow_raw
        except Exception:
            logger.debug("Failed to parse workflow for uploaded media")
            pass

    # We expect the graph, if embedded, to be a JSON-stringified Graph. We will store it as a string.
    graph_raw = invokeai_graph_override if invokeai_graph_override is not None else embedded.get(GRAPH_KEY)

    stringified_graph = _extract_graph(graph_raw, logger)

    return ExtractedMetadata(
        invokeai_metadata=stringified_metadata, invokeai_workflow=stringified_workflow, invokeai_graph=stringified_graph
    )


def extract_metadata_from_image(
    pil_image: Image.Image,
    invokeai_metadata_override: str | None,
    invokeai_workflow_override: str | None,
    invokeai_graph_override: str | None,
    logger: logging.Logger,
) -> ExtractedMetadata:
    """Extracts the three shared channels and optional resolved-input Graph JSON from an image's text chunks."""
    extracted = extract_metadata(
        pil_image.info,
        invokeai_metadata_override,
        invokeai_workflow_override,
        invokeai_graph_override,
        logger,
    )
    extracted.invokeai_resolved_inputs = _extract_graph(pil_image.info.get("invokeai_resolved_inputs"), logger)
    return extracted


def extract_metadata_from_video(
    video_path: Path,
    invokeai_metadata_override: str | None,
    invokeai_workflow_override: str | None,
    invokeai_graph_override: str | None,
    logger: logging.Logger,
) -> ExtractedMetadata:
    """Extracts and validates the metadata, workflow and graph embedded in an MP4's keyed metadata.

    A file that is not an MP4 or carries no tags simply yields all-None (or the overrides).
    """
    try:
        embedded = read_mp4_tags(video_path, keys=EMBEDDED_METADATA_KEYS)
    except OSError as e:
        logger.debug(f"Could not read embedded metadata from uploaded video, {e}")
        embedded = {}
    return extract_metadata(
        embedded,
        invokeai_metadata_override,
        invokeai_workflow_override,
        invokeai_graph_override,
        logger,
    )
