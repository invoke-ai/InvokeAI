import json
import logging
from dataclasses import dataclass

from PIL import Image

from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutIDValidator


@dataclass
class ExtractedMetadata:
    invokeai_metadata: str | None
    invokeai_workflow: str | None
    invokeai_graph: str | None


def extract_metadata_from_image(
    pil_image: Image.Image,
    invokeai_metadata_override: str | None,
    invokeai_workflow_override: str | None,
    invokeai_graph_override: str | None,
    logger: logging.Logger,
) -> ExtractedMetadata:
    """
    Extracts the "invokeai_metadata", "invokeai_workflow", and "invokeai_graph" data embedded in the PIL Image.

    These items are stored as stringified JSON in the image file's metadata, so we need to do some parsing to validate
    them. Once parsed, the values are returned as they came (as strings), or None if they are not present or invalid.

    In some situations, we may prefer to override the values extracted from the image file with some other values.

    For example, when uploading an image via API, the client can optionally provide the metadata directly in the request,
    as opposed to embedding it in the image file. In this case, the client-provided metadata will be used instead of the
    metadata embedded in the image file.

    Args:
        pil_image: The PIL Image object.
        invokeai_metadata_override: The metadata override provided by the client.
        invokeai_workflow_override: The workflow override provided by the client.
        invokeai_graph_override: The graph override provided by the client.
        logger: The logger to use for debug logging.

    Returns:
        ExtractedMetadata: The extracted metadata, workflow, and graph.
    """

    # The fallback value for metadata is None.
    stringified_metadata: str | None = None

    # Use the metadata override if provided, else attempt to extract it from the image file.
    metadata_raw = invokeai_metadata_override or pil_image.info.get("invokeai_metadata", None)

    # If the metadata is present in the image file, we will attempt to parse it as JSON. When we create images,
    # we always store metadata as a stringified JSON dict. So, we expect it to be a string here.
    if isinstance(metadata_raw, str):
        try:
            # Must be a JSON string
            metadata_parsed = json.loads(metadata_raw)
            # Must be a dict
            if isinstance(metadata_parsed, dict):
                # Looks good, overwrite the fallback value
                stringified_metadata = metadata_raw
        except Exception as e:
            logger.debug(f"Failed to parse metadata for uploaded image, {e}")
            pass

    # We expect the workflow, if embedded in the image, to be a JSON-stringified WorkflowWithoutID. We will store it
    # as a string.
    workflow_raw: str | None = invokeai_workflow_override or pil_image.info.get("invokeai_workflow", None)

    # The fallback value for workflow is None.
    stringified_workflow: str | None = None

    # If the workflow is present in the image file, we will attempt to parse it as JSON. When we create images, we
    # always store workflows as a stringified JSON WorkflowWithoutID. So, we expect it to be a string here.
    if isinstance(workflow_raw, str):
        try:
            # Validate the workflow JSON before storing it
            WorkflowWithoutIDValidator.validate_json(workflow_raw)
            # Looks good, overwrite the fallback value
            stringified_workflow = workflow_raw
        except Exception:
            logger.debug("Failed to parse workflow for uploaded image")
            pass

    # We expect the workflow, if embedded in the image, to be a JSON-stringified Graph. We will store it as a
    # string.
    graph_raw: str | None = invokeai_graph_override or pil_image.info.get("invokeai_graph", None)

    # The fallback value for graph is None.
    stringified_graph: str | None = None

    # If the graph is present in the image file, we will attempt to parse it as JSON. When we create images, we
    # always store graphs as a stringified JSON Graph. So, we expect it to be a string here.
    if isinstance(graph_raw, str):
        try:
            # TODO(psyche): Due to pydantic's handling of None values, it is possible for the graph to fail validation,
            # even if it is a direct dump of a valid graph. Node fields in the graph are allowed to have be unset if
            # they have incoming connections, but something about the ser/de process cannot adequately handle this.
            #
            # In lieu of fixing the graph validation, we will just do a simple check here to see if the graph is dict
            # with the correct keys. This is not a perfect solution, but it should be good enough for now.

            # FIX ME: Validate the graph JSON before storing it
            # Graph.model_validate_json(graph_raw)

            # Crappy workaround to validate JSON
            graph_parsed = json.loads(graph_raw)
            if not isinstance(graph_parsed, dict):
                raise ValueError("Not a dict")
            if not isinstance(graph_parsed.get("nodes", None), dict):
                raise ValueError("'nodes' is not a dict")
            if not isinstance(graph_parsed.get("edges", None), list):
                raise ValueError("'edges' is not a list")

            # Looks good, overwrite the fallback value
            stringified_graph = graph_raw
        except Exception as e:
            logger.debug(f"Failed to parse graph for uploaded image, {e}")
            pass

    return ExtractedMetadata(
        invokeai_metadata=stringified_metadata, invokeai_workflow=stringified_workflow, invokeai_graph=stringified_graph
    )
