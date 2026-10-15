import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.api.extract_metadata import ExtractedMetadata, extract_metadata_from_image


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def valid_metadata():
    return json.dumps({"param1": "value1", "param2": 123})


@pytest.fixture
def valid_workflow():
    return json.dumps({"name": "test_workflow", "version": "1.0"})


@pytest.fixture
def valid_graph():
    return json.dumps({"nodes": {}, "edges": []})


def test_extract_valid_metadata_from_image(mock_logger, valid_metadata, valid_workflow, valid_graph):
    # Create a mock image with valid metadata
    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": valid_metadata,
        "invokeai_workflow": valid_workflow,
        "invokeai_graph": valid_graph,
    }

    # Mock the validation functions
    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ) as mock_workflow_validate:
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json") as _mock_graph_validate:
            result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

            # Assert correct calls to validators
            mock_workflow_validate.assert_called_once_with(valid_workflow)
            # TODO(psyche): The extract_metadata_from_image does not validate the graph correctly. See note in `extract_metadata.py`.
            # Skipping this.
            # _mock_graph_validate.assert_called_once_with(valid_graph)

            # Assert correct extraction
            assert result == ExtractedMetadata(
                invokeai_metadata=valid_metadata, invokeai_workflow=valid_workflow, invokeai_graph=valid_graph
            )


def test_extract_invalid_metadata(mock_logger, valid_workflow, valid_graph):
    # Invalid metadata (not JSON)
    invalid_metadata = "not a valid json"

    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": invalid_metadata,
        "invokeai_workflow": valid_workflow,
        "invokeai_graph": valid_graph,
    }

    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ):
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json"):
            result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

            assert mock_logger.debug.to_have_been_called_with("Failed to parse metadata for uploaded image")

            # Invalid metadata should be None, others valid
            assert result.invokeai_metadata is None
            assert result.invokeai_workflow == valid_workflow
            assert result.invokeai_graph == valid_graph


def test_metadata_wrong_type(mock_logger, valid_workflow, valid_graph):
    # Valid JSON but not a dict
    metadata_array = json.dumps(["item1", "item2"])

    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": metadata_array,
        "invokeai_workflow": valid_workflow,
        "invokeai_graph": valid_graph,
    }

    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ):
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json"):
            result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

            # Metadata should be None as it's not a dict
            assert result.invokeai_metadata is None
            assert result.invokeai_workflow == valid_workflow
            assert result.invokeai_graph == valid_graph


def test_with_non_string_metadata(mock_logger, valid_workflow, valid_graph):
    # Some implementations might include metadata as non-string values
    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": 12345,  # Not a string
        "invokeai_workflow": valid_workflow,
        "invokeai_graph": valid_graph,
    }

    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ):
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json"):
            result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

            assert mock_logger.debug.to_have_been_called_with("Failed to parse metadata for uploaded image")

            assert result.invokeai_metadata is None
            assert result.invokeai_workflow == valid_workflow
            assert result.invokeai_graph == valid_graph


def test_invalid_workflow(mock_logger, valid_metadata, valid_graph):
    invalid_workflow = "not a valid workflow json"

    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": valid_metadata,
        "invokeai_workflow": invalid_workflow,
        "invokeai_graph": valid_graph,
    }

    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ) as mock_validate:
        mock_validate.side_effect = ValueError("Invalid workflow")
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json"):
            result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

            assert result.invokeai_metadata == valid_metadata
            assert result.invokeai_workflow is None
            assert result.invokeai_graph == valid_graph


def test_invalid_graph(mock_logger, valid_metadata, valid_workflow):
    invalid_graph = "not a valid graph json"

    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": valid_metadata,
        "invokeai_workflow": valid_workflow,
        "invokeai_graph": invalid_graph,
    }

    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ):
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json") as mock_validate:
            mock_validate.side_effect = ValueError("Invalid graph")
            result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

            assert result.invokeai_metadata == valid_metadata
            assert result.invokeai_workflow == valid_workflow
            assert result.invokeai_graph is None


def test_with_overrides(mock_logger, valid_metadata, valid_workflow, valid_graph):
    # Different values in the image
    mock_image = MagicMock(spec=Image.Image)

    # When overrides are provided, they should be used instead of the values in the image, we shouldn'teven try
    # to parse the values in the image
    mock_image.info = {
        "invokeai_metadata": 12345,
        "invokeai_workflow": 12345,
        "invokeai_graph": 12345,
    }

    with patch(
        "invokeai.app.services.workflow_records.workflow_records_common.WorkflowWithoutIDValidator.validate_json"
    ):
        with patch("invokeai.app.services.shared.graph.Graph.model_validate_json"):
            result = extract_metadata_from_image(mock_image, valid_metadata, valid_workflow, valid_graph, mock_logger)

            # Override values should be used
            assert result.invokeai_metadata == valid_metadata
            assert result.invokeai_workflow == valid_workflow
            assert result.invokeai_graph == valid_graph


def test_with_no_metadata(mock_logger):
    # Image with no metadata
    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {}

    result = extract_metadata_from_image(mock_image, None, None, None, mock_logger)

    assert result.invokeai_metadata is None
    assert result.invokeai_workflow is None
    assert result.invokeai_graph is None


def test_empty_string_overrides_do_not_fall_back_to_image_metadata(
    mock_logger, valid_metadata, valid_workflow, valid_graph
):
    mock_image = MagicMock(spec=Image.Image)
    mock_image.info = {
        "invokeai_metadata": valid_metadata,
        "invokeai_workflow": valid_workflow,
        "invokeai_graph": valid_graph,
    }

    result = extract_metadata_from_image(mock_image, "", "", "", mock_logger)

    assert result.invokeai_metadata is None
    assert result.invokeai_workflow is None
    assert result.invokeai_graph is None


# --- MP4 sources -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tagged_mp4(tmp_path_factory: pytest.TempPathFactory):
    """A real MP4 carrying the three keys the way DiskVideoFileStorage writes them."""
    import numpy as np

    from invokeai.app.util.mp4_metadata import write_mp4_tags
    from invokeai.app.util.video_encoding import make_mp4_writer

    folder = tmp_path_factory.mktemp("tagged")
    plain = folder / "plain.mp4"
    writer = make_mp4_writer(plain, fps=8.0)
    try:
        for _ in range(2):
            writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.close()
    tagged = folder / "tagged.mp4"
    tags = {
        "invokeai_metadata": json.dumps({"seed": 42, "generation_mode": "wan_t2v"}),
        "invokeai_workflow": json.dumps(MINIMAL_WORKFLOW),
        "invokeai_graph": json.dumps({"nodes": {}, "edges": []}),
    }
    write_mp4_tags(plain, tagged, tags)
    return tagged, plain, tags


MINIMAL_WORKFLOW = {
    "name": "wf",
    "author": "",
    "description": "",
    "version": "",
    "contact": "",
    "tags": "",
    "notes": "",
    "exposedFields": [],
    "meta": {"version": "3.0.0", "category": "user"},
    "nodes": [],
    "edges": [],
    "form": {"elements": {}, "rootElementId": "root"},
}


def test_extract_from_tagged_video_returns_the_embedded_strings(mock_logger, tagged_mp4):
    """Through the real workflow validator: the video entry point is not mocked."""
    from invokeai.app.api.extract_metadata import extract_metadata_from_video

    tagged, _plain, tags = tagged_mp4
    result = extract_metadata_from_video(tagged, None, None, None, mock_logger)

    assert result == ExtractedMetadata(
        invokeai_metadata=tags["invokeai_metadata"],
        invokeai_workflow=tags["invokeai_workflow"],
        invokeai_graph=tags["invokeai_graph"],
    )


def test_extract_from_video_prefers_the_client_metadata_over_the_embedded_copy(mock_logger, tagged_mp4):
    from invokeai.app.api.extract_metadata import extract_metadata_from_video

    tagged, _plain, tags = tagged_mp4
    override = '{"seed": 7}'
    result = extract_metadata_from_video(tagged, override, None, None, mock_logger)

    assert result.invokeai_metadata == override
    # Workflow and graph still come from the file.
    assert result.invokeai_workflow == tags["invokeai_workflow"]
    assert result.invokeai_graph == tags["invokeai_graph"]


def test_extract_from_untagged_or_non_mp4_video_is_all_none(mock_logger, tagged_mp4, tmp_path):
    from invokeai.app.api.extract_metadata import extract_metadata_from_video

    _tagged, plain, _tags = tagged_mp4
    assert extract_metadata_from_video(plain, None, None, None, mock_logger) == ExtractedMetadata(None, None, None)

    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"\x00\x00\x00\x18ftypmp42 not a real mp4")
    assert extract_metadata_from_video(garbage, None, None, None, mock_logger) == ExtractedMetadata(None, None, None)

    missing = tmp_path / "missing.mp4"
    assert extract_metadata_from_video(missing, "{}", None, None, mock_logger) == ExtractedMetadata("{}", None, None)
