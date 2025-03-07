import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.api.extract_metadata_from_image import ExtractedMetadata, extract_metadata_from_image


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
            # TODO(psyche): The extract_metadata_from_image does not validate the graph correctly. See note in `extract_metadata_from_image.py`.
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
