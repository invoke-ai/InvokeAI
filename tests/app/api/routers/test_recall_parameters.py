"""Tests for recall_parameters.load_image_file (Point 5).

Verifies that load_image_file uses the images service (which resolves subfolders
from the DB record) rather than accessing files directly.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.api.routers.recall_parameters import load_image_file


@pytest.fixture
def mock_api_deps():
    """Patch ApiDependencies.invoker with a mock that simulates subfolder-aware image service."""
    with patch("invokeai.app.api.routers.recall_parameters.ApiDependencies") as mock_deps:
        invoker = MagicMock()
        mock_deps.invoker = invoker

        images_service = invoker.services.images
        images_service.get_path.return_value = "/outputs/images/2026/04/05/test.png"
        images_service.validate_path.return_value = True

        pil_image = Image.new("RGB", (512, 768))
        images_service.get_pil_image.return_value = pil_image

        yield invoker


class TestLoadImageFile:
    def test_returns_image_info_for_subfolder_image(self, mock_api_deps: MagicMock):
        """load_image_file should work for images stored in subfolders."""
        result = load_image_file("test.png")

        assert result is not None
        assert result["image_name"] == "test.png"
        assert result["width"] == 512
        assert result["height"] == 768

        # Verify it used the images service (not image_files directly)
        mock_api_deps.services.images.get_path.assert_called_once_with("test.png")
        mock_api_deps.services.images.get_pil_image.assert_called_once_with("test.png")

    def test_returns_none_when_file_not_found(self, mock_api_deps: MagicMock):
        """load_image_file should return None if the resolved path doesn't exist."""
        mock_api_deps.services.images.validate_path.return_value = False

        result = load_image_file("missing.png")

        assert result is None

    def test_returns_none_on_service_exception(self, mock_api_deps: MagicMock):
        """load_image_file should return None if the images service raises."""
        mock_api_deps.services.images.get_path.side_effect = Exception("DB error")

        result = load_image_file("broken.png")

        assert result is None

    def test_uses_images_service_not_image_files(self, mock_api_deps: MagicMock):
        """Regression: load_image_file must go through images service (subfolder-aware),
        not image_files (flat-only)."""
        load_image_file("test.png")

        # image_files should NOT be called directly
        mock_api_deps.services.image_files.get.assert_not_called()
        mock_api_deps.services.image_files.get_path.assert_not_called()
