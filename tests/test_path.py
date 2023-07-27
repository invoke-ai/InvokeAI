"""
Not really a test, but a way to verify that the paths are existing
and fail early if they are not.
"""
import unittest
import pathlib
from os import path as osp
from PIL import Image

import invokeai.frontend.web.dist as frontend
import invokeai.configs as configs
import invokeai.app.assets.images as image_assets


class ConfigsTestCase(unittest.TestCase):
    """Test the configuration related imports and objects"""

    def get_configs_path(self) -> pathlib.Path:
        """Get the path of the configs folder"""
        configs_path = pathlib.Path(configs.__path__[0])
        return configs_path

    def get_frontend_path(self) -> pathlib.Path:
        """Get the path of the frontend dist folder"""
        return pathlib.Path(frontend.__path__[0])

    def test_configs_path(self):
        """Test that the configs path is correct"""
        TEST_PATH = str(self.get_configs_path())
        assert TEST_PATH.endswith(str(osp.join("invokeai", "configs")))

    def test_frontend_path(self):
        """Test that the frontend path is correct"""
        FRONTEND_PATH = str(self.get_frontend_path())
        assert FRONTEND_PATH.endswith(osp.join("invokeai", "frontend", "web", "dist"))

    def test_caution_img(self):
        """Verify the caution image"""
        caution_img = Image.open(osp.join(image_assets.__path__[0], "caution.png"))
        assert caution_img.width == int(500)
        assert caution_img.height == int(441)
        assert caution_img.format == str("PNG")


if __name__ == "__main__":
    unittest.main(
        verbosity=2,
    )
