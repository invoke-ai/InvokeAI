import pathlib
import unittest
from os import path as osp

import invokeai.configs as configs
import invokeai.frontend.dist as frontend


def get_frontend_path() -> pathlib.Path:
    """Get the path of the frontend dist folder"""
    return pathlib.Path(frontend.__path__[0])


def get_configs_path() -> pathlib.Path:
    """Get the path of the configs folder"""
    return pathlib.Path(configs.__path__[0])


def test_frontend_path():
    """Test that the frontend path is correct"""
    TEST_PATH = str(get_frontend_path())
    assert TEST_PATH.endswith(osp.join("invokeai", "frontend", "dist"))


def test_configs_path():
    """Test that the configs path is correct"""
    TEST_PATH = str(get_configs_path())
    assert TEST_PATH.endswith(osp.join("invokeai", "configs"))


if __name__ == "__main__":
    unittest.main()
