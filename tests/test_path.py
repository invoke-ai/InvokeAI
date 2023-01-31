import unittest
from os import path as osp
import pathlib

import invokeai.frontend.dist as frontend
import invokeai.configs as configs


def get_frontend_path() -> pathlib.Path:
    """Get the path of the frontend"""
    return pathlib.Path(frontend.__path__[0])


def get_configs_path() -> pathlib.Path:
    """Get the path of the configs folder"""
    return pathlib.Path(configs.__path__[0])


def test_frontend_path():
    """Test that the frontend path is correct"""
    # test path of the frontend
    TEST_PATH = str(get_frontend_path())
    assert TEST_PATH.endswith(osp.join("invokeai", "frontend", "dist"))


def test_configs_path():
    """Test that the frontend path is correct"""
    # test path of the frontend
    TEST_PATH = str(get_configs_path())
    assert TEST_PATH.endswith(osp.join("invokeai", "configs"))


if __name__ == "__main__":
    unittest.main()
