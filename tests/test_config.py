import os
import pytest
import sys

from omegaconf import OmegaConf
from pathlib import Path

os.environ["INVOKEAI_ROOT"] = "/tmp"

from invokeai.app.services.config import InvokeAIAppConfig

init1 = OmegaConf.create(
    """
InvokeAI:
  Features:
    always_use_cpu: false
  Memory/Performance:
    max_cache_size: 5
    tiled_decode: false
"""
)

init2 = OmegaConf.create(
    """
InvokeAI:
  Features:
    always_use_cpu: true
  Memory/Performance:
    max_cache_size: 2
    tiled_decode: true
"""
)


def test_use_init():
    # note that we explicitly set omegaconf dict and argv here
    # so that the values aren't read from ~invokeai/invokeai.yaml and
    # sys.argv respectively.
    conf1 = InvokeAIAppConfig.get_config()
    assert conf1
    conf1.parse_args(conf=init1, argv=[])
    assert not conf1.tiled_decode
    assert conf1.max_cache_size == 5
    assert not conf1.always_use_cpu

    conf2 = InvokeAIAppConfig.get_config()
    assert conf2
    conf2.parse_args(conf=init2, argv=[])
    assert conf2.tiled_decode
    assert conf2.max_cache_size == 2
    assert not hasattr(conf2, "invalid_attribute")


def test_argv_override():
    conf = InvokeAIAppConfig.get_config()
    conf.parse_args(conf=init1, argv=["--always_use_cpu", "--max_cache=10"])
    assert conf.always_use_cpu
    assert conf.max_cache_size == 10
    assert conf.outdir == Path("outputs")  # this is the default


def test_env_override():
    # argv overrides
    conf = InvokeAIAppConfig()
    conf.parse_args(conf=init1, argv=["--max_cache=10"])
    assert conf.always_use_cpu == False
    os.environ["INVOKEAI_always_use_cpu"] = "True"
    conf.parse_args(conf=init1, argv=["--max_cache=10"])
    assert conf.always_use_cpu == True

    # environment variables should be case insensitive
    os.environ["InvokeAI_Max_Cache_Size"] = "15"
    conf = InvokeAIAppConfig()
    conf.parse_args(conf=init1, argv=[])
    assert conf.max_cache_size == 15

    conf = InvokeAIAppConfig()
    conf.parse_args(conf=init1, argv=["--no-always_use_cpu", "--max_cache=10"])
    assert conf.always_use_cpu == False
    assert conf.max_cache_size == 10

    conf = InvokeAIAppConfig.get_config(max_cache_size=20)
    conf.parse_args(conf=init1, argv=[])
    assert conf.max_cache_size == 20


def test_type_coercion():
    conf = InvokeAIAppConfig().get_config()
    conf.parse_args(argv=["--root=/tmp/foobar"])
    assert conf.root == Path("/tmp/foobar")
    assert isinstance(conf.root, Path)
    conf = InvokeAIAppConfig.get_config(root="/tmp/different")
    conf.parse_args(argv=["--root=/tmp/foobar"])
    assert conf.root == Path("/tmp/different")
    assert isinstance(conf.root, Path)
