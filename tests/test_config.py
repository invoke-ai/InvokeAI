import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from invokeai.app.services.config.config_default import InvokeAIAppConfig, get_config, load_and_migrate_config
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

v4_config = """
schema_version: 4

host: "192.168.1.1"
port: 8080
"""

invalid_v5_config = """
schema_version: 5

host: "192.168.1.1"
port: 8080
"""


v3_config = """
InvokeAI:
  Web Server:
    host: 192.168.1.1
    port: 8080
  Features:
    esrgan: true
    internet_available: true
    log_tokenization: false
    patchmatch: true
    ignore_missing_core_models: false
  Paths:
    outdir: /some/outputs/dir
    conf_path: /custom/models.yaml
  Model Cache:
    max_cache_size: 100
    max_vram_cache_size: 50
"""

v3_config_with_bad_values = """
InvokeAI:
  Web Server:
    port: "ice cream"
"""

invalid_config = """
i like turtles
"""


@pytest.fixture
def patch_rootdir(tmp_path: Path, monkeypatch: Any) -> None:
    """This may be overkill since the current tests don't need the root dir to exist"""
    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))


def test_path_resolution_root_not_set():
    """Test path resolutions when the root is not explicitly set."""
    config = InvokeAIAppConfig()
    expected_root = InvokeAIAppConfig.find_root()
    assert config.root_path == expected_root


def test_read_config_from_file(tmp_path: Path):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v4_config)

    config = load_and_migrate_config(temp_config_file)
    assert config.host == "192.168.1.1"
    assert config.port == 8080


def test_arg_parsing():
    sys.argv = ["test_config.py", "--root", "/tmp"]
    InvokeAIArgs.parse_args()
    config = get_config()
    assert config.root_path == Path("/tmp")


def test_migrate_v3_config_from_file(tmp_path: Path):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v3_config)

    config = load_and_migrate_config(temp_config_file)
    assert config.outputs_dir == Path("/some/outputs/dir")
    assert config.host == "192.168.1.1"
    assert config.port == 8080
    assert config.ram == 100
    assert config.vram == 50
    assert config.legacy_models_yaml_path == Path("/custom/models.yaml")
    # This should be stripped out
    assert not hasattr(config, "esrgan")


def test_migrate_v3_backup(tmp_path: Path):
    """Test the backup of the config file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v3_config)

    load_and_migrate_config(temp_config_file)
    assert temp_config_file.with_suffix(".yaml.bak").exists()
    assert temp_config_file.with_suffix(".yaml.bak").read_text() == v3_config


def test_failed_migrate_backup(tmp_path: Path):
    """Test the failed migration of the config file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v3_config_with_bad_values)

    with pytest.raises(RuntimeError):
        load_and_migrate_config(temp_config_file)
    assert temp_config_file.with_suffix(".yaml.bak").exists()
    assert temp_config_file.with_suffix(".yaml.bak").read_text() == v3_config_with_bad_values
    assert temp_config_file.exists()
    assert temp_config_file.read_text() == v3_config_with_bad_values


def test_bails_on_invalid_config(tmp_path: Path):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(invalid_config)

    with pytest.raises(AssertionError):
        load_and_migrate_config(temp_config_file)


def test_bails_on_config_with_unsupported_version(tmp_path: Path):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(invalid_v5_config)

    with pytest.raises(RuntimeError, match="Invalid schema version"):
        load_and_migrate_config(temp_config_file)


def test_write_config_to_file():
    """Test writing configuration to a file, checking for correct output."""
    with TemporaryDirectory() as tmpdir:
        temp_config_path = Path(tmpdir) / "invokeai.yaml"
        config = InvokeAIAppConfig(host="192.168.1.1", port=8080)
        config.write_file(temp_config_path)
        # Load the file and check contents
        with open(temp_config_path, "r") as file:
            content = file.read()
            # This is a default value, so it should not be in the file
            assert "pil_compress_level" not in content
            assert "host: 192.168.1.1" in content
            assert "port: 8080" in content


def test_update_config_with_dict():
    """Test updating the config with a dictionary."""
    config = InvokeAIAppConfig()
    update_dict = {"host": "10.10.10.10", "port": 6060}
    config.update_config(update_dict)
    assert config.host == "10.10.10.10"
    assert config.port == 6060


def test_update_config_with_object():
    """Test updating the config with another config object."""
    config = InvokeAIAppConfig()
    new_config = InvokeAIAppConfig(host="10.10.10.10", port=6060)
    config.update_config(new_config)
    assert config.host == "10.10.10.10"
    assert config.port == 6060


def test_set_and_resolve_paths():
    """Test setting root and resolving paths based on it."""
    with TemporaryDirectory() as tmpdir:
        config = InvokeAIAppConfig()
        config.set_root(Path(tmpdir))
        assert config.models_path == Path(tmpdir).resolve() / "models"
        assert config.db_path == Path(tmpdir).resolve() / "databases" / "invokeai.db"


def test_singleton_behavior():
    """Test that get_config always returns the same instance."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


@pytest.mark.xfail(
    reason="""
    This test fails when run as part of the full test suite.

    This test needs to deny nodes from being included in the InvocationsUnion by providing
    an app configuration as a test fixture. Pytest executes all test files before running
    tests, so the app configuration is already initialized by the time this test runs, and
    the InvocationUnion is already created and the denied nodes are not omitted from it.

    This test passes when `test_config.py` is tested in isolation.

    Perhaps a solution would be to call `get_app_config().parse_args()` in
    other test files?
    """
)
def test_deny_nodes(patch_rootdir):
    # Allow integer, string and float, but explicitly deny float
    allow_deny_nodes_conf = OmegaConf.create(
        """
        InvokeAI:
          Nodes:
            allow_nodes:
              - integer
              - string
              - float
            deny_nodes:
              - float
        """
    )
    # must parse config before importing Graph, so its nodes union uses the config
    conf = get_config()
    conf.merge_from_file(conf=allow_deny_nodes_conf, argv=[])
    from invokeai.app.services.shared.graph import Graph

    # confirm graph validation fails when using denied node
    Graph(nodes={"1": {"id": "1", "type": "integer"}})
    Graph(nodes={"1": {"id": "1", "type": "string"}})

    with pytest.raises(ValidationError):
        Graph(nodes={"1": {"id": "1", "type": "float"}})

    from invokeai.app.invocations.baseinvocation import BaseInvocation

    # confirm invocations union will not have denied nodes
    all_invocations = BaseInvocation.get_invocations()

    has_integer = len([i for i in all_invocations if i.model_fields.get("type").default == "integer"]) == 1
    has_string = len([i for i in all_invocations if i.model_fields.get("type").default == "string"]) == 1
    has_float = len([i for i in all_invocations if i.model_fields.get("type").default == "float"]) == 1

    assert has_integer
    assert has_string
    assert not has_float
