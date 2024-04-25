from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from packaging.version import Version

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.config.config_default import (
    DefaultInvokeAIAppConfig,
    InvokeAIAppConfig,
    get_config,
    load_and_migrate_config,
)
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

invalid_v4_0_1_config = """
schema_version: 4.0.1

host: "192.168.1.1"
port: "ice cream"
"""

v4_config = """
schema_version: 4.0.0

precision: autocast
host: "192.168.1.1"
port: 8080
"""

invalid_v5_config = """
schema_version: 5.0.0

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


def test_path_resolution_root_not_set(patch_rootdir: None):
    """Test path resolutions when the root is not explicitly set."""
    config = InvokeAIAppConfig()
    expected_root = InvokeAIAppConfig.find_root()
    assert config.root_path == expected_root


def test_read_config_from_file(tmp_path: Path, patch_rootdir: None):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v4_config)

    config = load_and_migrate_config(temp_config_file)
    assert config.host == "192.168.1.1"
    assert config.port == 8080


def test_migrate_v3_config_from_file(tmp_path: Path, patch_rootdir: None):
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


@pytest.mark.parametrize(
    "legacy_conf_dir,expected_value,expected_is_set",
    [
        # not set, expected value is the default value
        ("configs/stable-diffusion", Path("configs"), False),
        # not set, expected value is the default value
        ("configs\\stable-diffusion", Path("configs"), False),
        # set, best-effort resolution of the path
        ("partial_custom_path/stable-diffusion", Path("partial_custom_path"), True),
        # set, exact path
        ("full/custom/path", Path("full/custom/path"), True),
    ],
)
def test_migrate_v3_legacy_conf_dir_defaults(
    tmp_path: Path, patch_rootdir: None, legacy_conf_dir: str, expected_value: Path, expected_is_set: bool
):
    """Test reading configuration from a file."""
    config_content = f"InvokeAI:\n    Paths:\n        legacy_conf_dir: {legacy_conf_dir}"
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(config_content)

    config = load_and_migrate_config(temp_config_file)
    assert config.legacy_conf_dir == expected_value
    assert ("legacy_conf_dir" in config.model_fields_set) is expected_is_set


def test_migrate_v3_backup(tmp_path: Path, patch_rootdir: None):
    """Test the backup of the config file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v3_config)

    load_and_migrate_config(temp_config_file)
    assert temp_config_file.with_suffix(".yaml.bak").exists()
    assert temp_config_file.with_suffix(".yaml.bak").read_text() == v3_config


def test_migrate_v4(tmp_path: Path, patch_rootdir: None):
    """Test migration from 4.0.0 to 4.0.1"""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v4_config)

    conf = load_and_migrate_config(temp_config_file)
    assert Version(conf.schema_version) >= Version("4.0.1")
    assert conf.precision == "auto"  # we expect 'autocast' to be replaced with 'auto' during 4.0.1 migration


def test_failed_migrate_backup(tmp_path: Path, patch_rootdir: None):
    """Test the failed migration of the config file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v3_config_with_bad_values)

    with pytest.raises(RuntimeError):
        load_and_migrate_config(temp_config_file)
    assert temp_config_file.with_suffix(".yaml.bak").exists()
    assert temp_config_file.with_suffix(".yaml.bak").read_text() == v3_config_with_bad_values
    assert temp_config_file.exists()
    assert temp_config_file.read_text() == v3_config_with_bad_values


def test_bails_on_invalid_config(tmp_path: Path, patch_rootdir: None):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(invalid_config)

    with pytest.raises(AssertionError):
        load_and_migrate_config(temp_config_file)


@pytest.mark.parametrize("config_content", [invalid_v5_config, invalid_v4_0_1_config])
def test_bails_on_config_with_unsupported_version(tmp_path: Path, patch_rootdir: None, config_content: str):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(config_content)

    #    with pytest.raises(RuntimeError, match="Invalid schema version"):
    with pytest.raises(RuntimeError):
        load_and_migrate_config(temp_config_file)


def test_write_config_to_file(patch_rootdir: None):
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


def test_update_config_with_dict(patch_rootdir: None):
    """Test updating the config with a dictionary."""
    config = InvokeAIAppConfig()
    update_dict = {"host": "10.10.10.10", "port": 6060}
    config.update_config(update_dict)
    assert config.host == "10.10.10.10"
    assert config.port == 6060


def test_update_config_with_object(patch_rootdir: None):
    """Test updating the config with another config object."""
    config = InvokeAIAppConfig()
    new_config = InvokeAIAppConfig(host="10.10.10.10", port=6060)
    config.update_config(new_config)
    assert config.host == "10.10.10.10"
    assert config.port == 6060


def test_set_and_resolve_paths(patch_rootdir: None):
    """Test setting root and resolving paths based on it."""
    with TemporaryDirectory() as tmpdir:
        config = InvokeAIAppConfig()
        config._root = Path(tmpdir)
        assert config.models_path == Path(tmpdir).resolve() / "models"
        assert config.db_path == Path(tmpdir).resolve() / "databases" / "invokeai.db"


def test_singleton_behavior(patch_rootdir: None):
    """Test that get_config always returns the same instance."""
    get_config.cache_clear()
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2
    get_config.cache_clear()


def test_default_config(patch_rootdir: None):
    """Test that the default config is as expected."""
    config = DefaultInvokeAIAppConfig()
    assert config.host == "127.0.0.1"


def test_env_vars(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that environment variables are merged into the config"""
    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))
    monkeypatch.setenv("INVOKEAI_HOST", "1.2.3.4")
    monkeypatch.setenv("INVOKEAI_PORT", "1234")
    config = InvokeAIAppConfig()
    assert config.host == "1.2.3.4"
    assert config.port == 1234
    assert config.root_path == tmp_path


def test_get_config_writing(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that get_config writes the appropriate files to disk"""
    # Trick the config into thinking it has already parsed args - this triggers the writing of the config file
    InvokeAIArgs.did_parse = True

    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))
    monkeypatch.setenv("INVOKEAI_HOST", "1.2.3.4")
    get_config.cache_clear()
    config = get_config()
    get_config.cache_clear()
    config_file_path = tmp_path / "invokeai.yaml"
    example_file_path = config_file_path.with_suffix(".example.yaml")
    assert config.config_file_path == config_file_path
    assert config_file_path.exists()
    assert example_file_path.exists()

    # The example file should have the default values
    example_file_content = example_file_path.read_text()
    assert "host: 127.0.0.1" in example_file_content
    assert "port: 9090" in example_file_content

    # It should also have the `remote_api_tokens` key
    assert "remote_api_tokens" in example_file_content

    # Neither env vars nor default values should be written to the config file
    config_file_content = config_file_path.read_text()
    assert "host" not in config_file_content

    # Undo our change to the singleton class
    InvokeAIArgs.did_parse = False


def test_deny_nodes():
    config = get_config()
    config.allow_nodes = ["integer", "string", "float"]
    config.deny_nodes = ["float"]

    # confirm invocations union will not have denied nodes
    all_invocations = BaseInvocation.get_invocations()

    has_integer = len([i for i in all_invocations if i.model_fields.get("type").default == "integer"]) == 1
    has_string = len([i for i in all_invocations if i.model_fields.get("type").default == "string"]) == 1
    has_float = len([i for i in all_invocations if i.model_fields.get("type").default == "float"]) == 1

    assert has_integer
    assert has_string
    assert not has_float

    # may not be necessary
    get_config.cache_clear()
