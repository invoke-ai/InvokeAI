import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.services.config.config_default import (
    DefaultInvokeAIAppConfig,
    InvokeAIAppConfig,
    ensure_fonts_dir,
    get_config,
    load_and_migrate_config,
    load_config_from_root,
)
from invokeai.app.services.shared.graph import Graph
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

v4_config = """
schema_version: 4.0.0

host: "192.168.1.1"
port: 8080
"""

v4_0_2_config = """
schema_version: "4.0.2"

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


def test_wan_memory_optimization_defaults_to_false_and_loads_from_yaml(tmp_path: Path, patch_rootdir: None) -> None:
    assert InvokeAIAppConfig().wan_memory_optimization is False

    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text('schema_version: "4.0.3"\nwan_memory_optimization: true\n')

    assert load_and_migrate_config(temp_config_file).wan_memory_optimization is True


@pytest.mark.parametrize(("yaml_line", "expected"), [("", True), ("multiuser: false\n", False)])
def test_sign_in_is_required_unless_single_user_mode_is_set_explicitly(
    tmp_path: Path, patch_rootdir: None, yaml_line: str, expected: bool
) -> None:
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(f'schema_version: "4.0.3"\n{yaml_line}')

    assert load_and_migrate_config(temp_config_file).multiuser is expected


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [("on", "on"), ("off", "off"), ('"on"', "on"), ("auto", "auto")],
    ids=["unquoted-on", "unquoted-off", "quoted", "auto"],
)
def test_rocm_aotriton_experimental_loads_unquoted_on_and_off(
    tmp_path: Path, patch_rootdir: None, yaml_value: str, expected: str
) -> None:
    """YAML 1.1 reads an unquoted `on`/`off` as a boolean; the startup log tells users to write exactly that."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(f'schema_version: "4.0.3"\nrocm_aotriton_experimental: {yaml_value}\n')

    assert load_and_migrate_config(temp_config_file).rocm_aotriton_experimental == expected


def test_db_synchronous_defaults_to_full_and_loads_from_yaml(tmp_path: Path, patch_rootdir: None) -> None:
    # The default must stay `full`: anything else would quietly reduce durability for every existing
    # install on upgrade.
    assert InvokeAIAppConfig().db_synchronous == "full"

    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text('schema_version: "4.0.3"\ndb_synchronous: normal\n')

    assert load_and_migrate_config(temp_config_file).db_synchronous == "normal"


def test_attention_backend_defaults_to_auto_and_loads_from_yaml(tmp_path: Path, patch_rootdir: None) -> None:
    # The default must stay `auto`: SageAttention changes images at the same seed, so it is opt-in only.
    assert InvokeAIAppConfig().attention_backend == "auto"

    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text('schema_version: "4.0.3"\nattention_backend: sage\n')

    assert load_and_migrate_config(temp_config_file).attention_backend == "sage"


@pytest.mark.parametrize("kernel", ["cudnn", "flash", "efficient", "math"])
def test_attention_backend_pins_a_pytorch_kernel_from_yaml(tmp_path: Path, patch_rootdir: None, kernel: str) -> None:
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(f'schema_version: "4.0.3"\nattention_backend: {kernel}\n')

    assert load_and_migrate_config(temp_config_file).attention_backend == kernel


def test_attention_backend_rejects_a_kernel_it_does_not_know() -> None:
    # `default` is an INVOKE_KREA2_SDPA_BACKEND value, not an attention_backend one.
    for value in ("xformers", "default"):
        with pytest.raises(ValidationError):
            InvokeAIAppConfig(attention_backend=value)


def test_load_config_from_root_reads_that_roots_file_and_writes_nothing(tmp_path: Path, patch_rootdir: None) -> None:
    root = tmp_path / "install"
    root.mkdir()
    (root / "invokeai.yaml").write_text(
        'schema_version: "4.0.3"\ndb_dir: elsewhere\noutputs_dir: pictures\ndb_synchronous: normal\n'
    )

    config = load_config_from_root(root)

    assert config.db_path == (root / "elsewhere" / "invokeai.db").resolve()
    assert config.outputs_path == (root / "pictures").resolve()
    assert config.db_synchronous == "normal"
    assert [path.name for path in root.iterdir()] == ["invokeai.yaml"]


def test_load_config_from_root_lets_the_environment_win_over_the_file(
    tmp_path: Path, patch_rootdir: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "invokeai.yaml").write_text('schema_version: "4.0.3"\ndb_synchronous: normal\n')
    monkeypatch.setenv("INVOKEAI_DB_SYNCHRONOUS", "full")

    assert load_config_from_root(tmp_path).db_synchronous == "full"


def test_read_config_from_file(tmp_path: Path, patch_rootdir: None):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v4_config)

    config = load_and_migrate_config(temp_config_file)
    assert config.host == "192.168.1.1"
    assert config.port == 8080


def test_pid_memory_optimization_defaults_to_false_and_loads_from_yaml(tmp_path: Path, patch_rootdir: None) -> None:
    assert InvokeAIAppConfig().pid_memory_optimization is False

    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text('schema_version: "4.0.3"\npid_memory_optimization: true\n')

    assert load_and_migrate_config(temp_config_file).pid_memory_optimization is True


def test_tiny_autoencoder_previews_defaults_to_false_and_loads_from_yaml(tmp_path: Path, patch_rootdir: None) -> None:
    assert InvokeAIAppConfig().tiny_autoencoder_previews is False

    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text('schema_version: "4.0.3"\ntiny_autoencoder_previews: true\n')

    assert load_and_migrate_config(temp_config_file).tiny_autoencoder_previews is True


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


def test_migrate_v4_0_2_to_4_0_3_config(tmp_path: Path, patch_rootdir: None):
    """Test that a v4.0.2 config is migrated to v4.0.3 (image subfolder support)."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(v4_0_2_config)

    config = load_and_migrate_config(temp_config_file)
    assert config.schema_version == "4.0.3"
    assert config.host == "192.168.1.1"
    assert config.port == 8080
    # image_subfolder_strategy should have its default value
    assert config.image_subfolder_strategy == "flat"
    # Older releases default to single-user mode, so the rewrite states sign-in explicitly.
    assert yaml.safe_load(temp_config_file.read_text())["multiuser"] is True


def test_bails_on_invalid_config(tmp_path: Path, patch_rootdir: None):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(invalid_config)

    with pytest.raises(AssertionError):
        load_and_migrate_config(temp_config_file)


def test_bails_on_config_with_unsupported_version(tmp_path: Path, patch_rootdir: None):
    """Test reading configuration from a file."""
    temp_config_file = tmp_path / "temp_invokeai.yaml"
    temp_config_file.write_text(invalid_v5_config)

    with pytest.raises(RuntimeError, match="Invalid schema version"):
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


def test_update_config_with_dict(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch):
    """Test updating the config with a dictionary."""
    monkeypatch.setenv("INVOKEAI_MULTIUSER", "false")
    config = InvokeAIAppConfig(multiuser=True)
    update_dict = {"host": "10.10.10.10", "port": 6060}
    config.update_config(update_dict)
    assert config.host == "10.10.10.10"
    assert config.port == 6060
    assert config.multiuser is True


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
    monkeypatch.setenv("INVOKEAI_MULTIUSER", "false")
    get_config.cache_clear()
    config = get_config()
    get_config.cache_clear()
    config_file_path = tmp_path / "invokeai.yaml"
    example_file_path = config_file_path.with_suffix(".example.yaml")
    assert config.config_file_path == config_file_path
    assert config_file_path.exists()
    assert example_file_path.exists()
    assert (tmp_path / "fonts").exists()
    assert (tmp_path / "fonts" / "README.txt").exists()

    # The example file should have the default values
    example_file_content = example_file_path.read_text()
    assert "host: 127.0.0.1" in example_file_content
    assert "port: 9090" in example_file_content

    # It should also have the `remote_api_tokens` key
    assert "remote_api_tokens" in example_file_content

    # Neither env vars nor default values should be written to the config file, except the sign-in choice, which
    # older releases default differently
    config_file_content = config_file_path.read_text()
    assert "host" not in config_file_content
    assert yaml.safe_load(config_file_content)["multiuser"] is True

    # Undo our change to the singleton class
    InvokeAIArgs.did_parse = False


def _start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InvokeAIAppConfig:
    monkeypatch.setattr(InvokeAIArgs, "did_parse", True)
    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))
    get_config.cache_clear()
    try:
        return get_config()
    finally:
        get_config.cache_clear()


@pytest.mark.parametrize("ending", ["\n", ""])
def test_startup_appends_sign_in_to_a_config_file_without_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ending: str
) -> None:
    monkeypatch.setenv("INVOKEAI_MULTIUSER", "false")
    content = '# my notes\nschema_version: "4.0.3"\nport: 8080  # kept\n# last' + ending
    (tmp_path / "invokeai.yaml").write_text(content)

    config = _start(tmp_path, monkeypatch)

    assert config.multiuser is False
    assert config.port == 8080
    assert (tmp_path / "invokeai.yaml").read_text() == content + ("" if ending else "\n") + "multiuser: true\n"


@pytest.mark.parametrize(
    ("content", "warns"),
    [
        ('# my notes\nschema_version: "4.0.3"\nport: 8080\nmultiuser: false\n', False),
        ('{schema_version: "4.0.3", port: 8080}\n', True),
    ],
    ids=["states-sign-in", "flow-style"],
)
def test_startup_leaves_a_config_file_untouched_when_it_states_sign_in_or_cannot_take_a_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str, warns: bool
) -> None:
    (tmp_path / "invokeai.yaml").write_text(content)

    with caplog.at_level("WARNING"):
        config = _start(tmp_path, monkeypatch)

    assert config.port == 8080
    assert (tmp_path / "invokeai.yaml").read_text() == content
    assert ("Unable to add the multiuser setting" in caplog.text) is warns


def test_startup_still_loads_a_read_only_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_file = tmp_path / "invokeai.yaml"
    content = 'schema_version: "4.0.3"\nport: 8080\n'
    config_file.write_text(content)
    config_file.chmod(0o444)
    try:
        if os.access(config_file, os.W_OK):
            pytest.skip("this user can write read-only files")
        with caplog.at_level("WARNING"):
            config = _start(tmp_path, monkeypatch)
    finally:
        config_file.chmod(0o644)

    assert config.port == 8080
    assert config_file.read_text() == content
    assert "Unable to add the multiuser setting" in caplog.text


def test_ensure_fonts_dir_logs_warning_on_oserror(
    patch_rootdir: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    original_mkdir = Path.mkdir
    fonts_path = tmp_path / "fonts"

    def mock_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == fonts_path:
            raise OSError("read-only")
        original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    with caplog.at_level("WARNING"):
        ensure_fonts_dir(fonts_path)

    assert "Unable to initialize fonts directory" in caplog.text


def test_get_config_reads_external_api_keys_file(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that API keys are loaded from the dedicated api_keys.yaml file."""
    InvokeAIArgs.did_parse = True
    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))
    (tmp_path / "invokeai.yaml").write_text("schema_version: 4.0.2\n")
    (tmp_path / "api_keys.yaml").write_text("external_openai_api_key: openai-key\n")

    get_config.cache_clear()
    config = get_config()
    get_config.cache_clear()

    assert config.external_openai_api_key == "openai-key"

    InvokeAIArgs.did_parse = False


def test_get_config_env_vars_override_external_api_keys_file(
    patch_rootdir: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Test that environment variables override values from api_keys.yaml."""
    InvokeAIArgs.did_parse = True
    monkeypatch.setenv("INVOKEAI_ROOT", str(tmp_path))
    monkeypatch.setenv("INVOKEAI_EXTERNAL_OPENAI_API_KEY", "env-openai-key")
    (tmp_path / "invokeai.yaml").write_text("schema_version: 4.0.2\n")
    (tmp_path / "api_keys.yaml").write_text("external_openai_api_key: file-openai-key\n")

    get_config.cache_clear()
    config = get_config()
    get_config.cache_clear()

    assert config.external_openai_api_key == "env-openai-key"

    InvokeAIArgs.did_parse = False


def test_deny_nodes(patch_rootdir):
    # Allow integer, string and float, but explicitly deny float
    conf = get_config()
    conf.allow_nodes = ["integer", "string", "float"]
    conf.deny_nodes = ["float"]

    # We've changed the config, we need to invalidate the typeadapter cache so that the new config is used for
    # subsequent graph validations
    InvocationRegistry.invalidate_invocation_typeadapter()

    # confirm graph validation fails when using denied node
    Graph.model_validate({"nodes": {"1": {"id": "1", "type": "integer"}}})
    Graph.model_validate({"nodes": {"1": {"id": "1", "type": "string"}}})

    with pytest.raises(ValidationError):
        Graph.model_validate({"nodes": {"1": {"id": "1", "type": "float"}}})

    # confirm invocations union will not have denied nodes
    all_invocations = InvocationRegistry.get_invocation_classes()

    has_integer = len([i for i in all_invocations if i.get_type() == "integer"]) == 1
    has_string = len([i for i in all_invocations if i.get_type() == "string"]) == 1
    has_float = len([i for i in all_invocations if i.get_type() == "float"]) == 1

    assert has_integer
    assert has_string
    assert not has_float

    # Reset the config so that it doesn't affect other tests
    get_config.cache_clear()
    InvocationRegistry.invalidate_invocation_typeadapter()


def test_storage_backend_defaults_to_disk(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch):
    for var in ("INVOKEAI_STORAGE_BACKEND", "INVOKEAI_S3_BUCKET", "INVOKEAI_S3_ENDPOINT_URL", "INVOKEAI_S3_REGION"):
        monkeypatch.delenv(var, raising=False)
    config = InvokeAIAppConfig()
    assert config.storage_backend == "disk"
    assert config.s3_bucket is None
    assert config.s3_endpoint_url is None
    assert config.s3_region is None


def test_storage_backend_accepts_s3(patch_rootdir: None):
    config = InvokeAIAppConfig(
        storage_backend="s3",
        s3_bucket="my-bucket",
        s3_endpoint_url="https://s3.us-west-004.backblazeb2.com",
        s3_region="us-west-004",
    )
    assert config.storage_backend == "s3"
    assert config.s3_bucket == "my-bucket"
    assert config.s3_endpoint_url == "https://s3.us-west-004.backblazeb2.com"
    assert config.s3_region == "us-west-004"


def test_storage_backend_rejects_unknown_value(patch_rootdir: None):
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(storage_backend="azure")


def test_storage_backend_s3_requires_bucket(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch):
    """`storage_backend="s3"` without `s3_bucket` (and no env var) must fail
    at config-load time rather than deep in dependencies.initialize().
    Per Copilot review on PR #9182."""
    monkeypatch.delenv("INVOKEAI_S3_BUCKET", raising=False)
    with pytest.raises(ValidationError, match="s3_bucket"):
        InvokeAIAppConfig(storage_backend="s3")


def test_storage_backend_s3_accepts_bucket_via_env(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch):
    """`INVOKEAI_S3_BUCKET` should also satisfy the bucket requirement."""
    monkeypatch.setenv("INVOKEAI_S3_BUCKET", "env-bucket")
    # Should not raise.
    config = InvokeAIAppConfig(storage_backend="s3")
    assert config.storage_backend == "s3"


def test_storage_backend_s3_rejects_whitespace_only_bucket(patch_rootdir: None, monkeypatch: pytest.MonkeyPatch):
    """A whitespace-only bucket must be treated as unset, not a valid bucket."""
    monkeypatch.delenv("INVOKEAI_S3_BUCKET", raising=False)
    with pytest.raises(ValidationError, match="s3_bucket"):
        InvokeAIAppConfig(storage_backend="s3", s3_bucket="   ")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("/invoke", "/invoke"),
        ("invoke", "/invoke"),
        ("invoke/", "/invoke"),
        ("/invoke/", "/invoke"),
        ("//invoke//", "/invoke"),
        (" /invoke ", "/invoke"),
        ("/invoke/sub", "/invoke/sub"),
        ("", None),
        ("/", None),
        ("   ", None),
        (None, None),
    ],
)
def test_base_url_validator_normalizes(raw: str | None, expected: str | None, patch_rootdir: None):
    """`base_url` is normalized to a single leading slash with no trailing slash; empty/`/` disable it."""
    config = InvokeAIAppConfig(base_url=raw)
    assert config.base_url == expected


@pytest.mark.parametrize(
    "reserved",
    ["/api", "api", "/ws", "/static", "/docs", "/redoc", "/openapi.json", "/locales", "/assets", "/api/foo"],
)
def test_base_url_validator_rejects_reserved_prefix(reserved: str, patch_rootdir: None):
    """A `base_url` whose first segment collides with a real route prefix must fail fast, not brick the server."""
    with pytest.raises(ValidationError, match="reserved path segment"):
        InvokeAIAppConfig(base_url=reserved)
