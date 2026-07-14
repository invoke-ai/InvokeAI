import json
from pathlib import Path

import pytest

from invokeai.backend.model_manager.configs.factory import _MAX_FILES_IN_MODEL_DIR, ModelConfigFactory


def _fill_directory(path: Path) -> None:
    for index in range(_MAX_FILES_IN_MODEL_DIR + 1):
        (path / f"asset-{index}.txt").touch()


def test_large_directory_with_generic_config_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"application": "not-a-model"}))
    _fill_directory(tmp_path)

    with pytest.raises(ValueError, match="general-purpose directory"):
        ModelConfigFactory._validate_path_looks_like_model(tmp_path)


def test_large_directory_with_non_object_config_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("[]")
    _fill_directory(tmp_path)

    with pytest.raises(ValueError, match="general-purpose directory"):
        ModelConfigFactory._validate_path_looks_like_model(tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "application"},
        {"architectures": ["ApplicationService"]},
        {"_class_name": "ApplicationPipeline"},
    ],
)
def test_large_directory_with_unrecognized_model_markers_is_rejected(tmp_path: Path, config: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))
    _fill_directory(tmp_path)

    with pytest.raises(ValueError, match="general-purpose directory"):
        ModelConfigFactory._validate_path_looks_like_model(tmp_path)


def test_large_directory_with_transformers_config_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3VLModel"]}))
    _fill_directory(tmp_path)

    ModelConfigFactory._validate_path_looks_like_model(tmp_path)


def test_large_directory_with_model_index_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Krea2Pipeline"}))
    _fill_directory(tmp_path)

    ModelConfigFactory._validate_path_looks_like_model(tmp_path)


def test_directory_with_utf8_non_ascii_config_is_accepted(tmp_path: Path) -> None:
    # config.json files are UTF-8. Reading them with the platform default encoding (cp1252 on Windows)
    # raises UnicodeDecodeError on non-ASCII bytes, which gets swallowed as "unrecognized" and wrongly
    # rejects a valid model directory. The config must be read explicitly as UTF-8.
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3VLModel"], "description": "café — ünïcödé 模型"}, ensure_ascii=False),
        encoding="utf-8",
    )

    ModelConfigFactory._validate_path_looks_like_model(tmp_path)
