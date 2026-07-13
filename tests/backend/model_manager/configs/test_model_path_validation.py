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


def test_large_directory_with_transformers_config_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3VLModel"]}))
    _fill_directory(tmp_path)

    ModelConfigFactory._validate_path_looks_like_model(tmp_path)


def test_large_directory_with_model_index_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Krea2Pipeline"}))
    _fill_directory(tmp_path)

    ModelConfigFactory._validate_path_looks_like_model(tmp_path)
