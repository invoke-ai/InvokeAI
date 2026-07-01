import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any

import pytest

from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
from invokeai.backend.model_manager.configs.factory import (
    ModelConfigFactory,
)
from invokeai.backend.model_manager.configs.main import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.model_identification.stripped_model_on_disk import StrippedModelOnDisk

logger = InvokeAILogger.get_logger(__file__)


@pytest.mark.parametrize(
    "model_name,preprocessor",
    [
        ("some_canny_model", "canny_image_processor"),
        ("some_depth_model", "depth_anything_image_processor"),
        ("some_pose_model", "dw_openpose_image_processor"),
        ("i like turtles", None),
    ],
)
def test_controlnet_t2i_default_settings(model_name: str, preprocessor: str | None):
    assert ControlAdapterDefaultSettings.from_model_name(model_name).preprocessor == preprocessor


@pytest.mark.parametrize(
    "base,attrs",
    [
        (BaseModelType.StableDiffusion1, {"width": 512, "height": 512}),
        (BaseModelType.StableDiffusion2, {"width": 768, "height": 768}),
        (BaseModelType.StableDiffusionXL, {"width": 1024, "height": 1024}),
        (BaseModelType.StableDiffusionXLRefiner, None),
        (BaseModelType.Any, None),
    ],
)
def test_default_settings_main(base: BaseModelType, attrs: dict[str, Any] | None):
    settings = MainModelDefaultSettings.from_base(base)
    if attrs is None:
        assert settings is None
    else:
        for key, value in attrs.items():
            assert getattr(settings, key) == value


@dataclass
class ModelAttributeMismatch:
    key: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        return f"{self.key} expected {self.expected}, got {self.actual}"


def _get_model_paths(datadir: Path) -> list[Path]:
    """Helper to collect model paths for parameterization."""
    return [p for p in (datadir / "stripped_models").iterdir() if p.is_dir()]


@pytest.mark.parametrize("model_path", _get_model_paths(Path(__file__).parent))
def test_model_identification(model_path: Path):
    """Verifies results from ModelConfigBase.classify are consistent with those from ModelProbe.probe.
    The test paths are gathered from the 'test_model_probe' directory.
    """
    id = model_path.name
    test_metadata_path = model_path / "__test_metadata__.json"
    test_metadata = json.loads(test_metadata_path.read_text())

    if file_name := test_metadata.get("file_name", ""):
        model_path = model_path / file_name

    mod = StrippedModelOnDisk(model_path)

    override_fields = test_metadata.get("override_fields", None)

    try:
        result = ModelConfigFactory.from_model_on_disk(mod, override_fields, allow_unknown=False)
    except Exception as e:
        print(mod.path)
        pytest.fail(f"{id}: Exception during model probing: {e}")

    if result.config is None:
        pytest.fail(f"{id}: no match, detailed results:\n{pformat(result.details)}")

    config = result.config

    mismatched_attrs: list[ModelAttributeMismatch] = []

    for key, expected_value in test_metadata["expected_config_attrs"].items():
        actual_value = getattr(config, key)
        if isinstance(actual_value, Enum):
            actual_value = actual_value.value
        if actual_value != expected_value:
            mismatched_attrs.append(ModelAttributeMismatch(key, expected_value, actual_value))

    if mismatched_attrs:
        msg = "; ".join(str(m) for m in mismatched_attrs)
        pytest.fail(f"{id}: {msg}")
