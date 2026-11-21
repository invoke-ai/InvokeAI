"""Which registered models have lost their files, by the one rule the Model Manager and the calibration share."""

from collections.abc import Iterable
from pathlib import Path

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat


def is_model_missing(config: AnyModelConfig, models_path: Path) -> bool:
    """Whether the files of a registered model are gone. External providers have none, so they are never missing."""
    if config.base == BaseModelType.External or config.format == ModelFormat.ExternalApi:
        return False
    return not (models_path / config.path).resolve().exists()


def find_missing_models(configs: Iterable[AnyModelConfig], models_path: Path) -> list[AnyModelConfig]:
    """The registered models among `configs` whose files are gone, in order."""
    return [config for config in configs if is_model_missing(config, models_path)]
