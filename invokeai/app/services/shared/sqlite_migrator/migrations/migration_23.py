import json
import sqlite3
from logging import Logger
from typing import Any

from pydantic import ValidationError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, AnyModelConfigValidator
from invokeai.backend.model_manager.taxonomy import BaseModelType, FluxVariantType, ModelType, SchedulerPredictionType


class Migration23Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger
        self._models_dir = app_config.models_path.resolve()

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        # Grab all model records
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        for model_id, config_json in rows:
            try:
                # Migrate the config JSON to the latest schema
                migrated_config = self._parse_and_migrate_config(config_json)
                # Write the migrated config back to the database
                cursor.execute(
                    "UPDATE models SET config = ? WHERE id = ?;",
                    (migrated_config.model_dump_json(), model_id),
                )
            except ValidationError as e:
                self._logger.error("Invalid config schema for model %s: %s", model_id, e)
                raise
            except json.JSONDecodeError as e:
                self._logger.error("Invalid config JSON for model %s: %s", model_id, e)
                raise

    def _parse_and_migrate_config(self, config_json: Any) -> AnyModelConfig:
        config_dict: dict[str, Any] = json.loads(config_json)

        # In v6.8.0 we made some improvements to the model taxonomy and the model config schemas. There are a changes
        # we need to make to old configs to bring them up to date.

        base = config_dict.get("base")
        type = config_dict.get("type")
        if base == BaseModelType.Flux.value and type == ModelType.Main.value:
            # Prior to v6.8.0, we used an awkward combination of `config_path` and `variant` to distinguish between FLUX
            # variants.
            #
            # `config_path` was set to one of:
            #  - flux-dev
            #  - flux-dev-fill
            #  - flux-schnell
            #
            # `variant` was set to ModelVariantType.Inpaint for FLUX Fill models and ModelVariantType.Normal for all other FLUX
            # models.
            #
            # We now use the `variant` field to directly represent the FLUX variant type, and `config_path` is no longer used.

            # Extract and remove `config_path` if present.
            config_path = config_dict.pop("config_path", None)

            match config_path:
                case "flux-dev":
                    config_dict["variant"] = FluxVariantType.Dev.value
                case "flux-dev-fill":
                    config_dict["variant"] = FluxVariantType.DevFill.value
                case "flux-schnell":
                    config_dict["variant"] = FluxVariantType.Schnell.value
                case _:
                    # Unknown config_path - default to Dev variant
                    config_dict["variant"] = FluxVariantType.Dev.value

        if (
            base
            in {
                BaseModelType.StableDiffusion1.value,
                BaseModelType.StableDiffusion2.value,
                BaseModelType.StableDiffusionXL.value,
                BaseModelType.StableDiffusionXLRefiner.value,
            }
            and type == "main"
        ):
            # Prior to v6.8.0, the prediction_type field was optional and would default to Epsilon if not present.
            # We now make it explicit and always present. Use the existing value if present, otherwise default to
            # Epsilon, matching the probe logic.
            #
            # It's only on SD1.x, SD2.x, and SDXL main models.
            config_dict["prediction_type"] = config_dict.get("prediction_type", SchedulerPredictionType.Epsilon.value)

        if type == ModelType.CLIPVision.value:
            # Prior to v6.8.0, some CLIP Vision models were associated with a specific base model architecture:
            # - CLIP-ViT-bigG-14-laion2B-39B-b160k is the image encoder for SDXL IP Adapter and was associated with SDXL
            # - CLIP-ViT-H-14-laion2B-s32B-b79K is the image encoder for SD1.5 IP Adapter and was associated with SD1.5
            #
            # While this made some sense at the time, it is more correct and flexible to treat CLIP Vision models
            # as independent of any specific base model architecture.
            config_dict["base"] = BaseModelType.Any.value

        migrated_config = AnyModelConfigValidator.validate_python(config_dict)
        return migrated_config


def build_migration_23(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    """Builds the migration object for migrating from version 22 to version 23.

    This migration updates model configurations to the latest config schemas for v6.8.0.
    """

    return Migration(
        from_version=22,
        to_version=23,
        callback=Migration23Callback(app_config=app_config, logger=logger),
    )
