"""Record the variant on Anima main models installed before Anima had more than one.

`Main_Checkpoint_Anima_Config.variant` became a required field when Anima-3.8B joined: its bundled
semantic connector needs a Qwen3.5 encoder beside the Qwen3 one, and the frontend reads the
variant to ask for it. Records written before that carry no `variant`, and a stored config that
fails to validate is *skipped* on read (`ModelRecordServiceSQL._select_models`) -- every installed
Anima model would silently vanish from the model list.

Nearly every such record is the Qwen3-only variant, because Anima-3.8B could not load before. It
could be *installed*, though: its checkpoint identified as an ordinary Anima. So the checkpoint's
header is read where the file is still there, and a bundled connector decides; a record whose file
cannot be read gets the Qwen3-only variant, which is what it was being loaded as.

Mirrors `2026_09_16_add_qwen3_vl_encoder_variant`.
"""

import json
import sqlite3
from logging import Logger
from pathlib import Path
from typing import Any

from safetensors import safe_open

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.taxonomy import AnimaVariantType, BaseModelType, ModelFormat, ModelType

# Kept literal rather than imported from the config module: a migration must keep meaning what it
# meant when it was written, whatever the config later renames.
_CONNECTOR_SEGMENT = "anima_v2_connector."


def _bundles_semantic_connector(path: Path) -> bool:
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        return any(key.startswith(_CONNECTOR_SEGMENT) or f".{_CONNECTOR_SEGMENT}" in key for key in checkpoint.keys())


class AddAnimaVariantCallback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._models_path = app_config.models_path
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        migrated: dict[str, int] = {}
        for model_id, config_json in rows:
            try:
                config: dict[str, Any] = json.loads(config_json)
            except json.JSONDecodeError as e:
                self._logger.error("Invalid config JSON for model %s: %s", model_id, e)
                raise

            # Only the single-file main model gained the field. Anima's LLLite control adapters share
            # the base and have no variant at all.
            if (
                config.get("type") != ModelType.Main.value
                or config.get("base") != BaseModelType.Anima.value
                or config.get("format") != ModelFormat.Checkpoint.value
                or "variant" in config
            ):
                continue

            variant = self._variant_of(config)
            config["variant"] = variant.value
            cursor.execute("UPDATE models SET config = ? WHERE id = ?;", (json.dumps(config), model_id))
            migrated[variant.value] = migrated.get(variant.value, 0) + 1

        if migrated:
            self._logger.info(f"Recorded the variant on Anima model config(s): {migrated}")

    def _variant_of(self, config: dict[str, Any]) -> AnimaVariantType:
        raw_path = config.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return AnimaVariantType.Qwen3
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._models_path / path
        try:
            if _bundles_semantic_connector(path):
                return AnimaVariantType.Qwen35
        except Exception as e:
            self._logger.warning(
                "Could not read the header of Anima model %s (%s); recording it as %s.",
                config.get("name", raw_path),
                e,
                AnimaVariantType.Qwen3.value,
            )
        return AnimaVariantType.Qwen3


def build_migration(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    return Migration(
        id="2026_10_01_add_anima_variant",
        depends_on="2026_09_26_add_workflow_revision",
        callback=AddAnimaVariantCallback(app_config=app_config, logger=logger),
    )
