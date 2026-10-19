"""Record the variant on Qwen3-VL encoders installed before there was more than one.

`Qwen3VLEncoder_*_Config.variant` became a required field when Ideogram 4's Qwen3-VL 8B encoder
joined Krea-2's 4B: the two install under the same model type and the loader picks the HuggingFace
architecture from this field. Records written before that carry no `variant` at all, and a stored
config that fails to validate is *skipped* on read (`ModelRecordServiceSQL._select_models`) — the
encoder would silently vanish from the model list rather than fail loudly.

Every such record is a 4B, because that is the only shape the config accepted until now -- with one
exception that must be left alone: MiniMax H3's truncated Qwen3-VL-**32B** shares this model type
under its own base and config class, which has no `variant` field at all. Stamping it would write an
untrue size into its stored JSON, harmless only until that class grows a variant of its own.

Mirrors `migration_25`, which did the same for the text-only Qwen3 encoders.
"""

import json
import sqlite3
from logging import Logger
from typing import Any

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, Qwen3VLVariantType


class AddQwen3VLEncoderVariantCallback:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        migrated = 0
        for model_id, config_json in rows:
            try:
                config: dict[str, Any] = json.loads(config_json)
            except json.JSONDecodeError as e:
                self._logger.error("Invalid config JSON for model %s: %s", model_id, e)
                raise

            # `base` is what separates the two classes sharing this type: the encoders that gained
            # the field are base-agnostic components, MiniMax H3's is pinned to its own base.
            if (
                config.get("type") != ModelType.Qwen3VLEncoder.value
                or config.get("base") != BaseModelType.Any.value
                or "variant" in config
            ):
                continue

            config["variant"] = Qwen3VLVariantType.Qwen3VL_4B.value
            cursor.execute("UPDATE models SET config = ? WHERE id = ?;", (json.dumps(config), model_id))
            migrated += 1

        if migrated:
            self._logger.info(f"Recorded the 4B variant on {migrated} Qwen3-VL encoder config(s)")


def build_migration(logger: Logger) -> Migration:
    return Migration(
        id="2026_09_16_add_qwen3_vl_encoder_variant",
        depends_on="2026_09_12_add_video_embeddings",
        callback=AddQwen3VLEncoderVariantCallback(logger=logger),
    )
