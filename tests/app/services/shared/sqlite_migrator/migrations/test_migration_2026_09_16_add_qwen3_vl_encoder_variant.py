"""The Qwen3-VL encoder variant backfill.

Without it, every Qwen3-VL encoder installed before the variant field existed fails validation on
read and is *skipped* with a log line — the encoder disappears from the model list and the Krea-2
graph that used it stops compiling, with nothing in the UI to say why.
"""

import json
import sqlite3
from logging import getLogger

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_16_add_qwen3_vl_encoder_variant import (
    build_migration,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, Qwen3VLVariantType


@pytest.fixture
def cursor() -> sqlite3.Cursor:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE models (id TEXT PRIMARY KEY, config TEXT NOT NULL);")
    return connection.cursor()


def _insert(cursor: sqlite3.Cursor, model_id: str, config: dict) -> None:
    cursor.execute("INSERT INTO models (id, config) VALUES (?, ?);", (model_id, json.dumps(config)))


def _config(cursor: sqlite3.Cursor, model_id: str) -> dict:
    cursor.execute("SELECT config FROM models WHERE id = ?;", (model_id,))
    return json.loads(cursor.fetchone()[0])


def _run(cursor: sqlite3.Cursor) -> None:
    build_migration(getLogger(__name__)).callback(cursor)  # type: ignore[misc]


def test_a_legacy_encoder_becomes_the_4b(cursor: sqlite3.Cursor) -> None:
    # The only shape the config accepted before Ideogram 4's 8B joined it.
    _insert(
        cursor,
        "legacy",
        {
            "type": ModelType.Qwen3VLEncoder.value,
            "base": BaseModelType.Any.value,
            "name": "qwen3vl_4b_fp8_scaled",
        },
    )

    _run(cursor)

    assert _config(cursor, "legacy")["variant"] == Qwen3VLVariantType.Qwen3VL_4B.value


def test_a_recorded_variant_is_left_alone(cursor: sqlite3.Cursor) -> None:
    _insert(
        cursor,
        "already-8b",
        {
            "type": ModelType.Qwen3VLEncoder.value,
            "base": BaseModelType.Any.value,
            "name": "qwen3vl_8b_fp8_scaled",
            "variant": Qwen3VLVariantType.Qwen3VL_8B.value,
        },
    )

    _run(cursor)

    assert _config(cursor, "already-8b")["variant"] == Qwen3VLVariantType.Qwen3VL_8B.value


def test_other_model_types_are_untouched(cursor: sqlite3.Cursor) -> None:
    # `variant` means something different on every type, so a blanket backfill would corrupt them.
    _insert(cursor, "main", {"type": ModelType.Main.value, "name": "krea2"})
    _insert(cursor, "qwen3", {"type": ModelType.Qwen3Encoder.value, "name": "qwen_3_4b"})

    _run(cursor)

    assert "variant" not in _config(cursor, "main")
    assert "variant" not in _config(cursor, "qwen3")


def test_minimax_h3s_encoder_is_left_alone(cursor: sqlite3.Cursor) -> None:
    """It shares the model type and is not one of the two that gained the field.

    Its config class is a truncated Qwen3-VL-32B with no `variant` at all, so stamping `4b` on it
    would store a size that is simply untrue — inert only until that class grows a variant of its
    own, at which point the wrong architecture would be built from it.
    """
    _insert(
        cursor,
        "minimax",
        {
            "type": ModelType.Qwen3VLEncoder.value,
            "base": BaseModelType.MiniMaxH3.value,
            "name": "qwen3vl_32b_minimax_h3_fp8",
        },
    )

    _run(cursor)

    assert "variant" not in _config(cursor, "minimax")
