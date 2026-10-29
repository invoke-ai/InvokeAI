"""The Anima variant backfill.

Without it every Anima model installed before the variant field existed fails validation on read and is
*skipped* -- the models disappear from the model list with nothing in the UI to say why.
"""

import json
import sqlite3
from logging import getLogger
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_10_01_add_anima_variant import (
    build_migration,
)
from invokeai.backend.model_manager.taxonomy import AnimaVariantType, BaseModelType, ModelFormat, ModelType


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


def _run(cursor: sqlite3.Cursor, models_path: Path) -> None:
    app_config = SimpleNamespace(models_path=models_path)
    build_migration(app_config, getLogger(__name__)).callback(cursor)  # type: ignore[arg-type,misc]


def _anima_main(path: str) -> dict:
    return {
        "type": ModelType.Main.value,
        "base": BaseModelType.Anima.value,
        "format": ModelFormat.Checkpoint.value,
        "name": Path(path).stem,
        "path": path,
    }


def test_a_legacy_anima_model_becomes_qwen3(cursor: sqlite3.Cursor, tmp_path: Path) -> None:
    checkpoint = tmp_path / "anima-base-v1.0.safetensors"
    save_file({"net.blocks.0.mlp.layer1.weight": torch.zeros(1)}, checkpoint)
    _insert(cursor, "base", _anima_main(str(checkpoint)))

    _run(cursor, tmp_path)

    assert _config(cursor, "base")["variant"] == AnimaVariantType.Qwen3.value


def test_an_installed_3_8b_bundle_becomes_qwen35(cursor: sqlite3.Cursor, tmp_path: Path) -> None:
    # It identified as a plain Anima before the variant existed; its header still says what it is.
    # A relative path resolves against the models directory, as the record service resolves it.
    (tmp_path / "anima").mkdir()
    save_file(
        {"net.blocks.0.mlp.layer1.weight": torch.zeros(1), "net.anima_v2_connector.query_tokens": torch.zeros(1)},
        tmp_path / "anima" / "Anima-3.8B-v1.1.safetensors",
    )
    _insert(cursor, "bundle", _anima_main("anima/Anima-3.8B-v1.1.safetensors"))

    _run(cursor, tmp_path)

    assert _config(cursor, "bundle")["variant"] == AnimaVariantType.Qwen35.value


def test_a_missing_file_becomes_qwen3(cursor: sqlite3.Cursor, tmp_path: Path) -> None:
    _insert(cursor, "gone", _anima_main(str(tmp_path / "gone.safetensors")))

    _run(cursor, tmp_path)

    assert _config(cursor, "gone")["variant"] == AnimaVariantType.Qwen3.value


def test_a_recorded_variant_is_left_alone(cursor: sqlite3.Cursor, tmp_path: Path) -> None:
    _insert(cursor, "done", {**_anima_main(str(tmp_path / "x.safetensors")), "variant": AnimaVariantType.Qwen35.value})

    _run(cursor, tmp_path)

    assert _config(cursor, "done")["variant"] == AnimaVariantType.Qwen35.value


def test_other_models_are_untouched(cursor: sqlite3.Cursor, tmp_path: Path) -> None:
    # Anima's LLLite adapters share the base and carry no variant; other mains mean something else by it.
    _insert(
        cursor,
        "lllite",
        {"type": ModelType.ControlNet.value, "base": BaseModelType.Anima.value, "format": "checkpoint"},
    )
    _insert(cursor, "krea", {"type": ModelType.Main.value, "base": BaseModelType.Krea2.value, "format": "checkpoint"})

    _run(cursor, tmp_path)

    assert "variant" not in _config(cursor, "lllite")
    assert "variant" not in _config(cursor, "krea")
