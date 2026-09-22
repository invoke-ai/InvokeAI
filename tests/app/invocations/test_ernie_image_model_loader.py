"""Tests for the ERNIE-Image prompt-enhancer availability gate.

`use_prompt_enhancer` defaults to true, so whatever this gate reports is the default experience.
`GenericDiffusersLoader.get_hf_load_class` resolves each submodel's class from `model_index.json`
and raises "the ... submodel is not available for this model" on a missing key -- so gating on the
directory alone would let a partial install pass and then hard-fail the whole generation at load
time. Both the LM and its tokenizer must be present *and* declared.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from invokeai.app.invocations.ernie_image_model_loader import ErnieImageModelLoaderInvocation

_FULL_INDEX = {
    "_class_name": "ErnieImagePipeline",
    "pe": ["transformers", "Ministral3ForCausalLM"],
    "pe_tokenizer": ["transformers", "TokenizersBackend"],
    "text_encoder": ["transformers", "Mistral3Model"],
    "tokenizer": ["transformers", "TokenizersBackend"],
    "transformer": ["diffusers", "ErnieImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def _make_pipeline(models_path: Path, *, subdirs: tuple[str, ...], index: dict | None) -> None:
    """Lay out a pipeline dir the way an install would, under a *relative* record-store path."""
    model_dir = models_path / "main" / "ernie-image" / "ERNIE-Image"
    for name in ("transformer", "vae", "text_encoder", "tokenizer", *subdirs):
        (model_dir / name).mkdir(parents=True, exist_ok=True)
    if index is not None:
        (model_dir / "model_index.json").write_text(json.dumps(index), encoding="utf-8")


def _context(models_path: Path) -> SimpleNamespace:
    # The record store holds paths relative to `models_path` for Invoke-managed models, and nothing
    # has absolutized this one yet -- only the loader does that, and it runs after this check.
    config = SimpleNamespace(path="main/ernie-image/ERNIE-Image")
    return SimpleNamespace(
        models=SimpleNamespace(get_config=lambda _model: config),
        config=SimpleNamespace(get=lambda: SimpleNamespace(models_path=models_path)),
    )


def _gate(models_path: Path) -> bool:
    invocation = ErnieImageModelLoaderInvocation.model_construct(model=None)
    return invocation._pipeline_has_prompt_enhancer(_context(models_path))  # type: ignore[arg-type]


def test_accepts_a_complete_prompt_enhancer(tmp_path: Path) -> None:
    _make_pipeline(tmp_path, subdirs=("pe", "pe_tokenizer"), index=_FULL_INDEX)

    assert _gate(tmp_path) is True


def test_relative_record_store_path_is_resolved_against_models_path(tmp_path: Path, monkeypatch) -> None:
    """The gate must not probe the server process's CWD.

    `config.path` is relative for Invoke-managed models, so a bare `Path(config.path)` would resolve
    against whatever directory the server happens to be running in and report "no enhancer" for
    every normally installed model.
    """
    _make_pipeline(tmp_path, subdirs=("pe", "pe_tokenizer"), index=_FULL_INDEX)
    monkeypatch.chdir(tmp_path.parent)

    assert _gate(tmp_path) is True


@pytest.mark.parametrize("missing", ["pe", "pe_tokenizer"])
def test_rejects_a_missing_submodel_directory(tmp_path: Path, missing: str) -> None:
    present = tuple(name for name in ("pe", "pe_tokenizer") if name != missing)
    _make_pipeline(tmp_path, subdirs=present, index=_FULL_INDEX)

    assert _gate(tmp_path) is False


@pytest.mark.parametrize("missing", ["pe", "pe_tokenizer"])
def test_rejects_a_submodel_absent_from_model_index(tmp_path: Path, missing: str) -> None:
    """Directory present but undeclared: the loader would raise, so the gate must say no."""
    index = {k: v for k, v in _FULL_INDEX.items() if k != missing}
    _make_pipeline(tmp_path, subdirs=("pe", "pe_tokenizer"), index=index)

    assert _gate(tmp_path) is False


def test_rejects_a_missing_or_unreadable_model_index(tmp_path: Path) -> None:
    _make_pipeline(tmp_path, subdirs=("pe", "pe_tokenizer"), index=None)
    assert _gate(tmp_path) is False

    model_dir = tmp_path / "main" / "ernie-image" / "ERNIE-Image"
    (model_dir / "model_index.json").write_text("{ not json", encoding="utf-8")
    assert _gate(tmp_path) is False
