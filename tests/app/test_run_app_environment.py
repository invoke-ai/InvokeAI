"""The console-script entry module pins process-wide environment defaults before any heavy import."""

import importlib
import os

import pytest


@pytest.fixture
def reload_run_app(monkeypatch: pytest.MonkeyPatch):
    """Re-execute ``invokeai.app.run_app``'s module body under a patched environment."""
    import invokeai.app.run_app as run_app

    def reload() -> None:
        importlib.reload(run_app)

    yield reload
    importlib.reload(run_app)


@pytest.mark.parametrize("name, default", [("MIOPEN_FIND_MODE", "FAST"), ("TOKENIZERS_PARALLELISM", "false")])
def test_run_app_sets_environment_defaults(name: str, default: str, monkeypatch: pytest.MonkeyPatch, reload_run_app):
    monkeypatch.delenv(name, raising=False)
    reload_run_app()
    assert os.environ[name] == default


@pytest.mark.parametrize("name", ["MIOPEN_FIND_MODE", "TOKENIZERS_PARALLELISM"])
def test_run_app_keeps_an_explicit_environment_value(name: str, monkeypatch: pytest.MonkeyPatch, reload_run_app):
    monkeypatch.setenv(name, "user-choice")
    reload_run_app()
    assert os.environ[name] == "user-choice"
