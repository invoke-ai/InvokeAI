from pathlib import Path
from types import SimpleNamespace

import pytest

from invokeai.app.services.model_records.missing_models import find_missing_models, is_model_missing
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat


def _config(key: str, path: str, base=BaseModelType.StableDiffusionXL, format=ModelFormat.Diffusers):
    return SimpleNamespace(key=key, path=path, base=base, format=format)


def test_a_model_is_missing_when_its_path_does_not_exist(tmp_path: Path) -> None:
    (tmp_path / "present").mkdir()

    assert not is_model_missing(_config("a", "present"), tmp_path)
    assert is_model_missing(_config("b", "absent"), tmp_path)


def test_external_models_have_no_files_to_miss(tmp_path: Path) -> None:
    assert not is_model_missing(_config("a", "nowhere", base=BaseModelType.External), tmp_path)
    assert not is_model_missing(_config("b", "nowhere", format=ModelFormat.ExternalApi), tmp_path)


@pytest.mark.parametrize("error", [PermissionError("denied"), RuntimeError("Symlink loop")])
def test_an_unreadable_path_is_unavailable_and_does_not_stop_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    (tmp_path / "healthy").mkdir()
    real_resolve = Path.resolve

    def resolve(self: Path, *args, **kwargs) -> Path:
        if self.name == "locked":
            raise error
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    configs = [_config("locked", "locked"), _config("healthy", "healthy"), _config("gone", "gone")]

    assert [config.key for config in find_missing_models(configs, tmp_path)] == ["locked", "gone"]
