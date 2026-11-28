from pathlib import Path

import torch
import yaml

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig, InvokeAIAppConfig
from invokeai.app.services.model_manager.model_manager_default import _resolve_keep_ram_copy_of_weights


def test_keep_ram_copy_is_automatic_when_unset() -> None:
    config = InvokeAIAppConfig()
    assert config.keep_ram_copy_of_weights is None
    assert "keep_ram_copy_of_weights" not in config.model_fields_set


def test_keep_ram_copy_explicit_values_are_preserved_in_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "invokeai.yaml"
    for value in (True, False):
        config = DefaultInvokeAIAppConfig(keep_ram_copy_of_weights=value)
        config.write_file(config_path)
        loaded = InvokeAIAppConfig.model_validate(yaml.safe_load(config_path.read_text()))
        assert loaded.keep_ram_copy_of_weights is value
        assert "keep_ram_copy_of_weights" in loaded.model_fields_set


def test_automatic_copy_policy_only_disables_native_integrated_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        "invokeai.app.services.model_manager.model_manager_default.device_is_integrated", lambda _: True
    )
    assert _resolve_keep_ram_copy_of_weights(None, torch.device("cuda:0")) is False
    assert _resolve_keep_ram_copy_of_weights(True, torch.device("cuda:0")) is True
    assert _resolve_keep_ram_copy_of_weights(False, torch.device("cuda:0")) is False
    assert _resolve_keep_ram_copy_of_weights(None, torch.device("xpu:0")) is True
    assert _resolve_keep_ram_copy_of_weights(None, torch.device("cpu")) is True


def test_automatic_copy_policy_keeps_unknown_and_discrete_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        "invokeai.app.services.model_manager.model_manager_default.device_is_integrated", lambda _: None
    )
    assert _resolve_keep_ram_copy_of_weights(None, torch.device("cuda:0")) is True
    monkeypatch.setattr(
        "invokeai.app.services.model_manager.model_manager_default.device_is_integrated", lambda _: False
    )
    assert _resolve_keep_ram_copy_of_weights(None, torch.device("cuda:0")) is True


def test_automatic_copy_policy_keeps_rocm_and_explicit_values(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(
        "invokeai.app.services.model_manager.model_manager_default.device_is_integrated", lambda _: True
    )
    assert _resolve_keep_ram_copy_of_weights(None, torch.device("cuda:0")) is True
    assert _resolve_keep_ram_copy_of_weights(False, torch.device("cuda:0")) is False
    assert _resolve_keep_ram_copy_of_weights(True, torch.device("cuda:0")) is True
