"""Tests for `ModelLoader._get_execution_device` — the helper that forces a model onto the CPU
when its config requests `cpu_only`.

A VAE (or text encoder) configured with `cpu_only=True` must load onto the CPU so its weights
never occupy VRAM. The loader signals this by returning `torch.device("cpu")` from
`_get_execution_device`, which is then passed to `ModelCache.put(..., execution_device=...)`.
"""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.taxonomy import SubModelType


def _loader() -> ModelLoader:
    # `_get_execution_device` only reads the config, so an uninitialized loader is sufficient.
    return ModelLoader.__new__(ModelLoader)


def _vae_config(cpu_only: Optional[bool]) -> SimpleNamespace:
    # Mirrors the relevant surface of a standalone VAE config: a `cpu_only` field and no
    # `default_settings` (VAE configs do not carry default settings).
    return SimpleNamespace(cpu_only=cpu_only, default_settings=None)


def test_vae_cpu_only_true_returns_cpu():
    assert _loader()._get_execution_device(_vae_config(cpu_only=True), None) == torch.device("cpu")


def test_vae_cpu_only_false_or_unset_returns_none():
    # Falsy values must not force CPU execution — the cache falls back to its default device.
    assert _loader()._get_execution_device(_vae_config(cpu_only=False), None) is None
    assert _loader()._get_execution_device(_vae_config(cpu_only=None), None) is None


def test_vae_cpu_only_applies_regardless_of_submodel_type():
    # The VAE is loaded as a standalone model (submodel_type=None), but the standalone branch
    # must not depend on the submodel type either way.
    loader = _loader()
    assert loader._get_execution_device(_vae_config(cpu_only=True), SubModelType.VAE) == torch.device("cpu")


def test_config_without_cpu_only_attr_returns_none():
    # A config type that has neither `cpu_only` nor `default_settings` must be left on the
    # cache default (return None), not crash.
    assert _loader()._get_execution_device(SimpleNamespace(), None) is None


@pytest.mark.parametrize("submodel_type", [SubModelType.TextEncoder, SubModelType.VisionEncoder])
def test_a_main_models_cpu_only_setting_moves_its_encoders_to_the_cpu(submodel_type):
    # The vision encoder runs inside the text encoder's forward (Krea-2 and Qwen-Image-2.1 edits), so `cpu_only`
    # has to put both on the same device.
    config = SimpleNamespace(default_settings=SimpleNamespace(cpu_only=True))

    assert _loader()._get_execution_device(config, submodel_type) == torch.device("cpu")
    assert _loader()._get_execution_device(config, SubModelType.Transformer) is None
