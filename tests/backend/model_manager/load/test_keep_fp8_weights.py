"""Whether a checkpoint's fp8 weights stay packed or get folded into bf16.

Two independent consumers want them packed: the fp8 matmul, and FP8 Storage. The decision used to
live in five loaders and asked only about the matmul, so a model with FP8 Storage on had its scales
folded in and the result re-quantized by the layerwise cast -- to *unscaled* fp8, which has no scale
to apply. Measured on FLUX.2 Klein 4B that loses ~3% of the weights to underflow, for exactly the
byte count the file already had.
"""

from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.taxonomy import SubModelType


def _loader() -> ModelLoader:
    loader = object.__new__(ModelLoader)
    loader._torch_device = torch.device("cuda")
    loader._logger = MagicMock()
    return loader


def _config(fp8_storage: bool | None) -> Main_Checkpoint_FLUX_Config:
    return Main_Checkpoint_FLUX_Config.model_construct(
        path="transformer.safetensors",
        default_settings=MainModelDefaultSettings(fp8_storage=fp8_storage),
    )


@pytest.mark.parametrize(
    "matmul, fp8_storage, device_supports_storage, expected",
    [
        # The matmul runs on fp8 weights directly; it never needs the setting.
        (True, None, False, True),
        # The case this exists for: storage asked of the model, so the file's own exact scale is
        # kept rather than folded and re-rounded without one.
        (False, True, True, True),
        # Nobody asked: folding is right, because staying packed would dequantize per forward.
        (False, None, True, False),
        # Device capability decides first -- a device that cannot hold fp8 gets the folded path
        # however the model is configured.
        (False, True, False, False),
    ],
)
def test_fp8_weights_are_kept_for_either_consumer(
    monkeypatch: pytest.MonkeyPatch,
    matmul: bool,
    fp8_storage: bool | None,
    device_supports_storage: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: matmul)
    monkeypatch.setattr(
        load_default, "_device_supports_fp8_storage", lambda _device, _logger=None: device_supports_storage
    )

    kept = _loader()._keep_fp8_weights(_config(fp8_storage), SubModelType.Transformer)

    assert kept is expected


def test_a_text_encoder_submodel_never_keeps_them_for_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP8 Storage excludes text encoders by design -- fp8 rounding costs text quality -- so the
    per-model setting must not reach them through this helper either. The encoder loaders that do
    keep fp8 ask the device directly, which is a different question."""
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: False)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda _device, _logger=None: True)

    kept = _loader()._keep_fp8_weights(_config(True), SubModelType.TextEncoder)

    assert kept is False
