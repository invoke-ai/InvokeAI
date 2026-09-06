"""Guards for Z-Image's FP8-storage wiring in the single-file loader.

`_load_from_singlefile` is the only place Z-Image checkpoints reach the layerwise cast, and nothing
else in the suite observes it: deleting the `_apply_fp8_layerwise_casting` call leaves the whole
`tests/backend/model_manager` suite green, so the toggle can go back to being rendered-and-inert
with CI passing.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import diffusers
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ZImage_Config
from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageCheckpointModel
from invokeai.backend.model_manager.taxonomy import SubModelType, ZImageVariantType


class _TinyZImageTransformer(torch.nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        # `x_pad_token` is one of the loader's `valid_exact` keys, so it survives key filtering.
        self.x_pad_token = torch.nn.Parameter(torch.empty(2, 2))


def _prepare_loader(monkeypatch, tmp_path, state_dict: dict[str, torch.Tensor]):
    checkpoint_path = tmp_path / "z_image.safetensors"
    checkpoint_path.touch()
    config = Main_Checkpoint_ZImage_Config.model_construct(
        path=str(checkpoint_path), variant=ZImageVariantType.Turbo, fp8_storage=True
    )

    monkeypatch.setattr(diffusers, "ZImageTransformer2DModel", _TinyZImageTransformer, raising=False)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _path: state_dict)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.z_image.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.z_image.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )

    loader = object.__new__(ZImageCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    return loader, config


def test_single_file_loader_applies_fp8_layerwise_casting(monkeypatch, tmp_path) -> None:
    """The `fp8_storage` toggle has to reach the cast for Z-Image checkpoints at all."""
    cast_calls: list[tuple[object, object]] = []

    loader, config = _prepare_loader(monkeypatch, tmp_path, {"x_pad_token": torch.ones(2, 2)})
    loader._apply_fp8_layerwise_casting = lambda model, cfg, submodel: (
        cast_calls.append((cfg, submodel)),
        model,
    )[1]

    model = loader._load_from_singlefile(config)

    assert isinstance(model, _TinyZImageTransformer)
    assert torch.equal(model.x_pad_token, torch.ones(2, 2))
    assert cast_calls == [(config, SubModelType.Transformer)]


def test_state_dict_is_released_before_the_fp8_cast(monkeypatch, tmp_path) -> None:
    """Peak RAM must not overshoot the `make_room()` reservation.

    `load_state_dict(..., assign=True)` aliases every param to its state-dict tensor. If the dict is
    still holding those references when the FP8 cast runs, `param.data.to(float8)` allocates the fp8
    copy while the original is still reachable, so the model is briefly resident ~1.5x over — about
    17.4GB actual against an ~11.5GB reservation for Z-Image. Nothing in the loader reads `sd` after
    the load, so the dict must be empty by the time the cast starts.
    """
    state_dict = {"x_pad_token": torch.ones(2, 2)}
    observed_sd_len: list[int] = []

    loader, config = _prepare_loader(monkeypatch, tmp_path, state_dict)
    loader._apply_fp8_layerwise_casting = lambda model, _cfg, _submodel: (
        observed_sd_len.append(len(state_dict)),
        model,
    )[1]

    loader._load_from_singlefile(config)

    assert observed_sd_len == [0]


def test_assign_true_really_aliases_the_state_dict() -> None:
    """The premise of the test above: without clearing, the originals stay alive through `sd`.

    If a future torch release stopped aliasing under `assign=True`, `sd.clear()` would become dead
    weight rather than a fix, and this is the test that says so.
    """
    model = _TinyZImageTransformer()
    sd = {"x_pad_token": torch.ones(2, 2)}

    model.load_state_dict(sd, assign=True)

    assert model.x_pad_token.data_ptr() == sd["x_pad_token"].data_ptr()

    # And the cast leaves the state dict's copy behind at the original dtype.
    model.x_pad_token.data = model.x_pad_token.data.to(torch.float16)
    assert sd["x_pad_token"].dtype == torch.float32
